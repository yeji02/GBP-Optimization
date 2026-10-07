"""
PPO-Clip Actor-Critic (PyTorch) - 연속 + 이산 혼합 행동
 - 연속: ServiceRate, ReleaseInterval  → 대각 가우시안 (정규화 공간 [-1, 1])
 - 이산: ActiveServers(1..N), Dispatch(INDEX/HEALTH) → 카테고리 분포
 - 연속 행동은 선택된 ActiveServers에 조건부 (parameterized action)
 - 상태 s_t를 입력받는 정책 π(a|s): 공정 상태가 바뀌면 파라미터도 바뀜 (동적 최적화)
 - GAE(λ)로 Advantage 추정, 시간 제한 종료(truncation)는 V(s_T)로 bootstrap
"""
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from xdevs_Job import Control
from gbp_env import N_SERVERS, SERVICE_RATE_BOUNDS, RELEASE_BOUNDS


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class ActorCritic(nn.Module):
    """
    조건부 혼합 행동 정책: 이산 행동(가동 대수·디스패칭)을 먼저 정하고,
    연속 행동(속도·투입간격)은 '선택된 가동 대수'를 입력으로 받아 결정
    → "2대면 고속, 3대면 저속" 같은 조합을 하나의 정책이 표현 가능
    """
    def __init__(self, obs_dim, n_cont=2, n_srv=N_SERVERS, n_dsp=2, h=64):
        super().__init__()
        self.n_srv = n_srv
        self.trunk = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.disc_head = nn.Linear(h, n_srv + n_dsp)
        self.cont_head = nn.Sequential(nn.Linear(h + n_srv, h), nn.Tanh(), nn.Linear(h, n_cont))
        self.critic = mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((n_cont,), -0.7))

    def disc_dists(self, obs):
        z = self.trunk(obs)
        logits = self.disc_head(z)
        return z, Categorical(logits=logits[:, :self.n_srv]), Categorical(logits=logits[:, self.n_srv:])

    def cont_dist(self, z, srv):
        onehot = nn.functional.one_hot(srv, self.n_srv).float()
        mu = self.cont_head(torch.cat([z, onehot], -1))
        return Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class PPOAgent:
    def __init__(self, obs_dim, lr=3e-4, gamma=0.95, lam=0.9, clip_eps=0.2, epochs=10,
                 minibatch=128, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, seed=0):
        torch.manual_seed(seed)
        self.net = ActorCritic(obs_dim)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip_eps = gamma, lam, clip_eps
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        z, ds, dd = self.net.disc_dists(o)
        srv = ds.probs.argmax(-1) if deterministic else ds.sample()
        dsp = dd.probs.argmax(-1) if deterministic else dd.sample()
        dc = self.net.cont_dist(z, srv)
        cont = dc.mean if deterministic else dc.sample()
        logp = dc.log_prob(cont).sum(-1) + ds.log_prob(srv) + dd.log_prob(dsp)
        a = dict(cont=cont[0].numpy(), srv=int(srv[0]), dsp=int(dsp[0]))
        return a, float(logp[0]), float(self.net.value(o)[0])

    @torch.no_grad()
    def value(self, obs):
        return float(self.net.value(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0])

    @staticmethod
    def to_control(a: Dict) -> Control:
        u = np.clip(a["cont"], -1.0, 1.0)
        scale = lambda x, b: b[0] + (x + 1.0) * 0.5 * (b[1] - b[0])
        return Control(service_rate=scale(u[0], SERVICE_RATE_BOUNDS), active_servers=a["srv"] + 1,
                       release_interval=scale(u[1], RELEASE_BOUNDS), dispatch=a["dsp"])

    # ------------------------------------------------------------------
    def gae(self, rewards, values, last_value):
        adv = np.zeros(len(rewards), dtype=np.float32)
        g, nxt = 0.0, last_value
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * nxt - values[t]
            g = delta + self.gamma * self.lam * g
            adv[t], nxt = g, values[t]
        return adv, adv + np.asarray(values, dtype=np.float32)

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        cont = torch.as_tensor(batch["cont"], dtype=torch.float32)
        srv = torch.as_tensor(batch["srv"], dtype=torch.long)
        dsp = torch.as_tensor(batch["dsp"], dtype=torch.long)
        old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32)
        adv = torch.as_tensor(batch["adv"], dtype=torch.float32)
        ret = torch.as_tensor(batch["ret"], dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        stats = dict(pi_loss=0.0, v_loss=0.0, entropy=0.0, clip_frac=0.0)
        n_mb = 0
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.minibatch):
                idx = perm[s:s + self.minibatch]
                z, ds, dd = self.net.disc_dists(obs[idx])
                dc = self.net.cont_dist(z, srv[idx])
                logp = dc.log_prob(cont[idx]).sum(-1) + ds.log_prob(srv[idx]) + dd.log_prob(dsp[idx])
                ratio = (logp - old_logp[idx]).exp()
                s1 = ratio * adv[idx]
                s2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv[idx]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = (self.net.value(obs[idx]) - ret[idx]).pow(2).mean()
                ent = (dc.entropy().sum(-1) + ds.entropy() + dd.entropy()).mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                stats["pi_loss"] += pi_loss.item(); stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["clip_frac"] += ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                n_mb += 1
        return {k: v / n_mb for k, v in stats.items()}

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        return self
