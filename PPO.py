
import numpy as np
from typing import List, Dict, Tuple
# ============================================================
# PPO-Clip (score-function gradient version, no autodiff)
#  - 연속: 가우시안(대각) 정책 (μ, σ)
#  - 이산: 소프트맥스 정책 (capacity, servers)
#  - 배치로 old_logp 저장 → 여러 epoch로 클립 비율 기반 업데이트
# ============================================================
class PPOMultiPolicy:
    def __init__(self,
                bounds: Dict[str, Tuple[float, float]],
                cont_init: Dict[str, float],
                capacity_choices: List[int],
                servers_choices: List[int],
                sigma_min=0.02, sigma_max=3.0,
                seed=0):
        # 랜덤 시드 고정 (재현성)
        self.rng = np.random.default_rng(seed)
        
        # --- 연속 행동(Continuous Action) 파라미터 ---
        self.cont_names = ["interarrival","service_rate","size_mu","size_sigma"] # 연속 행동의 이름들
        self.bounds = bounds  # 각 연속 행동의 최솟값/최댓값
        # 가우시안 분포의 평균(μ). 초기값으로 시작.
        self.mu = np.array([cont_init[k] for k in self.cont_names], dtype=float)
        # 가우시안 분포의 표준편차(σ). 범위의 1/4로 초기화하여 적당한 탐험을 유도.
        self.sigma = np.array([(bounds[k][1]-bounds[k][0])*0.25 for k in self.cont_names], dtype=float)
        self.sigma_min = sigma_min; self.sigma_max = sigma_max # σ가 너무 작거나 커지지 않도록 제한

        # --- 이산 행동(Discrete Action) 파라미터 ---
        self.capacity_choices = list(capacity_choices) # 선택 가능한 용량 목록
        self.servers_choices  = list(servers_choices) # 선택 가능한 서버 수 목록
        # 각 선택지의 확률을 결정하는 logit 값. 0으로 초기화 (모든 선택 확률이 동일).
        self.logits_capacity = np.zeros(len(self.capacity_choices), dtype=float)
        self.logits_servers  = np.zeros(len(self.servers_choices), dtype=float)

    @staticmethod
    def _softmax(logits):
        z = logits - np.max(logits)  # 오버플로우 방지를 위한 안정화 트릭
        e = np.exp(z)
        return e / (np.sum(e) + 1e-12) # 0으로 나누는 것을 방지
    
    def sample_batch(self, batch_size:int):
        # 1. 연속 행동 샘플링 (가우시안 분포)
        xs = self.rng.normal(self.mu, self.sigma, size=(batch_size, len(self.mu)))
        for i, k in enumerate(self.cont_names): # 경계를 벗어나지 않도록 clip
            lo, hi = self.bounds[k]
            xs[:, i] = np.clip(xs[:, i], lo, hi)

        # 2. 이산 행동 샘플링 (카테고리 분포)
        pc = self._softmax(self.logits_capacity) # 용량 확률
        ps = self._softmax(self.logits_servers)  # 서버 수 확률
        cap_idx = self.rng.choice(len(self.capacity_choices), size=batch_size, p=pc)
        srv_idx = self.rng.choice(len(self.servers_choices),  size=batch_size, p=ps)
        caps = np.array([self.capacity_choices[i] for i in cap_idx], dtype=int)
        srvs = np.array([self.servers_choices[i]  for i in srv_idx], dtype=int)

        # 3. 샘플링 시점의 로그 확률(old_logp) 계산 및 저장
        old_logp_cont = self.gaussian_log_prob(xs, self.mu, self.sigma)
        old_logp_cap  = np.log(pc[cap_idx] + 1e-12)
        old_logp_srv  = np.log(ps[srv_idx] + 1e-12)
        old_logp_total = old_logp_cont + old_logp_cap + old_logp_srv

        # 4. PPO 업데이트에 필요한 모든 정보를 cache에 저장
        cache = dict(xs=xs, cap_idx=cap_idx, srv_idx=srv_idx, old_logp_cont=old_logp_cont, 
                     old_logp_cap=old_logp_cap, old_logp_srv=old_logp_srv, old_logp_total=old_logp_total, 
                     pc=pc, ps=ps)
        return xs, caps, srvs, cache

    # 로그 확률 계산 함수
    @staticmethod
    def gaussian_log_prob(xs, mu, sigma):
        # xs: (B,D), mu/sigma: (D,)
        xs = np.asarray(xs, float)
        mu = np.asarray(mu, float)
        sigma = np.asarray(sigma, float)
        var = sigma**2 + 1e-12
        # per-dim logN then sum dims
        return -0.5*np.sum(((xs-mu)**2)/var + 2*np.log(sigma+1e-12) + np.log(2*np.pi), axis=1)

    def current_log_probs(self, xs, cap_idx, srv_idx):
        pc = self._softmax(self.logits_capacity)
        ps = self._softmax(self.logits_servers)
        lp_cont = self.gaussian_log_prob(xs, self.mu, self.sigma)
        lp_cap  = np.log(pc[cap_idx] + 1e-12)
        lp_srv  = np.log(ps[srv_idx] + 1e-12)
        return lp_cont, lp_cap, lp_srv, pc, ps

    # ---------- gradients of log-probs ----------
    def grad_logp_cont(self, xs):
        # returns per-sample grads wrt mu, sigma
        xs = np.asarray(xs, float)
        mu, sigma = self.mu, self.sigma
        var = sigma**2 + 1e-12
        grad_mu  = (xs - mu) / var            # shape (B,D)
        grad_sig = ((xs - mu)**2 - var) / (sigma**3 + 1e-12)  # (B,D)
        return grad_mu, grad_sig

    @staticmethod
    def grad_logp_categorical(indices, probs, n_class):
        # per-sample grad wrt logits: one_hot(idx) - probs
        grad = - np.tile(probs, (len(indices), 1))
        grad[np.arange(len(indices)), indices] += 1.0
        return grad  # shape (B, K)

    # ---------- PPO update (score-function approx) ----------
    def update_ppo(self, cache, rewards, clip_eps=0.2, epochs=5,
                   lr_mu=0.15, lr_sigma=0.05, lr_disc=0.1):
        #0. 필요한 데이터 수집
        xs       = cache["xs"]
        cap_idx  = cache["cap_idx"]
        srv_idx  = cache["srv_idx"]
        old_tot  = cache["old_logp_total"]

        #1. Advandate 계산
        r = np.asarray(rewards, float)
        adv = r - r.mean() #보상에서 평균을 빼서 Advantage 계산
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) #안정성 위해 표준화

        #2. 동일한 데이터로 여러 번 파라미터 업데이트
        for _ in range(epochs):
            #3. 현재 정책에서의 로그 확률과 비율 계산
            lp_cont, lp_cap, lp_srv, pc, ps = self.current_log_probs(xs, cap_idx, srv_idx)
            new_tot = lp_cont + lp_cap + lp_srv
            ratio = np.exp(new_tot - old_tot)  #비율 = 현재 확률 / 과거 확률

            #4. 비율 클리핑
            #정책 너무 급변하지 않도록 범위 제힌
            w = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)  # (B,)

            #5. 그래디언트 계산
            #연속형 파라미터 계산
            g_mu_samp, g_sig_samp = self.grad_logp_cont(xs)   # (B,D)
            # score-function with clipped weight
            g_mu  = np.mean(g_mu_samp  * w[:,None] * adv[:,None], axis=0)
            g_sig = np.mean(g_sig_samp * w[:,None] * adv[:,None], axis=0)

            # update μ, σ
            self.mu    = np.clip(self.mu    + lr_mu   * g_mu,
                                 [self.bounds[k][0] for k in self.cont_names],
                                 [self.bounds[k][1] for k in self.cont_names])
            self.sigma = np.clip(self.sigma + lr_sigma* g_sig,
                                 self.sigma_min, self.sigma_max)

            #이산형 파라미터 계산
            g_cap_samp = self.grad_logp_categorical(cap_idx, pc, len(pc))  # (B,Kc)
            g_srv_samp = self.grad_logp_categorical(srv_idx, ps, len(ps))  # (B,Ks)
            g_cap = np.mean(g_cap_samp * w[:,None] * adv[:,None], axis=0)
            g_srv = np.mean(g_srv_samp * w[:,None] * adv[:,None], axis=0)

            #파라미터가 유효한 범위 내에 있도록 클리핑
            self.logits_capacity += lr_disc * g_cap
            self.logits_servers  += lr_disc * g_srv

    def summary(self):
        return dict(mu=self.mu.copy(), sigma=self.sigma.copy(),
                    pc=self._softmax(self.logits_capacity),
                    ps=self._softmax(self.logits_servers))

