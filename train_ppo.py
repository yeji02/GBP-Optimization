"""
PPO 학습 스크립트
  python train_ppo.py --reward tradeoff      # v2: 처리량·대기·WIP·자원비용·제약위반 trade-off 보상
  python train_ppo.py --reward performance   # v1: 성능 지표만 보상 (자원 상한 수렴 현상 재현용)
학습 시나리오는 Normal / Demand Surge / Machine Degradation을 매 에피소드 무작위로 섞음(시점·크기도 랜덤)
"""
import argparse, json, os, time
import numpy as np

from gbp_env import GBPEnv, REWARD_PRESETS, OBS_DIM, config_dict
from PPO import PPOAgent

REWARD_SCALE = 0.2   # 가치함수 학습 안정화를 위한 보상 스케일링 (정책 최적해에는 영향 없음)


def run_episode(env, agent, seed, deterministic=False):
    obs = env.reset(seed=seed)
    traj = dict(obs=[], cont=[], srv=[], dsp=[], logp=[], val=[], rew=[])
    done, info = False, {}
    acts = []
    while not done:
        a, logp, v = agent.act(obs, deterministic)
        ctrl = agent.to_control(a)
        nobs, r, done, info = env.step(ctrl)
        for k, x in zip(("obs", "cont", "srv", "dsp", "logp", "val", "rew"),
                        (obs, a["cont"], a["srv"], a["dsp"], logp, v, r * REWARD_SCALE)):
            traj[k].append(x)
        acts.append((ctrl.active_servers, ctrl.service_rate, ctrl.release_interval, ctrl.dispatch))
        obs = nobs
    traj["last_value"] = agent.value(obs)   # 시간 제한 종료 → bootstrap
    info["mean_servers"], info["mean_rate"], info["mean_release"], info["health_dispatch"] = \
        map(float, np.mean(acts, axis=0))
    info["scenario"] = env.sc.name
    return traj, info


def train(reward="tradeoff", updates=250, episodes_per_update=8, seed=0, out_dir="runs", **ppo_kw):
    rc = REWARD_PRESETS[reward]
    run_dir = os.path.join(out_dir, f"ppo_{reward}")
    os.makedirs(run_dir, exist_ok=True)
    env = GBPEnv("mixed", reward=rc, seed=seed)
    agent = PPOAgent(OBS_DIM, seed=seed, **ppo_kw)
    history = []
    t0 = time.time()
    ep_seed = seed * 1_000_000

    for u in range(1, updates + 1):
        keys = ("obs", "cont", "srv", "dsp", "logp")
        batch = {k: [] for k in keys + ("adv", "ret")}
        infos = []
        for _ in range(episodes_per_update):
            ep_seed += 1
            traj, info = run_episode(env, agent, ep_seed)
            adv, ret = agent.gae(traj["rew"], traj["val"], traj["last_value"])
            for k in keys:
                batch[k].extend(traj[k])
            batch["adv"].extend(adv); batch["ret"].extend(ret)
            infos.append(info)
        stats = agent.update({k: np.asarray(v) for k, v in batch.items()})

        row = dict(update=u, episodes=u * episodes_per_update, **stats)
        for k in ("total_reward", "throughput", "avg_wait", "avg_wip", "resource_cost", "sla_violation",
                  "mean_servers", "mean_rate", "mean_release", "health_dispatch"):
            row[k] = float(np.mean([i[k] for i in infos]))
        history.append(row)
        if u % 10 == 0 or u == 1:
            print(f"[{reward}] upd {u:3d} | R {row['total_reward']:8.2f} | thr {row['throughput']:.3f} "
                  f"wait {row['avg_wait']:.3f} wip {row['avg_wip']:.2f} cost {row['resource_cost']:.2f} "
                  f"sla {row['sla_violation']:.3f} | srv {row['mean_servers']:.2f} rate {row['mean_rate']:.2f} "
                  f"rel {row['mean_release']:.2f} hd {row['health_dispatch']:.2f} | {time.time() - t0:.0f}s",
                  flush=True)

    agent.save(os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(dict(reward=reward, reward_config=config_dict(rc), history=history), f, indent=1)
    return agent, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="tradeoff", choices=list(REWARD_PRESETS))
    ap.add_argument("--updates", type=int, default=250)
    ap.add_argument("--episodes-per-update", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    train(args.reward, args.updates, args.episodes_per_update, args.seed, args.out)
