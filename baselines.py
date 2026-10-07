"""
비교 기준(baseline) 정책
 - Fixed      : 현장 기준 운영 조건 고정 (2대, 정격 속도 1.0, 즉시 투입, 고정 순서 할당)
 - Static-Opt : 학습과 동일한 시나리오 분포에서 '고정 파라미터 조합'의 최적값을 Grid Search로 탐색
                (정적 최적화 baseline. 추후 Bayesian Optimization으로 대체할 자리)
"""
import itertools, json, os, time
import numpy as np

from xdevs_Job import Control
from gbp_env import GBPEnv, RewardConfig, default_control, N_SERVERS, RELEASE_BOUNDS


class FixedPolicy:
    name = "Fixed"

    def __init__(self, ctrl: Control = None):
        self.ctrl = ctrl or default_control()

    def __call__(self, obs):
        c = self.ctrl
        return Control(c.service_rate, c.active_servers, c.release_interval, c.dispatch)


class PPOPolicy:
    name = "PPO"

    def __init__(self, agent):
        self.agent = agent

    def __call__(self, obs):
        a, _, _ = self.agent.act(obs, deterministic=True)
        return self.agent.to_control(a)


def rollout(env: GBPEnv, policy, seed=None, scenario=None):
    obs = env.reset(seed=seed, scenario=scenario)
    done, info = False, {}
    while not done:
        obs, _, done, info = env.step(policy(obs))
    return info


def static_grid_search(reward: RewardConfig = None, n_episodes=16, seed=777, verbose=True):
    rates = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
    servers = range(1, N_SERVERS + 1)
    releases = (RELEASE_BOUNDS[0], 0.25, 0.5)
    dispatches = (0, 1)
    env = GBPEnv("mixed", reward=reward)
    seeds = [seed * 1000 + i for i in range(n_episodes)]   # 모든 조합에 동일 시나리오/난수 사용 (CRN)

    results, t0 = [], time.time()
    for r, k, rel, d in itertools.product(rates, servers, releases, dispatches):
        if r * k < 1.3:            # 평균 수요(1.5)도 감당 못하는 조합은 제외
            continue
        ctrl = Control(r, k, rel, d)
        R = [rollout(env, FixedPolicy(ctrl), seed=s)["total_reward"] for s in seeds]
        results.append(dict(service_rate=r, active_servers=k, release_interval=rel, dispatch=d,
                            mean_reward=float(np.mean(R)), std_reward=float(np.std(R))))
    results.sort(key=lambda x: -x["mean_reward"])
    if verbose:
        print(f"[Static-Opt] {len(results)} combos × {n_episodes} episodes in {time.time() - t0:.0f}s")
        for x in results[:5]:
            print("   ", x)
    return results


def load_static_best(path):
    with open(path) as f:
        b = json.load(f)["best"]
    return Control(b["service_rate"], b["active_servers"], b["release_interval"], b["dispatch"])


if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    res = static_grid_search()
    with open("runs/static_opt.json", "w") as f:
        json.dump(dict(best=res[0], top=res[:20]), f, indent=1)
