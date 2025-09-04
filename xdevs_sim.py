import math, numpy as np
from typing import Dict
from xdevs.sim import Coordinator 

from xdevs_Coupled import GBPSystemMulti
from PPO import PPOMultiPolicy

def run_gbp_multi(n_jobs=400, interarrival=1.0, service_rate=1.2,
                  size_mu=1.0, size_sigma=0.2, capacity=math.inf,
                  servers=2, seed=0) -> Dict[str, float]:
    model = GBPSystemMulti(n_jobs=n_jobs, interarrival=interarrival, service_rate=service_rate,
                           size_mu=size_mu, size_sigma=size_sigma, capacity=capacity,
                           servers=servers, seed=seed)
    sim = Coordinator(model)
    sim.initialize()           
    sim.simulate()                

    return model.collector.metrics()

# =========================
# PPO Optimization loop 
# =========================
def optimize_multi_params_ppo(
    episodes=20, batch_size=8,
    n_jobs=400,
    bounds=None,
    capacity_choices=(20, 50, 100),
    servers_choices=(1, 2, 3, 4),
    cont_init=None,
    seed=123,
    cost_srv=0.0, cost_cap=0.0,
    w_time=0.7,         # avg time ↓
    w_mk=0.2,           # makespan ↓
    mk_scale=1000.0,    # makespan scaling
    w_thr=0.1,          # throughput ↑
    rho_low=0.7, rho_high=0.9, rho_penalty=0.5,
    # PPO hparams
    ppo_epochs=6, clip_eps=0.2,
    lr_mu=0.12, lr_sigma=0.04, lr_disc=0.08
):
    #0단계: 기본값 설정
    if bounds is None:
        bounds = dict(
            interarrival=(0.20, 1.20),
            service_rate=(0.8, 5.0),
            size_mu=(0.5, 1.2),
            size_sigma=(0.05, 0.6),
        )
    if cont_init is None:
        cont_init = dict(
            interarrival=1.0,
            service_rate=1.2,
            size_mu=1.0,
            size_sigma=0.2,
        )
    
    #1단계: ppo 에이전트 생성
    policy = PPOMultiPolicy(bounds, cont_init, list(capacity_choices), list(servers_choices), seed=seed)

    history = {
        "ep": [], "batch_obj_avg": [], "best_obj": [],
        "mu_interarrival": [], "mu_service_rate": [], "mu_size_mu": [], "mu_size_sigma": [],
        "pc_cap_0": [], "pc_cap_1": [], "pc_cap_2": [],
        "ps_srv_0": [], "ps_srv_1": [], "ps_srv_2": [], "ps_srv_3": []
    }
    trial_rows = []

    best = dict(params=None, obj=float("inf"))
    max_cap = max(capacity_choices) if len(capacity_choices)>0 else 1.0

    #2단계: 메인 루프(학습)
    for ep in range(1, episodes+1):
        #2a: 행동 샘플링
        xs, caps, srvs, cache = policy.sample_batch(batch_size) #현재 정책에 따라 batch size만큼의 행동 묶음 샘플링
        rewards, objs = [], [] #이번 배치에서 얻을 보상과 목적 함수 값을 담을 빈 리스트

        #2b: 시뮬레이션 실행
        for i in range(batch_size): 
            #xs에서 연속 행동 파라미터들을, caps에서 버퍼 용량 파라미터를, srvs에서 서버 개수 파라미터를 가져옴.
            interarrival, service_rate, size_mu, size_sigma = map(float, xs[i]) 
            capacity = int(caps[i]) 
            servers  = int(srvs[i])

            #시뮬레이션 실행
            m = run_gbp_multi(
                n_jobs=n_jobs,
                interarrival=interarrival,
                service_rate=service_rate,
                size_mu=size_mu,
                size_sigma=size_sigma,
                capacity=capacity,
                servers=servers,
                seed=1000 + ep*100 + i
            )

            #2c: 목적 함수 및 보상 계산 (결과 평가)
            obj_time = m["avg_time_in_system"]
            mk       = m["makespan"]
            thr      = m["throughput"]

            arrival_rate  = 1.0 / max(1e-12, interarrival)
            service_time  = (size_mu / max(1e-6, service_rate))
            mu_total      = servers / max(1e-6, service_time)    # ≈ servers * service_rate / size_mu
            rho           = arrival_rate / max(1e-6, mu_total)

            cost = cost_srv*servers + cost_cap*(capacity/max_cap)
            rho_pen = rho_penalty*(max(0.0, rho - rho_high)**2 + max(0.0, rho_low - rho)**2)

            scalar_obj = (w_time*obj_time + w_mk*(mk/mk_scale)) - (w_thr*thr) + cost + rho_pen
            if rho > 0.98:
                scalar_obj += 5.0
                
            #목적함수 최소화가 목적이므로 음수로 변환하여 에이전트에 전달
            r = -scalar_obj

            #모든 결과값 기록(학습 과정 시각화용)
            trial_rows.append({
                "ep": ep, "i": i,
                "interarrival": interarrival, "service_rate": service_rate,
                "size_mu": size_mu, "size_sigma": size_sigma,
                "capacity": capacity, "servers": servers,
                "avg_time": obj_time, "makespan": mk, "throughput": thr,
                "rho": rho, "cost": cost, "rho_pen": rho_pen,
                "scalar_obj": scalar_obj
            })

            rewards.append(r)
            objs.append(scalar_obj)

            #2d: 최고 점수로 파라미터 업데이트
            if scalar_obj < best["obj"]:
                best = dict(
                    params=dict(interarrival=interarrival, service_rate=service_rate,
                                size_mu=size_mu, size_sigma=size_sigma,
                                capacity=capacity, servers=servers),
                    obj=float(scalar_obj)
                )

        #2e: 정책 업데이트
        policy.update_ppo(cache, np.array(rewards), clip_eps=clip_eps, epochs=ppo_epochs,
                          lr_mu=lr_mu, lr_sigma=lr_sigma, lr_disc=lr_disc)

        #2f: 학습 과정 로깅
        #현재 에피소드의 평균 점수, 최고 점수, 정책 파라미터 변화 등을 기록
        summ = policy.summary()
        history["ep"].append(ep)
        history["batch_obj_avg"].append(float(np.mean(objs)))
        history["best_obj"].append(float(best["obj"]))
        history["mu_interarrival"].append(float(summ["mu"][0]))
        history["mu_service_rate"].append(float(summ["mu"][1]))
        history["mu_size_mu"].append(float(summ["mu"][2]))
        history["mu_size_sigma"].append(float(summ["mu"][3]))

        pc = np.pad(summ["pc"], (0, max(0, 3 - len(summ["pc"]))), constant_values=np.nan)
        ps = np.pad(summ["ps"], (0, max(0, 4 - len(summ["ps"]))), constant_values=np.nan)
        history["pc_cap_0"].append(float(pc[0]))
        history["pc_cap_1"].append(float(pc[1]))
        history["pc_cap_2"].append(float(pc[2]))
        history["ps_srv_0"].append(float(ps[0]))
        history["ps_srv_1"].append(float(ps[1]))
        history["ps_srv_2"].append(float(ps[2]))
        history["ps_srv_3"].append(float(ps[3]))
        
        if ep % 10 == 0:
            print(f"[EP {ep:02d}] "
                  f"mu={np.round(summ['mu'],3)}, sigma={np.round(summ['sigma'],3)} | "
                  f"pc={np.round(summ['pc'],3)}, ps={np.round(summ['ps'],3)} | "
                  f"batch_obj_avg={np.mean(objs):.3f} | best_obj={best['obj']:.3f}")

    if best["params"] is not None:
        p = best["params"]
        m = run_gbp_multi(n_jobs=n_jobs, **p, seed=9999)
        print("\n[VERIFY] best_params=", p)
        print("        metrics     =",
              dict(avg_time_in_system=round(m["avg_time_in_system"],3),
                   avg_queueing_time=round(m["avg_queueing_time"],3),
                   throughput=round(m["throughput"],3),
                   makespan=round(m["makespan"],3),
                   count=m["count"]))
    return best
