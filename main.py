from xdevs_sim import run_gbp_multi, optimize_multi_params_ppo

if __name__ == "__main__":
    N_JOBS = 400
    BOUNDS = dict(
        interarrival=(0.20, 1.20),
        service_rate=(0.8, 5.0),
        size_mu=(0.5, 1.2),
        size_sigma=(0.05, 0.6),
    )
    CAP_CHOICES = (20, 50, 100)
    SRV_CHOICES = (1, 2, 3, 4)
    CONT_INIT   = dict(interarrival=1.0, service_rate=1.2, size_mu=1.0, size_sigma=0.2)

    print("=== 단일 실행 예시 ===")
    demo = run_gbp_multi(n_jobs=N_JOBS, interarrival=1.0, service_rate=1.2,
                         size_mu=1.0, size_sigma=0.2, capacity=50, servers=2, seed=0)
    print("metrics:", demo)

    print("\n=== PPO(다변수) 최적화 시작 ===")
    best = optimize_multi_params_ppo(
        episodes=100, batch_size=8, n_jobs=N_JOBS,
        bounds=BOUNDS,
        capacity_choices=CAP_CHOICES,
        servers_choices=SRV_CHOICES,
        cont_init=CONT_INIT,
        seed=123,
        cost_srv=0.05, cost_cap=0.02,
        ppo_epochs=6, clip_eps=0.2,
        lr_mu=0.12, lr_sigma=0.04, lr_disc=0.08
    )
    print("\n최적화 결과:", best)
