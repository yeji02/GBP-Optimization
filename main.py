"""
전체 파이프라인
 0) 주문 CSV 생성 (ERP 주문 형식) (data/erp_orders.csv가 없을 때)
 1) 단일 실행 예시 (Fixed 운영 조건, 평가 기간 첫 구간)
 2) Static-Opt: 고정 파라미터 조합 Grid Search (정적 최적화 baseline)
 3) PPO 학습: v1(성능 지표만) / v2(trade-off) 보상
 4) 평가: Fixed vs Static-Opt vs PPO, 시나리오별 KPI + 그림 → results/
"""
import json, os, subprocess, sys

from erp_data import DATA_CSV, prepare

RAW_XLSX = "data/raw/online_retail_II.xlsx"

if __name__ == "__main__":
    if not os.path.exists(DATA_CSV):
        print("=== 주문 CSV 생성 (ERP 주문 형식) ===")
        prepare(RAW_XLSX)

    from gbp_env import GBPEnv
    from baselines import FixedPolicy, rollout, static_grid_search
    from train_ppo import train

    os.makedirs("runs", exist_ok=True)

    print("=== 단일 실행 예시 (Fixed, 실제 주문 수요) ===")
    print(rollout(GBPEnv("erp"), FixedPolicy(), seed=0))

    print("\n=== Static-Opt (Grid Search) ===")
    res = static_grid_search()
    with open("runs/static_opt.json", "w") as f:
        json.dump(dict(best=res[0], top=res[:20]), f, indent=1)

    for reward in ("performance", "tradeoff"):
        print(f"\n=== PPO 학습 ({reward}) ===")
        train(reward=reward, updates=400)

    print("\n=== 평가 ===")
    subprocess.run([sys.executable, "evaluate.py"], check=True)
