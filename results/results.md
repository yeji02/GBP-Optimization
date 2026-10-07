# 평가 결과

- 시나리오별 30개 시드(공통 난수), 에피소드 길이 600, 의사결정 주기 10
- 모든 방법은 동일한 trade-off 보상(v2)으로 채점

## 1. 방법별 KPI (3개 시나리오 평균)

| Method | Throughput ↑ | Waiting ↓ | WIP ↓ | Resource Cost ↓ | SLA Violation ↓ | Reward ↑ |
|---|---|---|---|---|---|---|
| Fixed | 1.537 | 1.407 | 3.80 | 2.000 | 9.82% | -71.80 |
| Static-Opt | 1.538 | 0.345 | 1.82 | 2.440 | 0.25% | -12.81 |
| PPO | 1.539 | 0.286 | 1.73 | 2.482 | 0.01% | -10.95 |

## 2. 시나리오별 보상 (mean ± std)

| Scenario | Fixed | Static-Opt | PPO |
|---|---|---|---|
| Normal | -20.22 ± 10.66 | -10.88 ± 1.83 | -10.79 ± 1.94 |
| Demand Surge | -116.16 ± 94.65 | -16.38 ± 10.03 | -11.00 ± 2.53 |
| Machine Degradation | -79.01 ± 52.76 | -11.18 ± 1.92 | -11.06 ± 1.87 |

## 3. 시나리오별 상세 KPI

**Normal**

| Method | Throughput ↑ | Waiting ↓ | WIP ↓ | Resource Cost ↓ | SLA Violation ↓ | Reward ↑ |
|---|---|---|---|---|---|---|
| Fixed | 1.491 | 0.667 | 2.49 | 2.000 | 1.36% | -20.22 |
| Static-Opt | 1.491 | 0.273 | 1.65 | 2.440 | 0.01% | -10.88 |
| PPO | 1.491 | 0.254 | 1.62 | 2.464 | 0.00% | -10.79 |

**Demand Surge**

| Method | Throughput ↑ | Waiting ↓ | WIP ↓ | Resource Cost ↓ | SLA Violation ↓ | Reward ↑ |
|---|---|---|---|---|---|---|
| Fixed | 1.632 | 2.009 | 4.92 | 2.000 | 16.97% | -116.16 |
| Static-Opt | 1.633 | 0.483 | 2.15 | 2.440 | 0.69% | -16.38 |
| PPO | 1.633 | 0.348 | 1.94 | 2.515 | 0.01% | -11.00 |

**Machine Degradation**

| Method | Throughput ↑ | Waiting ↓ | WIP ↓ | Resource Cost ↓ | SLA Violation ↓ | Reward ↑ |
|---|---|---|---|---|---|---|
| Fixed | 1.488 | 1.546 | 3.99 | 2.000 | 11.12% | -79.01 |
| Static-Opt | 1.491 | 0.278 | 1.66 | 2.440 | 0.07% | -11.18 |
| PPO | 1.491 | 0.257 | 1.63 | 2.467 | 0.02% | -11.06 |

## 4. 보상 설계 비교: v1(성능 지표만) vs v2(trade-off)

| Reward | 평균 가동 대수 | 평균 Service Rate | Waiting ↓ | Resource Cost ↓ | Reward(v2 기준) ↑ |
|---|---|---|---|---|---|
| v1: T − W | 4.00 | 1.160 | 0.013 | 4.691 | -65.39 |
| v2: T − W − WIP − C − V | 2.05 | 1.195 | 0.286 | 2.482 | -10.95 |

## 5. PPO의 상황별 의사결정 (seed 0)

| Scenario | 구간 | 평균 가동 대수 | 평균 Service Rate | HEALTH 디스패칭 비율 |
|---|---|---|---|---|
| Demand Surge | 평상시 (0~200) | 2.00 | 1.198 | 100% |
| Demand Surge | 수요 급증 (200~400) | 2.20 | 1.200 | 100% |
| Demand Surge | 수요 복귀 (400~600) | 2.00 | 1.200 | 100% |
| Machine Degradation | 정상 (0~200) | 2.00 | 1.198 | 100% |
| Machine Degradation | P0 성능 저하 (200~600) | 2.10 | 1.187 | 100% |
| Normal | 전체 | 2.03 | 1.197 | 100% |

**설비 성능 저하 이후(t≥200) Processor별 작업 분담 비율 (seed 0)**

| Method | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| Fixed | 42% | 58% | 0% | 0% |
| Static-Opt | 1% | 9% | 48% | 42% |
| PPO | 1% | 24% | 47% | 28% |

## 그림

![training](training_curve.png)

![dynamic](dynamic_response.png)

![kpi](kpi_comparison.png)
