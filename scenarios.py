"""
환경 변수(AI가 통제할 수 없는 공정 조건) 정의
 - Demand Arrival : 시간에 따라 변하는 수요 도착률 λ(t)
 - Job Size       : 제품별 작업량 분포
 - Processing Noise: 처리시간 변동 (lognormal 곱셈 노이즈)
 - Machine Degradation: 특정 Processor의 처리 성능 저하
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

HORIZON = 600.0          # 에피소드 길이 (simulation time)
BASE_RATE = 1.5          # 평상시 수요 도착률 (jobs / time)
SCENARIOS = ("normal", "demand_surge", "machine_degradation")


@dataclass
class Scenario:
    name: str = "normal"
    base_rate: float = BASE_RATE
    surge: Optional[Tuple[float, float, float]] = None        # (t_start, t_end, factor)
    degradation: Optional[Tuple[int, float, float]] = None    # (server_idx, t_start, factor)
    size_mu: float = 1.0
    size_sigma: float = 0.2
    proc_noise_cv: float = 0.1
    horizon: float = HORIZON

    def arrival_rate(self, t: float) -> float:
        if self.surge and self.surge[0] <= t < self.surge[1]:
            return self.base_rate * self.surge[2]
        return self.base_rate

    def max_arrival_rate(self) -> float:
        return self.base_rate * max(1.0, self.surge[2] if self.surge else 1.0)

    def health(self, server: int, t: float) -> float:
        """설비 실제 성능 배율 (1.0 = 정상). AI는 이 값을 직접 볼 수 없고 처리시간으로 추정만 가능"""
        if self.degradation and server == self.degradation[0] and t >= self.degradation[1]:
            return self.degradation[2]
        return 1.0


def make_scenario(name: str, rng: Optional[np.random.Generator] = None) -> Scenario:
    """
    rng=None  → 평가용 고정 시나리오 (변동 시점 고정)
    rng 지정  → 학습용 무작위 시나리오 (변동 시점·크기·대상 설비를 랜덤화해 특정 시점 암기 방지)
    """
    if name == "mixed":
        assert rng is not None
        name = SCENARIOS[rng.integers(len(SCENARIOS))]

    if rng is None:
        if name == "normal":
            return Scenario("normal")
        if name == "demand_surge":
            return Scenario("demand_surge", surge=(200.0, 400.0, 1.3))
        if name == "machine_degradation":
            return Scenario("machine_degradation", degradation=(0, 200.0, 0.7))
        raise ValueError(name)

    base = BASE_RATE * rng.uniform(0.9, 1.1)
    if name == "normal":
        return Scenario("normal", base_rate=base)
    if name == "demand_surge":
        t0 = rng.uniform(100, 300)
        return Scenario("demand_surge", base_rate=base,
                        surge=(t0, t0 + rng.uniform(150, 250), rng.uniform(1.2, 1.4)))
    if name == "machine_degradation":
        return Scenario("machine_degradation", base_rate=base,
                        degradation=(int(rng.integers(4)), rng.uniform(100, 300), rng.uniform(0.6, 0.8)))
    raise ValueError(name)
