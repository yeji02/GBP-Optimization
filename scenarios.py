"""
환경 변수(AI가 통제할 수 없는 공정 조건) 정의
 - Demand Arrival / Job Size : 실제 판매 주문 데이터에서 재생 (erp_data.py)
 - Processing Noise          : 처리시간 변동 (lognormal 곱셈 노이즈)
 - Machine Degradation       : 특정 Processor의 처리 성능 저하
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from erp_data import order_book, SIM_PER_DAY

EPISODE_DAYS = 5
HORIZON = EPISODE_DAYS * SIM_PER_DAY     # 에피소드 길이 (simulation time)
BASE_RATE = 1.0                          # 관측값 정규화용 기준 도착률 (학습 기간 평균 ≈ 1.02)
SCENARIOS = ("erp", "erp_degradation")


@dataclass
class Scenario:
    name: str
    orders: np.ndarray                                        # (N, 2) [도착 시각, 작업량]
    degradation: Optional[Tuple[int, float, float]] = None    # (server_idx, t_start, factor)
    proc_noise_cv: float = 0.1
    horizon: float = HORIZON
    period: str = ""                                          # 실제 주문 기간
    start_day: int = 0

    def health(self, server: int, t: float) -> float:
        """설비 실제 성능 배율 (1.0 = 정상). AI는 이 값을 직접 볼 수 없고 처리시간으로 추정만 가능"""
        if self.degradation and server == self.degradation[0] and t >= self.degradation[1]:
            return self.degradation[2]
        return 1.0


def make_scenario(name: str, rng: Optional[np.random.Generator] = None,
                  start_day: Optional[int] = None) -> Scenario:
    """
    학습용 (name="mixed", rng 지정): 학습 기간(~2011-06)의 임의 6영업일 구간,
        절반 확률로 임의 설비·시점·크기의 성능 저하를 추가
    평가용 (start_day 지정): 평가 기간(2011-07~)의 해당 구간.
        erp_degradation은 3번째 영업일 시작(t=100)부터 Processor0 성능 -30%
    """
    book = order_book()
    rng = rng if rng is not None else np.random.default_rng(0 if start_day is None else start_day)

    if name == "mixed":
        start = int(rng.choice(book.train_days[:len(book.train_days) - EPISODE_DAYS + 1]))
        deg = None
        if rng.random() < 0.5:
            deg = (int(rng.integers(4)), float(rng.uniform(0.2, 0.7) * HORIZON), float(rng.uniform(0.6, 0.8)))
        return Scenario("erp_degradation" if deg else "erp", book.window(start, EPISODE_DAYS, rng),
                        degradation=deg, period=book.period(start, EPISODE_DAYS), start_day=start)

    if start_day is None:
        start_day = book.test_days[0]
    orders = book.window(start_day, EPISODE_DAYS, rng)
    period = book.period(start_day, EPISODE_DAYS)
    if name == "erp":
        return Scenario("erp", orders, period=period, start_day=start_day)
    if name == "erp_degradation":
        return Scenario("erp_degradation", orders, degradation=(0, 2 * SIM_PER_DAY, 0.7),
                        period=period, start_day=start_day)
    raise ValueError(name)


def test_windows():
    book = order_book()
    return book.windows(book.test_days, EPISODE_DAYS)
