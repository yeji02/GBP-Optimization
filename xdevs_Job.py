from dataclasses import dataclass, field
from typing import List, Optional


class Job:
    """작업(Lot) 메시지. 공정 단계별 타임스탬프를 기록해 KPI 계산에 사용"""
    def __init__(self, jid, creation_time, size=1.0):
        self.id = int(jid)
        self.creation_time = float(creation_time)   # 수요(주문) 도착 시각
        self.size = float(size)                     # 작업량 (환경 변수)
        self.release_time: Optional[float] = None   # 공정 투입 시각 (Release)
        self.start_time: Optional[float] = None     # 가공 시작 시각
        self.finish_time: Optional[float] = None    # 가공 완료 시각
        self.processing_time: Optional[float] = None
        self.expected_time: Optional[float] = None  # 정상 설비 기준 예상 가공시간 (설비 상태 추정용)
        self.server: Optional[int] = None

    @property
    def waiting_time(self) -> float:
        """주문 도착 ~ 가공 시작까지의 대기시간 (투입 대기 + 버퍼 대기)"""
        return self.start_time - self.creation_time

    @property
    def cycle_time(self) -> float:
        return self.finish_time - self.creation_time

    def __repr__(self):
        return f"Job(id={self.id}, size={self.size:.3f}, t0={self.creation_time:.3f})"


@dataclass
class Control:
    """AI 제어변수 묶음. 의사결정 시점마다 DEVS 외부 이벤트로 주입됨"""
    service_rate: float                 # 설비 처리 속도
    active_servers: int                 # 가동할 Processor 수
    release_interval: float             # 작업 투입 최소 간격
    dispatch: int                       # 0: 고정 순서(INDEX), 1: 설비 상태 기반(HEALTH)
    health_est: List[float] = field(default_factory=list)  # 관측된 설비별 처리 성능 추정치


class TimeAvg:
    """구간별 시간가중 평균(Queue 길이, WIP, 가동률 등)을 위한 누적기"""
    def __init__(self, level=0.0):
        self.level = float(level)
        self.area = 0.0
        self.t = 0.0

    def set(self, now, level):
        self.area += self.level * (now - self.t)
        self.t = now
        self.level = float(level)

    def area_until(self, now):
        return self.area + self.level * (now - self.t)
