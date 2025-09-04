import math
from typing import Dict
from xdevs.models import Atomic, Port

class Collector(Atomic):
    def __init__(self):
        super().__init__("Collector")
        self.in_event = Port("in_event"); self.add_in_port(self.in_event)
        self.now = 0.0
        self.n  = 0
        self.sum_time_in_system = 0.0
        self.sum_queueing_time  = 0.0
        self.first_time = None
        self.last_time = 0.0
        self.sigma = math.inf

    # 초기화: 내부 이벤트 스케줄 없음
    def initialize(self):
        self.sigma = math.inf

    # 내부 시간: 항상 sigma 반환
    def ta(self):
        return self.sigma

    # 출력 없음(수집기라서 아무 것도 내보내지 않음)
    def lambdaf(self):
        pass

    # 내부 전이: 스케줄 해둘 이벤트가 없으므로 그대로 무한대 유지
    def deltint(self):
        self.sigma = math.inf

    # 외부 전이: 완료된 잡 수신 후 지표 계산 및 업데이트
    def deltext(self, e):
        self.now += e
        if not self.in_event.empty():
            for job in self.in_event.values:
                t_sys = self.now - job.creation_time
                q_t   = max(0.0, t_sys - (job.processing_time or 0.0))
                self.n += 1
                self.sum_time_in_system += t_sys
                self.sum_queueing_time  += q_t
                if self.first_time is None:
                    self.first_time = self.now
                self.last_time = self.now
            self.in_event.clear()
        # 출력이 없고, 외부 입력 처리만 하므로 다음 내부 이벤트 없음
        self.sigma = math.inf

    # 동시 전이: 관례적으로 내부 외부 순으로
    def deltcon(self, e):
        self.deltint()
        self.deltext(0.0)

    def exit(self):
        pass

    def metrics(self) -> Dict[str, float]:
        if self.n == 0:
            return dict(avg_time_in_system=float("nan"), avg_queueing_time=float("nan"),
                        count=0, throughput=0.0, makespan=0.0)
        makespan = max(0.0, (self.last_time - (self.first_time or self.last_time)))
        thr = self.n / makespan if makespan > 0 else float("inf")
        return dict(
            avg_time_in_system = self.sum_time_in_system / self.n,
            avg_queueing_time  = self.sum_queueing_time  / self.n,
            count = self.n,
            throughput = thr,
            makespan = makespan
        )
