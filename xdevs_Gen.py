import math
from xdevs.models import Atomic, Port

from xdevs_Job import Job
from scenarios import Scenario


class Generator(Atomic):
    """수요 발생기 (환경 변수): 실제 판매 주문 데이터의 도착 시각·작업량 그대로 Job을 생성"""
    def __init__(self, scenario: Scenario):
        super().__init__("Generator")
        self.out: Port[Job] = Port(Job, "out")
        self.add_out_port(self.out)

        self.orders = scenario.orders
        self.horizon = scenario.horizon
        self.now = 0.0
        self.i = 0
        self.n_arrived = 0
        self.sigma = math.inf

    def _schedule(self):
        if self.i < len(self.orders) and self.orders[self.i, 0] < self.horizon:
            self.sigma = max(0.0, self.orders[self.i, 0] - self.now)
        else:
            self.sigma = math.inf

    def initialize(self):
        self.now = 0.0
        self._schedule()

    def ta(self):
        return self.sigma

    def lambdaf(self):
        t, size = self.orders[self.i]
        self.out.add(Job(self.i, creation_time=float(t), size=float(size)))

    def deltint(self):
        self.now += self.sigma
        self.i += 1
        self.n_arrived += 1
        self._schedule()

    def deltext(self, e):
        self.now += e
        self.sigma -= e

    def deltcon(self):
        self.deltint()
        self.deltext(0.0)

    def exit(self):
        pass
