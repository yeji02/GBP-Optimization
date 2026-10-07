import math
import numpy as np
from xdevs.models import Atomic, Port

from xdevs_Job import Job
from scenarios import Scenario


class Generator(Atomic):
    """수요 발생기 (환경 변수): 시간가변 도착률 λ(t)의 비정상 포아송 과정(thinning)으로 주문 생성"""
    def __init__(self, scenario: Scenario, seed=0):
        super().__init__("Generator")
        self.out: Port[Job] = Port(Job, "out")
        self.add_out_port(self.out)

        self.sc = scenario
        self.rng = np.random.default_rng(seed)
        self.lam_max = scenario.max_arrival_rate()

        self.now = 0.0
        self.next_id = 0
        self.n_arrived = 0
        self.sigma = math.inf

    def _next_arrival(self, t):
        # thinning: λ_max로 후보를 뽑고 λ(t)/λ_max 확률로 채택
        while True:
            t += self.rng.exponential(1.0 / self.lam_max)
            if t >= self.sc.horizon:
                return math.inf
            if self.rng.random() < self.sc.arrival_rate(t) / self.lam_max:
                return t

    def initialize(self):
        self.now = 0.0
        self.sigma = self._next_arrival(0.0)

    def ta(self):
        return self.sigma

    def lambdaf(self):
        t = self.now + self.sigma
        size = max(0.05, self.rng.normal(self.sc.size_mu, self.sc.size_sigma))
        self.out.add(Job(self.next_id, creation_time=t, size=size))

    def deltint(self):
        self.now += self.sigma
        self.next_id += 1
        self.n_arrived += 1
        self.sigma = self._next_arrival(self.now) - self.now

    def deltext(self, e):
        self.now += e
        self.sigma -= e

    def deltcon(self):
        self.deltint()
        self.deltext(0.0)

    def exit(self):
        pass
