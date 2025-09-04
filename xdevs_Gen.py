import math, random
from xdevs.models import Atomic, Port

from xdevs_Job import Job

class Generator(Atomic):
    """지수 간격 interarrival, N(size_mu, size_sigma^2) 크기의 잡을 n_jobs개 생성"""
    def __init__(self, interarrival=1.0, n_jobs=400, size_mu=1.0, size_sigma=0.2, seed=0):
        super().__init__("Generator")
        self.out: Port[Job] = Port(Job, "out")
        self.add_out_port(self.out)

        self.interarrival = float(interarrival)
        self.n_jobs = int(n_jobs)
        self.size_mu = float(size_mu)
        self.size_sigma = float(size_sigma)
        self.rng = random.Random(int(seed))

        self.now = 0.0
        self.left = self.n_jobs
        self.next_id = 0
        self.sigma = 0.0  # 다음 내부 이벤트까지 남은 시간

    def initialize(self):
        # 첫 이벤트까지의 대기시간 설정
        if self.left > 0:
            self.sigma = 0.0  # 즉시 첫 잡 생성
        else:
            self.sigma = math.inf

    def ta(self):
        return self.sigma

    def lambdaf(self):
        # 출력은 포트에 add
        t = self.now + self.sigma
        size = max(1e-3, self.rng.gauss(self.size_mu, self.size_sigma))
        self.out.add(Job(self.next_id, creation_time=t, size=size))

    def deltint(self):
        # 내부 전이: 다음 도착 예약
        self.now += self.sigma
        self.next_id += 1
        self.left -= 1
        if self.left > 0:
            gap = self.rng.expovariate(1.0 / self.interarrival) if self.interarrival > 0 else 0.0
            self.sigma = gap
        else:
            self.sigma = math.inf

    def deltext(self, e):
        self.now += e

    def deltcon(self, e):
        self.deltint()
        self.deltext(0.0)

    def exit(self):
        pass