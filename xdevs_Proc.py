import math
import numpy as np
from xdevs.models import Atomic, Port
from xdevs.sim import SimulationClock

from xdevs_Job import Job, Control, TimeAvg
from scenarios import Scenario


class Processor(Atomic):
    """
    단일 설비. 처리시간 = size / (service_rate × health(t)) × noise
     - service_rate : AI 제어변수 (다음 작업부터 적용, 진행 중 작업은 기존 조건 유지)
     - health(t)    : 환경 변수 (설비 성능 저하)
     - noise        : 환경 변수 (처리시간 변동, lognormal)
    """
    def __init__(self, clock: SimulationClock, scenario: Scenario, idx: int,
                 service_rate=1.0, seed=0):
        super().__init__(f"Processor{idx}")
        self.clock = clock
        self.sc = scenario
        self.idx = idx
        self.rng = np.random.default_rng(seed)
        s2 = math.log(1.0 + scenario.proc_noise_cv ** 2)
        self._noise_sd, self._noise_mu = math.sqrt(s2), -0.5 * s2   # E[noise] = 1

        self.in_job: Port[Job] = Port(Job, "in_job");             self.add_in_port(self.in_job)
        self.in_ctrl: Port[Control] = Port(Control, "in_ctrl");   self.add_in_port(self.in_ctrl)
        self.out_done: Port[bool] = Port(bool, "out_done");       self.add_out_port(self.out_done)
        self.out_job: Port[Job] = Port(Job, "out_job");           self.add_out_port(self.out_job)

        self.job = None
        self.service_rate = float(service_rate)
        self.busy = TimeAvg()
        self.sigma = math.inf

    def initialize(self):
        self.sigma = math.inf

    def ta(self):
        return self.sigma

    def lambdaf(self):
        if self.job is not None:
            self.job.finish_time = self.clock.time
            self.out_done.add(True)
            self.out_job.add(self.job)

    def deltint(self):
        self.job = None
        self.busy.set(self.clock.time, 0)
        self.sigma = math.inf

    def deltext(self, e):
        now = self.clock.time
        for c in self.in_ctrl.values:
            self.service_rate = float(c.service_rate)
        if self.job is not None:
            self.sigma -= e
        msgs = list(self.in_job.values)
        if self.job is None and msgs:
            job = msgs[0]
            noise = self.rng.lognormal(self._noise_mu, self._noise_sd)
            rate = max(1e-6, self.service_rate)
            job.expected_time = job.size / rate
            job.processing_time = job.size / (rate * self.sc.health(self.idx, now)) * noise
            job.start_time = now
            job.server = self.idx
            self.job = job
            self.busy.set(now, 1)
            self.sigma = job.processing_time

    def deltcon(self):
        self.deltint()
        self.deltext(0.0)

    def exit(self):
        pass
