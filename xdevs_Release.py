import math
from typing import List
from xdevs.models import Atomic, Port
from xdevs.sim import SimulationClock

from xdevs_Job import Job, Control, TimeAvg


class Release(Atomic):
    """
    작업 투입(Order Release) 모델 - AI 제어변수: release_interval
    도착한 주문은 Backlog에 쌓이고, 최소 release_interval 간격으로 공정(Buffer)에 투입됨.
    간격이 짧으면 공정 내 WIP 증가, 길면 Backlog 대기 증가 → WIP vs 대기시간 trade-off
    """
    def __init__(self, clock: SimulationClock, release_interval=0.25):
        super().__init__("Release")
        self.clock = clock
        self.in_job: Port[Job] = Port(Job, "in_job");             self.add_in_port(self.in_job)
        self.in_ctrl: Port[Control] = Port(Control, "in_ctrl");   self.add_in_port(self.in_ctrl)
        self.out_job: Port[Job] = Port(Job, "out_job");           self.add_out_port(self.out_job)

        self.release_interval = float(release_interval)
        self.backlog: List[Job] = []
        self.last_release = -math.inf
        self.n_backlog = TimeAvg()
        self.sigma = math.inf

    def _schedule(self):
        now = self.clock.time
        if self.backlog:
            self.sigma = max(0.0, self.last_release + self.release_interval - now)
        else:
            self.sigma = math.inf

    def initialize(self):
        self.sigma = math.inf

    def ta(self):
        return self.sigma

    def lambdaf(self):
        if self.backlog:
            self.out_job.add(self.backlog[0])

    def deltint(self):
        now = self.clock.time
        job = self.backlog.pop(0)
        job.release_time = now
        self.last_release = now
        self.n_backlog.set(now, len(self.backlog))
        self._schedule()

    def deltext(self, e):
        now = self.clock.time
        for job in self.in_job.values:
            self.backlog.append(job)
        for c in self.in_ctrl.values:
            self.release_interval = float(c.release_interval)
        self.n_backlog.set(now, len(self.backlog))
        self._schedule()

    def deltcon(self):
        self.deltint()
        self.deltext(0.0)

    def exit(self):
        pass
