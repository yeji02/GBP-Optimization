import math
from typing import List
from xdevs.models import Atomic, Port

from xdevs_Job import Job


class Collector(Atomic):
    """완료된 작업을 수집. 구간/에피소드 KPI는 환경(GBPEnv)이 jobs 기록을 읽어 계산"""
    def __init__(self):
        super().__init__("Collector")
        self.in_event: Port[Job] = Port(Job, "in_event"); self.add_in_port(self.in_event)
        self.jobs: List[Job] = []
        self.sigma = math.inf

    def initialize(self):
        self.sigma = math.inf

    def ta(self):
        return self.sigma

    def lambdaf(self):
        pass

    def deltint(self):
        self.sigma = math.inf

    def deltext(self, e):
        self.jobs.extend(self.in_event.values)
        self.sigma = math.inf

    def deltcon(self):
        self.deltint()
        self.deltext(0.0)

    def exit(self):
        pass
