import math
from typing import Dict
from xdevs.models import Atomic, Port

from xdevs_Job import Job

class Processor(Atomic):
    """단일 서버: 처리시간 = job.size / service_rate"""
    def __init__(self, service_rate=1.2, name="Processor"):
        super().__init__(name)
        self.in_job: Port[Job] = Port(Job, "in_job");       self.add_in_port(self.in_job)
        self.out_done: Port[bool] = Port(bool, "out_done"); self.add_out_port(self.out_done)
        self.out_job: Port[Job] = Port(Job, "out_job");     self.add_out_port(self.out_job)

        self.job = None
        self.service_rate = float(service_rate)
        self.sigma = math.inf

    def _service_time(self, job):
        return job.size / max(1e-6, self.service_rate)

    def initialize(self):
        self.sigma = math.inf

    def ta(self):
        return self.sigma

    def lambdaf(self):
        # 완료 시 신호와 완료된 Job을 동시에 송신
        if self.job is not None:
            self.out_done.add(True)
            self.out_job.add(self.job)

    def deltint(self):
        # 서비스 완료 처리
        self.job = None
        self.sigma = math.inf

    def deltext(self, e):
        # 유휴 상태일 때만 새 잡 수락
        if self.job is None and not self.in_job.empty():
            # values는 generator → 리스트로 받아서 첫 번째만 사용
            msgs = list(self.in_job.values)
            self.in_job.clear()
            if msgs:
                job = msgs[0]
                job.processing_time = self._service_time(job)
                self.job = job
                self.remaining = job.processing_time
                self.sigma = self.remaining
                
    def deltcon(self, e):
        self.deltint()
        self.deltext(0.0)

    def exit(self):
        pass
