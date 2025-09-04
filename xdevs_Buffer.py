import math
from typing import List
from xdevs.models import Atomic, Port

from xdevs_Job import Job

class BufferMulti(Atomic):
    """다중 서버 앞 FIFO 버퍼: idle 서버가 있으면 즉시 해당 서버로 1개 전송"""
    def __init__(self, servers:int, capacity=math.inf):
        super().__init__("Buffer")

        # 입력 포트
        self.in_job: Port[Job] = Port(Job, "in_job")
        self.add_in_port(self.in_job)

        self.servers = int(servers)
        self.capacity = math.inf if capacity is None else int(capacity)

        # 출력/입력 포트(서버 수만큼)
        self.out_job: List[Port[Job]] = []
        self.in_done: List[Port[bool]] = []
        for i in range(self.servers):
            op = Port(Job, f"out_job_{i}")
            self.add_out_port(op)
            self.out_job.append(op)

            ip = Port(bool, f"in_done_{i}")
            self.add_in_port(ip)
            self.in_done.append(ip)

        self.q: List[Job] = []
        self.busy = [False] * self.servers
        self._target = None
        self.sigma = math.inf

    def initialize(self):
        self.sigma = 0.0 if (self.q and self._first_idle() is not None) else math.inf

    def _first_idle(self):
        for i, b in enumerate(self.busy):
            if not b:
                return i
        return None

    def ta(self):
        return self.sigma

    def lambdaf(self):
        # 보낼 수 있으면 즉시 헤드 1건을 선택 서버로 전송
        i = self._first_idle()
        if i is not None and self.q:
            self._target = i
            self.out_job[i].add(self.q[0])

    def deltint(self):
        # 실제 전송 반영(큐 pop 및 해당 서버 busy)
        if self._target is not None and self.q:
            self.q.pop(0)
            self.busy[self._target] = True
        self._target = None
        self.sigma = 0.0 if (self.q and self._first_idle() is not None) else math.inf

    def deltext(self, e):
        # 시간 경과
        # 잡 도착 처리
        if not self.in_job.empty():
            for job in self.in_job.values:
                if len(self.q) < self.capacity:
                    self.q.append(job)
            self.in_job.clear()
        # 완료 신호 처리
        for i, pin in enumerate(self.in_done):
            if not pin.empty():
                if any(bool(x) for x in pin.values):
                    self.busy[i] = False
                pin.clear()
        # 즉시 보낼 수 있으면 0, 아니면 ∞
        self.sigma = 0.0 if (self.q and self._first_idle() is not None) else math.inf

    def deltcon(self, e):
        # 내부 처리 후 외부 순서
        self.deltint()
        self.deltext(0.0)

    def exit(self):
        pass