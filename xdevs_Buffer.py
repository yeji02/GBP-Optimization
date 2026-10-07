import math
from typing import List
from xdevs.models import Atomic, Port
from xdevs.sim import SimulationClock

from xdevs_Job import Job, Control, TimeAvg

DISPATCH_INDEX = 0    # 고정 순서: Processor0부터 가동/할당
DISPATCH_HEALTH = 1   # 상태 기반: 관측 처리성능이 높은 Processor부터 가동/할당
DISPATCH_NAMES = ("INDEX", "HEALTH")


class BufferMulti(Atomic):
    """
    다중 서버 앞 FIFO 버퍼 + 디스패처
    AI 제어변수: active_servers(가동 대수), dispatch(할당 정책)
    비가동 전환된 Processor는 진행 중인 작업만 마치고 새 작업을 받지 않음
    """
    def __init__(self, clock: SimulationClock, n_servers: int, active_servers: int = 2,
                 dispatch: int = DISPATCH_INDEX):
        super().__init__("Buffer")
        self.clock = clock
        self.in_job: Port[Job] = Port(Job, "in_job");             self.add_in_port(self.in_job)
        self.in_ctrl: Port[Control] = Port(Control, "in_ctrl");   self.add_in_port(self.in_ctrl)

        self.n_servers = int(n_servers)
        self.out_job: List[Port[Job]] = []
        self.in_done: List[Port[bool]] = []
        for i in range(self.n_servers):
            op = Port(Job, f"out_job_{i}");  self.add_out_port(op); self.out_job.append(op)
            ip = Port(bool, f"in_done_{i}"); self.add_in_port(ip);  self.in_done.append(ip)

        self.q: List[Job] = []
        self.busy = [False] * self.n_servers
        self.health_est = [1.0] * self.n_servers
        self.dispatch = int(dispatch)
        self.active_servers = int(active_servers)
        self.active = self._active_set()
        self._target = None
        self.n_queue = TimeAvg()
        self.sigma = math.inf

    def _order(self):
        idx = list(range(self.n_servers))
        if self.dispatch == DISPATCH_HEALTH:
            idx.sort(key=lambda i: (-self.health_est[i], i))
        return idx

    def _active_set(self):
        return set(self._order()[:self.active_servers])

    def _pick_idle(self):
        for i in self._order():
            if i in self.active and not self.busy[i]:
                return i
        return None

    def _schedule(self):
        self.sigma = 0.0 if (self.q and self._pick_idle() is not None) else math.inf

    def initialize(self):
        self._schedule()

    def ta(self):
        return self.sigma

    def lambdaf(self):
        i = self._pick_idle()
        if i is not None and self.q:
            self._target = i
            self.out_job[i].add(self.q[0])

    def deltint(self):
        if self._target is not None and self.q:
            self.q.pop(0)
            self.busy[self._target] = True
            self.n_queue.set(self.clock.time, len(self.q))
        self._target = None
        self._schedule()

    def deltext(self, e):
        for job in self.in_job.values:
            self.q.append(job)
        for i, pin in enumerate(self.in_done):
            if any(bool(x) for x in pin.values):
                self.busy[i] = False
        for c in self.in_ctrl.values:
            self.active_servers = int(c.active_servers)
            self.dispatch = int(c.dispatch)
            if c.health_est:
                self.health_est = list(c.health_est)
            self.active = self._active_set()
        self.n_queue.set(self.clock.time, len(self.q))
        self._schedule()

    def deltcon(self):
        self.deltint()
        self.deltext(0.0)

    def exit(self):
        pass
