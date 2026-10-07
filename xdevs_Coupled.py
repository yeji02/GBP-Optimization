from xdevs.models import Coupled, Port
from xdevs.sim import SimulationClock

from xdevs_Job import Control
from xdevs_Gen import Generator
from xdevs_Release import Release
from xdevs_Buffer import BufferMulti
from xdevs_Proc import Processor
from xdevs_Coll import Collector
from scenarios import Scenario


class GBPSystemMulti(Coupled):
    """
    Generator → Release → Buffer → Processor[0..N-1] → Collector
    in_ctrl(외부 입력 포트): AI가 결정한 Control 메시지를 Release/Buffer/Processor로 전파
    Processor는 최대 대수(n_servers)만큼 생성해 두고 가동 여부는 Buffer가 제어
    """
    def __init__(self, clock: SimulationClock, scenario: Scenario, init_ctrl: Control,
                 n_servers=4, seed=0, name="GBPSystemMulti"):
        super().__init__(name)
        self.in_ctrl: Port[Control] = Port(Control, "in_ctrl")
        self.add_in_port(self.in_ctrl)

        gen = Generator(scenario, seed=seed)
        rel = Release(clock, release_interval=init_ctrl.release_interval)
        buf = BufferMulti(clock, n_servers, active_servers=init_ctrl.active_servers,
                          dispatch=init_ctrl.dispatch)
        procs = [Processor(clock, scenario, i, service_rate=init_ctrl.service_rate,
                           seed=seed * 101 + i + 1) for i in range(n_servers)]
        col = Collector()

        for c in (gen, rel, buf, *procs, col):
            self.add_component(c)

        self.add_coupling(gen.out, rel.in_job)
        self.add_coupling(rel.out_job, buf.in_job)
        for i, p in enumerate(procs):
            self.add_coupling(buf.out_job[i], p.in_job)
            self.add_coupling(p.out_done, buf.in_done[i])
            self.add_coupling(p.out_job, col.in_event)

        # 제어 명령 전파 (EIC)
        self.add_coupling(self.in_ctrl, rel.in_ctrl)
        self.add_coupling(self.in_ctrl, buf.in_ctrl)
        for p in procs:
            self.add_coupling(self.in_ctrl, p.in_ctrl)

        self.generator, self.release, self.buffer = gen, rel, buf
        self.processors, self.collector = procs, col
