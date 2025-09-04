import math
from xdevs.models import Coupled

from xdevs_Gen import Generator
from xdevs_Buffer import BufferMulti
from xdevs_Proc import Processor
from xdevs_Coll import Collector

class GBPSystemMulti(Coupled):
    def __init__(self, n_jobs=400, interarrival=1.0, service_rate=1.2,
                 size_mu=1.0, size_sigma=0.2, capacity=math.inf, servers=2, seed=0,
                 name="GBPSystemMulti"):
        super().__init__(name)

        # 서브모델 생성
        gen  = Generator(interarrival=interarrival, n_jobs=n_jobs,
                         size_mu=size_mu, size_sigma=size_sigma, seed=seed)
        buf  = BufferMulti(servers=servers, capacity=capacity)
        procs = [Processor(service_rate=service_rate, name=f"Processor{i}") for i in range(servers)]
        col  = Collector()

        # 서브모델 등록 
        self.add_component(gen)
        self.add_component(buf)
        for p in procs:
            self.add_component(p)
        self.add_component(col)

        # 연결
        self.add_coupling(gen.out, buf.in_job)
        for i, p in enumerate(procs):
            self.add_coupling(buf.out_job[i], p.in_job)
            self.add_coupling(p.out_done,    buf.in_done[i])
            self.add_coupling(p.out_job,     col.in_event)

        self.collector = col