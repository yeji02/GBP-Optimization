"""
동적 공정 파라미터 최적화 환경

  공정 상태 수집 → AI가 운영 파라미터 결정 → DEVS 시뮬레이션(Δt) → KPI/보상 계산 → 다음 의사결정

 - 의사결정 주기 DT(=10 time)마다 상태 s_t를 관측하고 행동 a_t를 Control 메시지로 DEVS 모델에 주입
 - s_t = [Queue, Backlog, WIP, Utilization, Throughput, AvgWaiting, ArrivalRate, ArrivalTrend,
          ActiveServers, ServiceRate, ReleaseInterval, ServerHealth×4]
 - a_t = [ServiceRate, ActiveServers, ReleaseInterval, Dispatch]
 - R_t = w1·T − w2·W − w3·WIP − w4·C − w5·V
"""
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import numpy as np
from xdevs.sim import Coordinator, SimulationClock

from xdevs_Job import Control
from xdevs_Coupled import GBPSystemMulti
from scenarios import Scenario, make_scenario, BASE_RATE

N_SERVERS = 4
DT = 10.0
SERVICE_RATE_BOUNDS = (0.7, 1.2)
RELEASE_BOUNDS = (0.05, 1.0)
OBS_DIM = 11 + N_SERVERS


@dataclass
class RewardConfig:
    w_thr: float = 1.0      # T  : Throughput (구간 완료 작업 수 / Δt)
    w_wait: float = 0.3     # W  : 대기 누적량 / Δt  (= 시간평균 대기 작업 수, Little's law상 λ·평균대기시간)
    w_wip: float = 0.2      # WIP: 공정 내 시간평균 재공 (Buffer + 가공 중)
    w_cost: float = 0.5     # C  : 자원 비용률 = Σ_active (c_fixed + c_speed·rate²)
    w_viol: float = 2.0     # V  : 제약 위반 = SLA 초과 비율 + WIP 상한 초과분
    c_fixed: float = 0.5    # 설비 1대 가동 고정비 (time당)
    c_speed: float = 0.5    # 처리속도 증가에 따른 에너지/마모 비용 (속도²에 비례)
    sla: float = 5.0        # 사이클타임(주문~완료) 허용 상한
    wip_max: float = 8.0    # 공정 내 WIP 허용 상한


REWARD_PRESETS = {
    # v1: 성능 지표만 보상 → 자원을 최대로 쓰는 정책으로 편향되는지 확인용
    "performance": RewardConfig(w_thr=1.0, w_wait=0.3, w_wip=0.0, w_cost=0.0, w_viol=0.0),
    # v2: 처리량·대기·WIP·자원비용·제약위반 trade-off
    "tradeoff": RewardConfig(),
}


def default_control() -> Control:
    """현장 기준 운영 조건 (Fixed baseline): 2대 가동, 정격 속도, 즉시 투입, 고정 순서 할당"""
    return Control(service_rate=1.0, active_servers=2, release_interval=RELEASE_BOUNDS[0], dispatch=0)


def resource_cost_rate(ctrl: Control, rc: RewardConfig) -> float:
    return ctrl.active_servers * (rc.c_fixed + rc.c_speed * ctrl.service_rate ** 2)


class GBPEnv:
    def __init__(self, scenario: str = "mixed", reward: RewardConfig = None, dt: float = DT,
                 seed: Optional[int] = None, record: bool = False):
        self.scenario_name = scenario
        self.rc = reward or RewardConfig()
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.record = record

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, scenario: Optional[Scenario] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if scenario is None:
            name = self.scenario_name
            scenario = make_scenario(name, self.rng) if name == "mixed" else make_scenario(name)
        self.sc = scenario
        self.ctrl = default_control()
        self.clock = SimulationClock()
        self.model = GBPSystemMulti(self.clock, scenario, self.ctrl, n_servers=N_SERVERS,
                                    seed=int(self.rng.integers(1 << 30)))
        self.coord = Coordinator(self.model, clock=self.clock)
        self.coord.initialize()

        self.t = 0.0
        self.health_est = [1.0] * N_SERVERS
        self._n_jobs_seen = 0
        self._prev = self._areas(0.0)
        self._prev_arrived = 0
        self.arr_trend = None
        self.total_reward = 0.0
        self.total_cost = 0.0
        self.trace = []
        return self._observe(self._interval_stats(empty=True))

    # ------------------------------------------------------------------
    def step(self, ctrl: Control):
        ctrl = self._clip(ctrl)
        ctrl.health_est = list(self.health_est)
        self.ctrl = ctrl
        self.coord.inject(self.model.in_ctrl, ctrl, self.t - self.coord.time_last)

        t_end = self.t + self.dt
        self._advance(t_end)
        self.t = t_end

        st = self._interval_stats()
        cost = resource_cost_rate(ctrl, self.rc)
        r = self._reward(st, cost)
        self.total_reward += r
        self.total_cost += cost * self.dt

        if self.record:
            self.trace.append(dict(t=self.t, demand_rate=self.sc.arrival_rate(self.t - 1e-9),
                                   true_health=[self.sc.health(i, self.t) for i in range(N_SERVERS)],
                                   reward=r, cost=cost, **st,
                                   service_rate=ctrl.service_rate, active_servers=ctrl.active_servers,
                                   release_interval=ctrl.release_interval, dispatch=ctrl.dispatch,
                                   health_est=list(self.health_est)))

        done = self.t >= self.sc.horizon - 1e-9
        info = self.episode_kpis() if done else {}
        return self._observe(st), r, done, info

    # ------------------------------------------------------------------
    def _clip(self, c: Control) -> Control:
        return Control(service_rate=float(np.clip(c.service_rate, *SERVICE_RATE_BOUNDS)),
                       active_servers=int(np.clip(c.active_servers, 1, N_SERVERS)),
                       release_interval=float(np.clip(c.release_interval, *RELEASE_BOUNDS)),
                       dispatch=int(c.dispatch))

    def _advance(self, t_end):
        c = self.coord
        while c.time_next <= t_end:
            self.clock.time = c.time_next
            c.lambdaf()
            c.deltfcn()
            c.clear()
        self.clock.time = t_end

    def _areas(self, now):
        m = self.model
        busy = [p.busy.area_until(now) for p in m.processors]
        return dict(backlog=m.release.n_backlog.area_until(now),
                    queue=m.buffer.n_queue.area_until(now),
                    busy=sum(busy))

    def _interval_stats(self, empty=False) -> Dict[str, float]:
        m = self.model
        if empty:
            return dict(queue=0.0, backlog=0.0, wip=0.0, util=0.0, throughput=0.0, avg_wait=0.0,
                        arrival_rate=0.0, arrival_trend=0.0, wait_load=0.0, sla_frac=0.0, n_done=0)
        cur = self._areas(self.t)
        d = {k: cur[k] - self._prev[k] for k in cur}
        self._prev = cur

        new_jobs = m.collector.jobs[self._n_jobs_seen:]
        self._n_jobs_seen = len(m.collector.jobs)
        for j in new_jobs:   # 처리시간 실측 → 설비 상태(성능) 추정 (EWMA)
            h = j.expected_time / max(1e-9, j.processing_time)
            self.health_est[j.server] = 0.8 * self.health_est[j.server] + 0.2 * h

        arrived = m.generator.n_arrived
        n_arr, self._prev_arrived = arrived - self._prev_arrived, arrived
        rate = n_arr / self.dt
        self.arr_trend = rate if self.arr_trend is None else 0.6 * self.arr_trend + 0.4 * rate
        n = len(new_jobs)
        return dict(
            queue=float(len(m.buffer.q)),
            backlog=float(len(m.release.backlog)),
            wip=(d["queue"] + d["busy"]) / self.dt,
            util=d["busy"] / (self.ctrl.active_servers * self.dt),
            throughput=n / self.dt,
            avg_wait=float(np.mean([j.waiting_time for j in new_jobs])) if n else 0.0,
            arrival_rate=rate,
            arrival_trend=self.arr_trend,
            wait_load=(d["backlog"] + d["queue"]) / self.dt,
            sla_frac=float(np.mean([j.cycle_time > self.rc.sla for j in new_jobs])) if n else 0.0,
            n_done=n,
        )

    def _reward(self, st, cost) -> float:
        rc = self.rc
        viol = st["sla_frac"] + max(0.0, st["wip"] - rc.wip_max) / rc.wip_max
        return (rc.w_thr * st["throughput"] - rc.w_wait * st["wait_load"] - rc.w_wip * st["wip"]
                - rc.w_cost * cost - rc.w_viol * viol)

    def _observe(self, st) -> np.ndarray:
        lo, hi = SERVICE_RATE_BOUNDS
        rlo, rhi = RELEASE_BOUNDS
        obs = [st["queue"] / 10, st["backlog"] / 10, st["wip"] / 10, st["util"],
               st["throughput"] / BASE_RATE, st["avg_wait"] / 5, st["arrival_rate"] / BASE_RATE, st["arrival_trend"] / BASE_RATE,
               self.ctrl.active_servers / N_SERVERS,
               (self.ctrl.service_rate - lo) / (hi - lo),
               (self.ctrl.release_interval - rlo) / (rhi - rlo),
               *self.health_est]
        return np.clip(np.asarray(obs, dtype=np.float32), 0.0, 5.0)

    # ------------------------------------------------------------------
    def episode_kpis(self) -> Dict[str, float]:
        m, H = self.model, self.sc.horizon
        done = m.collector.jobs
        unfinished = list(m.release.backlog) + list(m.buffer.q) + [p.job for p in m.processors if p.job]
        n_viol = sum(j.cycle_time > self.rc.sla for j in done) + \
                 sum(H - j.creation_time > self.rc.sla for j in unfinished)
        areas = self._areas(H)
        return dict(
            throughput=len(done) / H,
            avg_wait=float(np.mean([j.waiting_time for j in done])) if done else float("nan"),
            avg_cycle=float(np.mean([j.cycle_time for j in done])) if done else float("nan"),
            avg_wip=(areas["queue"] + areas["busy"]) / H,
            resource_cost=self.total_cost / H,
            sla_violation=n_viol / max(1, len(done) + len(unfinished)),
            unfinished=len(unfinished),
            total_reward=self.total_reward,
        )


def config_dict(rc: RewardConfig) -> dict:
    return asdict(rc)
