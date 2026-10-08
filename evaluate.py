"""
평가/분석 스크립트: Fixed vs Static-Opt vs PPO (+ 보상 설계 v1 vs v2 비교)
 - 평가 데이터: ERP 주문 평가 기간(2011-07 ~ 2011-12)을 5영업일 구간으로 나눠 전부 사용
 - 시나리오: ERP 수요 그대로 (비수기 7~9월 / 성수기 10~12월로 구분) + 설비 성능 저하(3일차부터 P0 -30%)
 - 모든 방법을 동일 구간·동일 시드에서 trade-off 보상(v2) 기준으로 평가
 - 출력: results/results.md, results/kpi_runs.csv, results/*.png
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gbp_env import GBPEnv, REWARD_PRESETS, OBS_DIM, N_SERVERS, DT
from scenarios import SCENARIOS, EPISODE_DAYS, make_scenario, test_windows
from erp_data import order_book, SIM_PER_DAY, DAY_START_HOUR
from baselines import FixedPolicy, PPOPolicy, load_static_best
from PPO import PPOAgent

N_SEEDS = 3
EVAL_SEED0 = 50_000
OUT = "results"
GROUPS = ("offpeak", "peak", "degradation")
GROUP_LABEL = {"offpeak": "비수기 (7~9월)", "peak": "성수기 (10~12월)", "degradation": "설비 성능 저하"}
KPIS = [("throughput", "Throughput ↑", "{:.3f}"), ("avg_wait", "Waiting ↓", "{:.3f}"),
        ("avg_wip", "WIP ↓", "{:.2f}"), ("resource_cost", "Resource Cost ↓", "{:.3f}"),
        ("sla_violation", "SLA Violation ↓", "{:.2%}"), ("total_reward", "Reward ↑", "{:.2f}")]
STEPS_PER_DAY = int(SIM_PER_DAY / DT)

# 검증된 categorical 팔레트(dataviz reference) - 방법별 색 고정
COLOR = {"PPO": "#2a78d6", "Static-Opt": "#eb6834", "Fixed": "#1baf7a", "PPO (v1 reward)": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def run(env, policy, scenario):
    obs = env.reset(scenario=scenario)
    done, info = False, {}
    while not done:
        obs, _, done, info = env.step(policy(obs))
    tr = env.trace
    info["mean_servers"] = float(np.mean([x["active_servers"] for x in tr]))
    info["mean_rate"] = float(np.mean([x["service_rate"] for x in tr]))
    return info, tr, list(env.model.collector.jobs)


def load_methods():
    methods = {"Fixed": FixedPolicy()}
    if os.path.exists("runs/static_opt.json"):
        p = FixedPolicy(load_static_best("runs/static_opt.json")); p.name = "Static-Opt"
        methods["Static-Opt"] = p
    for reward, name in (("tradeoff", "PPO"), ("performance", "PPO (v1 reward)")):
        path = f"runs/ppo_{reward}/model.pt"
        if os.path.exists(path):
            methods[name] = PPOPolicy(PPOAgent(OBS_DIM).load(path))
    return methods


def group_of(sc_name, start_day):
    if sc_name == "erp_degradation":
        return "degradation"
    return "peak" if order_book().days[start_day].month >= 10 else "offpeak"


def evaluate(methods):
    env = GBPEnv(reward=REWARD_PRESETS["tradeoff"], record=True)
    rows, traces, jobs = [], {}, {}
    windows = test_windows()
    for sc in SCENARIOS:
        for w in windows:
            g = group_of(sc, w)
            for name, pol in methods.items():
                for i in range(N_SEEDS):
                    scen = make_scenario(sc, rng=np.random.default_rng(EVAL_SEED0 + i), start_day=w)
                    env.rng = np.random.default_rng(EVAL_SEED0 + 1000 * w + i)   # 처리 노이즈 시드 (방법 간 공통)
                    info, tr, jb = run(env, pol, scen)
                    rows.append(dict(group=g, window=scen.period, method=name, seed=i, **info))
                    if i == 0:
                        traces[(g, name, w)], jobs[(g, name, w)] = tr, jb
        print(f"evaluated {sc}: {len(windows)} windows")
    return rows, traces, jobs


def agg(rows, g=None, method=None, key="total_reward"):
    v = [r[key] for r in rows if (g is None or r["group"] == g) and (method is None or r["method"] == method)]
    return float(np.mean(v)), float(np.std(v))


# ----------------------------------------------------------------------------- tables
def by_hour(traces, method, groups=("offpeak", "peak")):
    """시간대별 평균 가동 대수 / 시간당 주문 도착 (평가 구간 전체, seed 0)"""
    srv = np.zeros(STEPS_PER_DAY); arr = np.zeros(STEPS_PER_DAY); n = np.zeros(STEPS_PER_DAY)
    for (g, m, _), tr in traces.items():
        if m != method or g not in groups:
            continue
        for i, x in enumerate(tr):
            h = i % STEPS_PER_DAY
            srv[h] += x["active_servers"]; arr[h] += x["demand_rate"]; n[h] += 1
    return srv / np.maximum(n, 1), arr / np.maximum(n, 1)


def server_share(job_list, t0):
    js = [j for j in job_list if j.start_time >= t0]
    c = np.bincount([j.server for j in js], minlength=N_SERVERS)
    return c / max(1, c.sum())


def write_report(rows, traces, jobs, methods):
    main = [m for m in ("Fixed", "Static-Opt", "PPO") if m in methods]
    n_win = len(test_windows())
    L = ["# 평가 결과", "",
         f"- 수요: ERP 주문 데이터(UCI Online Retail II) 평가 기간 2011-07 ~ 2011-12, 5영업일 구간 {n_win}개 × 시드 {N_SEEDS}개",
         "- 시간축: 영업일 07~21시, 1 sim time = 12분, 의사결정 주기 = 1시간",
         "- 모든 방법은 동일한 trade-off 보상(v2)으로 채점", ""]

    L += ["## 1. 방법별 KPI (전체 평균)", "",
          "| Method | " + " | ".join(k[1] for k in KPIS) + " |", "|---|" + "---|" * len(KPIS)]
    for m in main:
        L.append(f"| {m} | " + " | ".join(f.format(agg(rows, None, m, k)[0]) for k, _, f in KPIS) + " |")

    L += ["", "## 2. 구간별 보상 (mean ± std)", "",
          "| 구간 | " + " | ".join(main) + " |", "|---|" + "---|" * len(main)]
    for g in GROUPS:
        L.append(f"| {GROUP_LABEL[g]} | " + " | ".join("{:.2f} ± {:.2f}".format(*agg(rows, g, m)) for m in main) + " |")

    L += ["", "## 3. 구간별 상세 KPI", ""]
    for g in GROUPS:
        L += [f"**{GROUP_LABEL[g]}**", "", "| Method | " + " | ".join(k[1] for k in KPIS) + " |",
              "|---|" + "---|" * len(KPIS)]
        for m in main:
            L.append(f"| {m} | " + " | ".join(f.format(agg(rows, g, m, k)[0]) for k, _, f in KPIS) + " |")
        L.append("")

    if "PPO (v1 reward)" in methods:
        L += ["## 4. 보상 설계 비교: v1(성능 지표만) vs v2(trade-off)", "",
              "| Reward | 평균 가동 대수 | 평균 Service Rate | Waiting ↓ | Resource Cost ↓ | Reward(v2 기준) ↑ |",
              "|---|---|---|---|---|---|"]
        for m, lab in (("PPO (v1 reward)", "v1: T - W"), ("PPO", "v2: T - W - WIP - C - V")):
            L.append(f"| {lab} | {agg(rows, None, m, 'mean_servers')[0]:.2f} | {agg(rows, None, m, 'mean_rate')[0]:.3f} | "
                     f"{agg(rows, None, m, 'avg_wait')[0]:.3f} | {agg(rows, None, m, 'resource_cost')[0]:.3f} | "
                     f"{agg(rows, None, m, 'total_reward')[0]:.2f} |")
        L.append("")

    if "PPO" in methods:
        srv, arr = by_hour(traces, "PPO")
        L += ["## 5. PPO의 시간대별 의사결정 (설비 저하 제외 평가 구간 평균)", "",
              "| 시간대 | 시간당 주문 도착 (로트/sim time) | PPO 평균 가동 대수 |", "|---|---|---|"]
        for h in range(STEPS_PER_DAY):
            L.append(f"| {DAY_START_HOUR + h:02d}:00~{DAY_START_HOUR + h + 1:02d}:00 | {arr[h]:.2f} | {srv[h]:.2f} |")
        if "Static-Opt" in methods:
            b = load_static_best("runs/static_opt.json")
            L += ["", f"Static-Opt는 모든 시간대에 {b.active_servers}대 × 속도 {b.service_rate:.1f} 고정"]
        t0 = 2 * SIM_PER_DAY
        L += ["", "**설비 성능 저하 이후(3일차~) Processor별 작업 분담 비율 (평가 구간 평균)**", "",
              "| Method | " + " | ".join(f"P{i}" for i in range(N_SERVERS)) + " |", "|---|" + "---|" * N_SERVERS]
        for m in main:
            shares = [server_share(jb, t0) for (g, mm, _), jb in jobs.items() if g == "degradation" and mm == m]
            L.append(f"| {m} | " + " | ".join(f"{x:.0%}" for x in np.mean(shares, axis=0)) + " |")
        L.append("")

    L += ["## 그림", "", "![training](training_curve.png)", "", "![dynamic](dynamic_response.png)", "",
          "![hourly](hourly_servers.png)", "", "![kpi](kpi_comparison.png)", ""]
    with open(os.path.join(OUT, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


# ----------------------------------------------------------------------------- plots
def plot_training():
    runs = {}
    for reward, name in (("tradeoff", "PPO"), ("performance", "PPO (v1 reward)")):
        p = f"runs/ppo_{reward}/history.json"
        if os.path.exists(p):
            runs[name] = json.load(open(p))["history"]
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    smooth = lambda y, k=10: np.convolve(y, np.ones(k) / k, mode="valid")
    panels = [("total_reward", "학습 보상 (v2, 이동평균)", ["PPO"]),
              ("mean_servers", "평균 가동 Processor 수", list(runs)),
              ("resource_cost", "평균 자원 비용률", list(runs))]
    for ax, (key, title, names) in zip(axes, panels):
        for n in names:
            if n not in runs:
                continue
            h = runs[n]
            y = smooth([r[key] for r in h])
            x = [r["episodes"] for r in h][len(h) - len(y):]
            ax.plot(x, y, color=COLOR[n], label=n)
            ax.annotate(n, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        color=INK2, fontsize=8, va="center")
        ax.set_title(title, loc="left")
        ax.set_xlabel("episodes")
    axes[1].legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curve.png"), dpi=140)
    plt.close(fig)


def pick_windows(rows):
    """그림용 대표 구간: 비수기 중 수요 최저, 성수기 중 수요 최고, 설비 저하는 성수기 첫 구간"""
    book = order_book()
    windows = test_windows()
    vol = {w: len(make_scenario("erp", start_day=w).orders) for w in windows}
    off = [w for w in windows if book.days[w].month < 10]
    peak = [w for w in windows if book.days[w].month >= 10]
    return [("offpeak", min(off, key=vol.get)), ("peak", max(peak, key=vol.get)), ("degradation", peak[0])]


def plot_dynamic(traces, methods, rows):
    names = [m for m in ("Fixed", "Static-Opt", "PPO") if m in methods]
    picks = pick_windows(rows)
    panels = [("demand_rate", "주문 도착률 (시간당)"), ("active_servers", "가동 Processor 수"),
              ("service_rate", "Service Rate"), ("wait_load", "대기 작업 수 (시간평균)")]
    book = order_book()
    fig, axes = plt.subplots(len(panels), len(picks), figsize=(15, 9), sharex=True)
    for j, (g, w) in enumerate(picks):
        for i, (key, lab) in enumerate(panels):
            ax = axes[i, j]
            if key == "demand_rate":
                tr = traces[(g, names[0], w)]
                ax.step([x["t"] / SIM_PER_DAY for x in tr], [x[key] for x in tr], where="pre", color=INK2)
                if g == "degradation":
                    ax.axvline(2, color=INK2, ls="--", lw=1)
                    ax.text(2.05, ax.get_ylim()[1] * 0.9, "P0 성능 -30%", color=INK2, fontsize=8)
            else:
                for n in sorted(names, key=lambda m: m == "Static-Opt"):
                    tr = traces[(g, n, w)]
                    ax.step([x["t"] / SIM_PER_DAY for x in tr], [x[key] for x in tr], where="pre", color=COLOR[n],
                            label=n, lw=2 if n == "PPO" else 1.5, ls="--" if n == "Static-Opt" else "-")
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
            if i == 0:
                ax.set_title(f"{GROUP_LABEL[g]}  {book.period(w, EPISODE_DAYS)}", loc="left", fontsize=10)
            if i == len(panels) - 1:
                ax.set_xlabel("영업일")
    axes[1, 0].legend(fontsize=8, loc="upper left")
    for ax in axes[1]:
        ax.set_ylim(0.5, N_SERVERS + 0.5); ax.set_yticks(range(1, N_SERVERS + 1))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dynamic_response.png"), dpi=140)
    plt.close(fig)


def plot_hourly(traces, methods):
    if "PPO" not in methods:
        return
    srv, arr = by_hour(traces, "PPO")
    hours = [f"{DAY_START_HOUR + h:02d}" for h in range(STEPS_PER_DAY)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
    axes[0].bar(hours, arr, color=INK2, width=0.7)
    axes[0].set_title("시간대별 평균 주문 도착률 (ERP)", loc="left")
    axes[1].plot(hours, srv, color=COLOR["PPO"], marker="o", ms=5, label="PPO")
    if "Static-Opt" in methods:
        b = load_static_best("runs/static_opt.json")
        axes[1].plot(hours, [b.active_servers] * len(hours), color=COLOR["Static-Opt"], ls="--", lw=1.5,
                     label="Static-Opt")
    axes[1].set_ylim(0.5, N_SERVERS + 0.5)
    axes[1].set_title("시간대별 평균 가동 Processor 수", loc="left")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("시각")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hourly_servers.png"), dpi=140)
    plt.close(fig)


def plot_kpis(rows, methods):
    names = [m for m in ("Fixed", "Static-Opt", "PPO") if m in methods]
    kp = [("avg_wait", "평균 대기시간 ↓"), ("resource_cost", "자원 비용률 ↓"), ("total_reward", "보상 (v2) ↑")]
    fig, axes = plt.subplots(1, len(kp), figsize=(14, 3.8))
    w = 0.8 / len(names)
    x = np.arange(len(GROUPS))
    for ax, (key, title) in zip(axes, kp):
        for k, n in enumerate(names):
            vals = [agg(rows, g, n, key)[0] for g in GROUPS]
            ax.bar(x + (k - (len(names) - 1) / 2) * w, vals, w * 0.92, color=COLOR[n], label=n)
        ax.set_xticks(x, [GROUP_LABEL[g] for g in GROUPS], fontsize=8)
        ax.set_title(title, loc="left")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kpi_comparison.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    methods = load_methods()
    print("methods:", list(methods))
    rows, traces, jobs = evaluate(methods)
    with open(os.path.join(OUT, "kpi_runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    write_report(rows, traces, jobs, methods)
    plot_training()
    plot_dynamic(traces, methods, rows)
    plot_hourly(traces, methods)
    plot_kpis(rows, methods)
