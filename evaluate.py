"""
평가/분석 스크립트: Fixed vs Static-Opt vs PPO (+ 보상 설계 v1 vs v2 비교)
 - 평가 시나리오: Normal / Demand Surge(t=200~400, 수요 +30%) / Machine Degradation(t≥200, P0 성능 -30%)
 - 모든 방법을 동일 시드(공통 난수)에서 trade-off 보상(v2) 기준으로 평가
 - 출력: results/results.md, results/kpi_runs.csv, results/*.png
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gbp_env import GBPEnv, REWARD_PRESETS, OBS_DIM, N_SERVERS
from scenarios import SCENARIOS, make_scenario
from baselines import FixedPolicy, PPOPolicy, load_static_best
from PPO import PPOAgent
from xdevs_Buffer import DISPATCH_NAMES

N_SEEDS = 30
EVAL_SEED0 = 50_000
OUT = "results"
SC_LABEL = {"normal": "Normal", "demand_surge": "Demand Surge", "machine_degradation": "Machine Degradation"}
KPIS = [("throughput", "Throughput ↑", "{:.3f}"), ("avg_wait", "Waiting ↓", "{:.3f}"),
        ("avg_wip", "WIP ↓", "{:.2f}"), ("resource_cost", "Resource Cost ↓", "{:.3f}"),
        ("sla_violation", "SLA Violation ↓", "{:.2%}"), ("total_reward", "Reward ↑", "{:.2f}")]

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


def run(env, policy, scenario, seed):
    obs = env.reset(seed=seed, scenario=scenario)
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


def evaluate(methods):
    env = GBPEnv(reward=REWARD_PRESETS["tradeoff"], record=True)
    rows, traces, jobs = [], {}, {}
    for sc in SCENARIOS:
        for name, pol in methods.items():
            for i in range(N_SEEDS):
                info, tr, jb = run(env, pol, make_scenario(sc), EVAL_SEED0 + i)
                rows.append(dict(scenario=sc, method=name, seed=i, **info))
                if i == 0:
                    traces[(sc, name)], jobs[(sc, name)] = tr, jb
        print(f"evaluated {sc}")
    return rows, traces, jobs


def agg(rows, sc=None, method=None, key="total_reward"):
    v = [r[key] for r in rows if (sc is None or r["scenario"] == sc) and (method is None or r["method"] == method)]
    return float(np.mean(v)), float(np.std(v))


# ----------------------------------------------------------------------------- tables
def phase_actions(traces, sc, name, phases):
    tr = traces[(sc, name)]
    out = []
    for lo, hi in phases:
        seg = [x for x in tr if lo < x["t"] <= hi]
        out.append((np.mean([x["active_servers"] for x in seg]), np.mean([x["service_rate"] for x in seg]),
                    np.mean([x["dispatch"] for x in seg])))
    return out


def server_share(job_list, t0):
    js = [j for j in job_list if j.start_time >= t0]
    c = np.bincount([j.server for j in js], minlength=N_SERVERS)
    return c / max(1, c.sum())


def write_report(rows, traces, jobs, methods):
    main = [m for m in ("Fixed", "Static-Opt", "PPO") if m in methods]
    L = ["# 평가 결과", "",
         f"- 시나리오별 {N_SEEDS}개 시드(공통 난수), 에피소드 길이 600, 의사결정 주기 10",
         "- 모든 방법은 동일한 trade-off 보상(v2)으로 채점", ""]

    L += ["## 1. 방법별 KPI (3개 시나리오 평균)", "",
          "| Method | " + " | ".join(k[1] for k in KPIS) + " |",
          "|---|" + "---|" * len(KPIS)]
    for m in main:
        L.append(f"| {m} | " + " | ".join(f.format(agg(rows, None, m, k)[0]) for k, _, f in KPIS) + " |")

    L += ["", "## 2. 시나리오별 보상 (mean ± std)", "",
          "| Scenario | " + " | ".join(main) + " |", "|---|" + "---|" * len(main)]
    for sc in SCENARIOS:
        L.append(f"| {SC_LABEL[sc]} | " + " | ".join("{:.2f} ± {:.2f}".format(*agg(rows, sc, m)) for m in main) + " |")

    L += ["", "## 3. 시나리오별 상세 KPI", ""]
    for sc in SCENARIOS:
        L += [f"**{SC_LABEL[sc]}**", "", "| Method | " + " | ".join(k[1] for k in KPIS) + " |",
              "|---|" + "---|" * len(KPIS)]
        for m in main:
            L.append(f"| {m} | " + " | ".join(f.format(agg(rows, sc, m, k)[0]) for k, _, f in KPIS) + " |")
        L.append("")

    if "PPO (v1 reward)" in methods:
        L += ["## 4. 보상 설계 비교: v1(성능 지표만) vs v2(trade-off)", "",
              "| Reward | 평균 가동 대수 | 평균 Service Rate | Waiting ↓ | Resource Cost ↓ | Reward(v2 기준) ↑ |",
              "|---|---|---|---|---|---|"]
        for m, lab in (("PPO (v1 reward)", "v1: T − W"), ("PPO", "v2: T − W − WIP − C − V")):
            L.append(f"| {lab} | {agg(rows, None, m, 'mean_servers')[0]:.2f} | {agg(rows, None, m, 'mean_rate')[0]:.3f} | "
                     f"{agg(rows, None, m, 'avg_wait')[0]:.3f} | {agg(rows, None, m, 'resource_cost')[0]:.3f} | "
                     f"{agg(rows, None, m, 'total_reward')[0]:.2f} |")
        L.append("")

    if "PPO" in methods:
        L += ["## 5. PPO의 상황별 의사결정 (seed 0)", "",
              "| Scenario | 구간 | 평균 가동 대수 | 평균 Service Rate | HEALTH 디스패칭 비율 |", "|---|---|---|---|---|"]
        for sc, phases, labels in (
                ("demand_surge", [(0, 200), (200, 400), (400, 600)], ["평상시 (0~200)", "수요 급증 (200~400)", "수요 복귀 (400~600)"]),
                ("machine_degradation", [(0, 200), (200, 600)], ["정상 (0~200)", "P0 성능 저하 (200~600)"]),
                ("normal", [(0, 600)], ["전체"])):
            for (k, r, d), lab in zip(phase_actions(traces, sc, "PPO", phases), labels):
                L.append(f"| {SC_LABEL[sc]} | {lab} | {k:.2f} | {r:.3f} | {d:.0%} |")
        L += ["", "**설비 성능 저하 이후(t≥200) Processor별 작업 분담 비율 (seed 0)**", "",
              "| Method | " + " | ".join(f"P{i}" for i in range(N_SERVERS)) + " |", "|---|" + "---|" * N_SERVERS]
        for m in main:
            share = server_share(jobs[("machine_degradation", m)], 200.0)
            L.append(f"| {m} | " + " | ".join(f"{x:.0%}" for x in share) + " |")
        L.append("")

    L += ["## 그림", "", "![training](training_curve.png)", "", "![dynamic](dynamic_response.png)", "",
          "![kpi](kpi_comparison.png)", ""]
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


def plot_dynamic(traces, methods):
    names = [m for m in ("Fixed", "Static-Opt", "PPO") if m in methods]
    rows = [("demand_rate", "수요 도착률 λ(t)"), ("active_servers", "가동 Processor 수"),
            ("service_rate", "Service Rate"), ("wait_load", "대기 작업 수 (시간평균)")]
    fig, axes = plt.subplots(len(rows), len(SCENARIOS), figsize=(14, 9), sharex=True)
    for j, sc in enumerate(SCENARIOS):
        for i, (key, lab) in enumerate(rows):
            ax = axes[i, j]
            if key == "demand_rate":
                tr = traces[(sc, names[0])]
                ax.step([x["t"] for x in tr], [x[key] for x in tr], where="pre", color=INK2)
                if sc == "machine_degradation":
                    ax.axvline(200, color=INK2, ls="--", lw=1)
                    ax.text(205, 1.55, "P0 성능 -30%", color=INK2, fontsize=8, va="bottom")
            else:
                # Static-Opt는 PPO와 겹치는 구간이 많아 마지막에 점선으로 그림
                for n in sorted(names, key=lambda m: m == "Static-Opt"):
                    tr = traces[(sc, n)]
                    ax.step([x["t"] for x in tr], [x[key] for x in tr], where="pre", color=COLOR[n], label=n,
                            lw=2 if n == "PPO" else 1.5, ls="--" if n == "Static-Opt" else "-")
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
            if i == 0:
                ax.set_title(SC_LABEL[sc], loc="left")
            if i == len(rows) - 1:
                ax.set_xlabel("simulation time")
    axes[1, 0].legend(fontsize=8, loc="upper left")
    for ax in axes[0]:
        ax.set_ylim(1.2, 2.1)
    for ax in axes[1]:
        ax.set_ylim(0.5, N_SERVERS + 0.5); ax.set_yticks(range(1, N_SERVERS + 1))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dynamic_response.png"), dpi=140)
    plt.close(fig)


def plot_kpis(rows, methods):
    names = [m for m in ("Fixed", "Static-Opt", "PPO") if m in methods]
    kp = [("avg_wait", "평균 대기시간 ↓"), ("resource_cost", "자원 비용률 ↓"), ("total_reward", "보상 (v2) ↑")]
    fig, axes = plt.subplots(1, len(kp), figsize=(14, 3.8))
    w = 0.8 / len(names)
    x = np.arange(len(SCENARIOS))
    for ax, (key, title) in zip(axes, kp):
        for k, n in enumerate(names):
            vals = [agg(rows, sc, n, key)[0] for sc in SCENARIOS]
            ax.bar(x + (k - (len(names) - 1) / 2) * w, vals, w * 0.92, color=COLOR[n], label=n)
        ax.set_xticks(x, [SC_LABEL[s] for s in SCENARIOS], fontsize=8)
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
    with open(os.path.join(OUT, "kpi_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    write_report(rows, traces, jobs, methods)
    plot_training()
    plot_dynamic(traces, methods)
    plot_kpis(rows, methods)
