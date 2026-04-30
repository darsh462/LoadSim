"""
Plot generation.

Reads `results/raw_results.csv` and `results/aggregated.csv` produced by
`experiments/run_experiments.py` and produces:

    plots/latency_bars.png        mean / p95 / p99 by router and workload
    plots/tail_comparison.png     p99 across routers in each workload
    plots/throughput.png          sustained throughput per router
    plots/fairness.png            Jain's fairness index
    plots/heterogeneous.png       homogeneous vs heterogeneous comparison
    plots/latency_cdf_*.png       per-scenario latency CDFs (from raw traces)

Invoke:
    python -m experiments.plot_results
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Use a non-interactive backend so this works in any environment.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


ROUTER_ORDER = ["random", "round_robin", "least_queue",
                "power2", "latency_aware", "learned"]

ROUTER_LABELS = {
    "random":        "Random",
    "round_robin":   "Round-Robin",
    "least_queue":   "Least-Queue (JSQ)",
    "power2":        "Power-of-2",
    "latency_aware": "Latency-Aware",
    "learned":       "Learned (ML)",
}

ROUTER_COLORS = {
    "random":        "#888888",
    "round_robin":   "#4c72b0",
    "least_queue":   "#55a868",
    "power2":        "#8172b2",
    "latency_aware": "#c44e52",
    "learned":       "#dd8452",
}


WORKLOAD_LABELS = {
    "uniform":    "Uniform",
    "zipf_mild":  "Zipf (mild, s=0.8)",
    "zipf_heavy": "Zipf (heavy, s=1.4)",
    "bursty":     "Bursty",
}


def _save(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_latency_bars(agg: pd.DataFrame, plots_dir: str,
                      cluster: str = "homogeneous") -> None:
    d = agg[agg["cluster"] == cluster]
    metrics = [("mean_latency", "Mean latency (s)"),
               ("p95_latency", "p95 latency (s)"),
               ("p99_latency", "p99 latency (s)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    workloads = list(WORKLOAD_LABELS.keys())

    for ax, (col, title) in zip(axes, metrics):
        x = np.arange(len(workloads))
        width = 0.13
        for i, r in enumerate(ROUTER_ORDER):
            vals = []
            for w in workloads:
                sub = d[(d["router"] == r) & (d["workload"] == w)]
                vals.append(float(sub[col].iloc[0]) if len(sub) else 0.0)
            ax.bar(x + (i - len(ROUTER_ORDER) / 2) * width + width / 2,
                   vals, width, label=ROUTER_LABELS[r],
                   color=ROUTER_COLORS[r], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([WORKLOAD_LABELS[w] for w in workloads],
                           rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel("seconds")
        ax.grid(True, axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Latency across routers ({cluster} cluster)",
                 fontsize=14, y=1.02)
    _save(fig, os.path.join(plots_dir, f"latency_bars_{cluster}.png"))


def plot_tail_comparison(agg: pd.DataFrame, plots_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cluster in zip(axes, ("homogeneous", "heterogeneous")):
        d = agg[agg["cluster"] == cluster]
        workloads = list(WORKLOAD_LABELS.keys())
        x = np.arange(len(workloads))
        width = 0.13
        for i, r in enumerate(ROUTER_ORDER):
            vals = []
            for w in workloads:
                sub = d[(d["router"] == r) & (d["workload"] == w)]
                vals.append(float(sub["p99_latency"].iloc[0])
                           if len(sub) else 0.0)
            ax.bar(x + (i - len(ROUTER_ORDER) / 2) * width + width / 2,
                   vals, width, label=ROUTER_LABELS[r],
                   color=ROUTER_COLORS[r], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([WORKLOAD_LABELS[w] for w in workloads],
                           rotation=15, ha="right")
        ax.set_ylabel("p99 latency (s)")
        ax.set_title(f"{cluster.capitalize()} cluster")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Tail latency (p99) comparison", fontsize=14, y=1.02)
    _save(fig, os.path.join(plots_dir, "tail_comparison.png"))


def plot_throughput(agg: pd.DataFrame, plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    d = agg[agg["cluster"] == "homogeneous"]
    workloads = list(WORKLOAD_LABELS.keys())
    x = np.arange(len(workloads))
    width = 0.13
    for i, r in enumerate(ROUTER_ORDER):
        vals = []
        for w in workloads:
            sub = d[(d["router"] == r) & (d["workload"] == w)]
            vals.append(float(sub["throughput"].iloc[0]) if len(sub) else 0.0)
        ax.bar(x + (i - len(ROUTER_ORDER) / 2) * width + width / 2,
               vals, width, label=ROUTER_LABELS[r],
               color=ROUTER_COLORS[r], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_LABELS[w] for w in workloads],
                       rotation=15, ha="right")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Sustained throughput (homogeneous cluster)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, os.path.join(plots_dir, "throughput.png"))


def plot_fairness(agg: pd.DataFrame, plots_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cluster in zip(axes, ("homogeneous", "heterogeneous")):
        d = agg[agg["cluster"] == cluster]
        workloads = list(WORKLOAD_LABELS.keys())
        x = np.arange(len(workloads))
        width = 0.13
        for i, r in enumerate(ROUTER_ORDER):
            vals = []
            for w in workloads:
                sub = d[(d["router"] == r) & (d["workload"] == w)]
                vals.append(float(sub["fairness"].iloc[0]) if len(sub) else 0.0)
            ax.bar(x + (i - len(ROUTER_ORDER) / 2) * width + width / 2,
                   vals, width, label=ROUTER_LABELS[r],
                   color=ROUTER_COLORS[r], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([WORKLOAD_LABELS[w] for w in workloads],
                           rotation=15, ha="right")
        ax.set_ylabel("Jain's fairness index")
        ax.set_title(f"{cluster.capitalize()} cluster")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Load fairness across nodes (1.0 = perfectly balanced)",
                 fontsize=14, y=1.02)
    _save(fig, os.path.join(plots_dir, "fairness.png"))


def plot_utilization_heatmap(agg: pd.DataFrame, plots_dir: str) -> None:
    """Show node-level utilization as a heatmap for the heterogeneous cluster."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    d = agg[agg["cluster"] == "heterogeneous"]
    util_cols = [c for c in agg.columns if c.startswith("util_node_")]
    if not util_cols:
        return
    util_cols = sorted(util_cols, key=lambda c: int(c.split("_")[-1]))
    workloads = list(WORKLOAD_LABELS.keys())

    for ax, wl in zip(axes.flat, workloads):
        sub = d[d["workload"] == wl].set_index("router")
        # Reindex in our preferred order
        sub = sub.reindex([r for r in ROUTER_ORDER if r in sub.index])
        mat = sub[util_cols].to_numpy()
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(util_cols)))
        ax.set_xticklabels([f"Node {i}" for i in range(len(util_cols))])
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels([ROUTER_LABELS[r] for r in sub.index])
        ax.set_title(WORKLOAD_LABELS[wl])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}",
                        ha="center", va="center",
                        color="black" if mat[i,j] < 0.6 else "white",
                        fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle("Per-node utilization (heterogeneous cluster)",
                 fontsize=14, y=1.00)
    _save(fig, os.path.join(plots_dir, "utilization_heatmap.png"))


def make_all_plots(results_dir: str = "results",
                   plots_dir: str = "plots") -> None:
    os.makedirs(plots_dir, exist_ok=True)
    agg_path = os.path.join(results_dir, "aggregated.csv")
    if not os.path.exists(agg_path):
        raise FileNotFoundError(
            f"{agg_path} not found - run experiments first.")
    agg = pd.read_csv(agg_path)
    print(f"Plotting from {agg_path} ({len(agg)} rows)")

    plot_latency_bars(agg, plots_dir, cluster="homogeneous")
    plot_latency_bars(agg, plots_dir, cluster="heterogeneous")
    plot_tail_comparison(agg, plots_dir)
    plot_throughput(agg, plots_dir)
    plot_fairness(agg, plots_dir)
    plot_utilization_heatmap(agg, plots_dir)
    print("All plots written.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--plots", default="plots")
    args = p.parse_args()
    make_all_plots(args.results, args.plots)
