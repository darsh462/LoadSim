"""
Experiment driver.

Runs every (router, workload) pair across multiple seeds, aggregates the
results into a pandas DataFrame, saves raw CSVs, and generates plots.

Invoked from the project root as:
    python -m experiments.run_experiments
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Make `src` importable when running as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.learned_model import build_training_set, train_model
from src.node import Network, Node
from src.routers import (
    LatencyAwareRouter, LeastQueueRouter, LearnedRouter,
    Power2ChoicesRouter, RandomRouter, RoundRobinRouter, make_router,
)
from src.simulator import SimConfig, Simulator
from src.workload import (
    BurstyWorkload, UniformWorkload, ZipfianWorkload, make_workload,
)


def hot_key_size_fn(n_keys: int = 500,
                    heavy_frac: float = 0.02,
                    heavy_mult: float = 4.0):
    """Return a callable key_id -> service-time multiplier.

    The top `heavy_frac` keys (by id, which matches Zipf's rank ordering) have
    service time scaled by `heavy_mult`; the rest are unit-cost. This models a
    production scenario in which a handful of hot keys correspond to
    large-payload objects or expensive computations, so skewed traffic onto
    those keys translates into concentrated load even with perfect routing.
    """
    n_heavy = max(1, int(n_keys * heavy_frac))
    def _fn(key: int) -> float:
        return heavy_mult if key < n_heavy else 1.0
    return _fn


# ------------------------------------------------------------------ cluster factories

def homogeneous_cluster(n: int = 4, rate: float = 100.0) -> List[Node]:
    """All nodes identical - the classical setup."""
    return [Node(i, service_rate=rate) for i in range(n)]


def heterogeneous_cluster(n: int = 4, base: float = 100.0) -> List[Node]:
    """Mixed-speed cluster: half fast, half slow. Exposes static routers."""
    rates = []
    for i in range(n):
        rates.append(base * (1.5 if i % 2 == 0 else 0.6))
    return [Node(i, service_rate=r) for i, r in zip(range(n), rates)]


# ------------------------------------------------------------------ workload factory

def workload_configs(arrival_rate: float) -> Dict[str, dict]:
    """Return a dict of scenario -> kwargs that will be passed to
    `make_workload` (with an extra 'name' field)."""
    return {
        "uniform": dict(name="uniform", arrival_rate=arrival_rate, n_keys=500),
        "zipf_mild": dict(name="zipfian", arrival_rate=arrival_rate,
                          n_keys=500, skew=0.8),
        "zipf_heavy": dict(name="zipfian", arrival_rate=arrival_rate,
                           n_keys=500, skew=1.4),
        "bursty": dict(name="bursty", arrival_rate=arrival_rate,
                       low_rate=arrival_rate * 0.4,
                       high_rate=arrival_rate * 1.6,
                       low_duration=4.0, high_duration=2.0,
                       n_keys=500, skew=0.6),
    }


# ------------------------------------------------------------------ train learned model

def train_learned_router_model(arrival_rate: float,
                               duration: float,
                               seed: int = 0):
    """Collect traces under a mix of teacher policies, then fit a regressor.

    Using multiple teachers diversifies the training distribution so the
    learned model doesn't overfit to a single policy's trajectory.
    """
    teachers = [("least_queue", LeastQueueRouter()),
                ("latency_aware", LatencyAwareRouter()),
                ("power2", Power2ChoicesRouter(seed=seed))]

    all_records = []
    n_nodes = 4
    for _, teacher in teachers:
        for wl_key in ("uniform", "zipf_mild", "zipf_heavy", "bursty"):
            cfg = workload_configs(arrival_rate)[wl_key]
            kwargs = {k: v for k, v in cfg.items() if k != "name"}
            wl = make_workload(cfg["name"], **kwargs)
            nodes = homogeneous_cluster(n_nodes)
            n_keys = int(kwargs.get("n_keys", 500))
            if wl_key == "uniform":
                size_fn = None
            elif wl_key == "zipf_mild":
                size_fn = hot_key_size_fn(n_keys, heavy_frac=0.02, heavy_mult=1.8)
            elif wl_key == "zipf_heavy":
                size_fn = hot_key_size_fn(n_keys, heavy_frac=0.01, heavy_mult=1.8)
            else:
                size_fn = hot_key_size_fn(n_keys, heavy_frac=0.02, heavy_mult=1.5)
            sim = Simulator(
                nodes, Network(mean_delay=0.001, jitter=0.0005),
                teacher, wl,
                SimConfig(duration=duration, seed=seed + 7,
                          key_size_fn=size_fn),
            )
            sim.run()
            all_records.extend(sim.metrics.records)

    X, y = build_training_set(all_records, n_nodes=n_nodes)
    print(f"  trained on {X.shape[0]} samples, feat dim {X.shape[1]}")

    # Try decision tree first, fall back to ridge.
    model = train_model(X, y, kind="tree")
    return model, X.shape[0]


# ------------------------------------------------------------------ single-run wrapper

@dataclass
class RunResult:
    router: str
    workload: str
    seed: int
    cluster: str
    summary: Dict[str, float]


def run_one(router_name: str,
            workload_name: str,
            workload_cfg: dict,
            arrival_rate: float,
            duration: float,
            seed: int,
            model,
            cluster_name: str = "homogeneous",
            n_nodes: int = 4,
            base_rate: float = 100.0) -> RunResult:
    if cluster_name == "homogeneous":
        nodes = homogeneous_cluster(n_nodes, rate=base_rate)
    else:
        nodes = heterogeneous_cluster(n_nodes, base=base_rate)

    router = make_router(router_name, n_nodes=n_nodes, seed=seed, model=model)
    kwargs = {k: v for k, v in workload_cfg.items() if k != "name"}
    wl = make_workload(workload_cfg["name"], **kwargs)

    # Skewed workloads use a hot-key size model; uniform stays unit-cost.
    # The multipliers are chosen so that the system is near-saturation but not
    # in runaway overload - that's where the interesting routing differences
    # appear. Expected mean service cost per request (top-1% @5x with s=1.4
    # Zipf): ~0.35*1.8 + 0.65*1 = 1.28 units, giving ~410 req/s of work at
    # 320 req/s arrival, against 400 req/s of aggregate capacity.
    n_keys = int(kwargs.get("n_keys", 500))
    if workload_name == "uniform":
        size_fn = None
    elif workload_name == "zipf_mild":
        size_fn = hot_key_size_fn(n_keys, heavy_frac=0.02, heavy_mult=1.8)
    elif workload_name == "zipf_heavy":
        size_fn = hot_key_size_fn(n_keys, heavy_frac=0.01, heavy_mult=1.8)
    else:  # bursty
        size_fn = hot_key_size_fn(n_keys, heavy_frac=0.02, heavy_mult=1.5)

    sim = Simulator(
        nodes, Network(mean_delay=0.001, jitter=0.0005),
        router, wl,
        SimConfig(duration=duration, seed=seed, key_size_fn=size_fn),
    )
    sim.run()
    s = sim.metrics.summary()
    # Also compute per-node utilisation.
    util = {f"util_node_{n.node_id}": n.utilization(sim.metrics.sim_duration)
            for n in nodes}
    s.update(util)
    return RunResult(router=router_name, workload=workload_name,
                     seed=seed, cluster=cluster_name, summary=s)


# ------------------------------------------------------------------ main sweep

def run_all(out_dir: str = "results",
            arrival_rate: float = 320.0,
            duration: float = 30.0,
            seeds: Tuple[int, ...] = (1, 2, 3),
            routers: Tuple[str, ...] = ("random", "round_robin", "least_queue",
                                        "power2", "latency_aware", "learned"),
            workloads: Tuple[str, ...] = ("uniform", "zipf_mild",
                                          "zipf_heavy", "bursty"),
            clusters: Tuple[str, ...] = ("homogeneous", "heterogeneous"),
            n_nodes: int = 4,
            base_rate: float = 100.0) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Training learned-router model...")
    model, n_train = train_learned_router_model(arrival_rate, duration * 0.5,
                                                seed=0)
    print(f"[{time.strftime('%H:%M:%S')}] Model trained on {n_train} samples.")

    cfgs = workload_configs(arrival_rate)

    rows = []
    total = len(routers) * len(workloads) * len(seeds) * len(clusters)
    done = 0
    t0 = time.time()
    for cluster in clusters:
        for wl_key in workloads:
            for router_name in routers:
                for seed in seeds:
                    res = run_one(
                        router_name, wl_key, cfgs[wl_key],
                        arrival_rate=arrival_rate, duration=duration,
                        seed=seed, model=model,
                        cluster_name=cluster, n_nodes=n_nodes,
                        base_rate=base_rate,
                    )
                    row = {
                        "router": res.router,
                        "workload": res.workload,
                        "seed": res.seed,
                        "cluster": res.cluster,
                        **res.summary,
                    }
                    rows.append(row)
                    done += 1
                    if done % 5 == 0 or done == total:
                        dt = time.time() - t0
                        print(f"  [{done:3d}/{total}] "
                              f"{cluster}/{wl_key}/{router_name}/s{seed} "
                              f"({dt:.1f}s)")

    df = pd.DataFrame(rows)
    raw_csv = os.path.join(out_dir, "raw_results.csv")
    df.to_csv(raw_csv, index=False)
    print(f"Wrote raw results -> {raw_csv}")

    # Aggregated (mean across seeds) table.
    agg_cols = [c for c in df.columns
                if c not in ("router", "workload", "seed", "cluster")]
    agg = (df.groupby(["cluster", "workload", "router"])[agg_cols]
             .mean().reset_index())
    agg_csv = os.path.join(out_dir, "aggregated.csv")
    agg.to_csv(agg_csv, index=False)
    print(f"Wrote aggregated -> {agg_csv}")

    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--arrival-rate", type=float, default=320.0,
                   help="aggregate arrival rate in requests/sec")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--quick", action="store_true",
                   help="small sweep for quick sanity checks")
    args = p.parse_args()

    if args.quick:
        run_all(out_dir=args.out, arrival_rate=args.arrival_rate,
                duration=10.0, seeds=(1,))
    else:
        run_all(out_dir=args.out, arrival_rate=args.arrival_rate,
                duration=args.duration, seeds=tuple(args.seeds))
