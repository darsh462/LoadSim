<div align="center">

# LoadSim

### A high-fidelity discrete-event simulator for distributed load balancing

*Comparing classical and ML-based routing policies under realistic workloads — without spinning up a single server.*

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![Made with NumPy](https://img.shields.io/badge/made%20with-NumPy-013243.svg)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E.svg)](https://scikit-learn.org/)

[Quick Start](#-quick-start) •
[What It Does](#-what-it-does) •
[Results](#-results) •
[Architecture](#-architecture) •
[Extending](#-extending)

</div>

---

## What It Does

When a request hits a distributed system, *something* has to pick which server handles it. That something is the **load balancer**, and the rule it follows is the **routing policy**. The wrong policy creates hot spots, stragglers, and tail-latency disasters. The right one makes the cluster sing.

LoadSim lets you measure exactly how big that gap is, across **6 routing policies**, **4 workload patterns**, and **2 cluster topologies**, without renting a single VM.

It's a faithful discrete-event simulator that models:

- **Servers** with FIFO queues, configurable service rates, and stochastic processing times
- **Network** delays with jitter
- **Workload generators** that mirror real-world traffic shapes (Poisson, Zipf, bursty)
- **Hot keys** — the dirty secret behind most production tail-latency horror stories
- **A learned router** trained on traces from classical "teacher" policies

If you've ever read *"The Tail at Scale"* and wondered which routing trick actually pays off, this is your sandbox.

---

## Quick Start

```bash
git clone https://github.com/darsh462/LoadSim.git
cd loadsim
pip install -r requirements.txt

# 1. Sanity check (~2 seconds)
python3 tests/smoke.py

# 2. Quick experiment sweep (~30 seconds)
python3 -m experiments.run_experiments --quick

# 3. Generate plots
python3 -m experiments.plot_results

# 4. Open the results
open plots/tail_comparison.png       # macOS
xdg-open plots/tail_comparison.png   # Linux
start plots\tail_comparison.png      # Windows
```

That's it. You now have a full latency / throughput / fairness comparison across all 6 routers in your `plots/` folder.

> For a publication-grade run (3 seeds, 30s each, 144 simulations, ~5 min), drop the `--quick` flag.

---

## The Routing Policies

LoadSim implements six routers spanning the design spectrum from "blind" to "learned":

| Router | Strategy | Knows about node state? | Cost |
|---|---|:---:|:---:|
| **Random** | Pick uniformly at random | ❌ | O(1) |
| **Round-Robin** | Cycle through nodes | ❌ | O(1) |
| **Least-Queue (JSQ)** | Send to the shortest queue | ✅ | O(N) |
| **Power-of-2 Choices** | Sample 2 nodes, pick less loaded | ✅ | O(1) |
| **Latency-Aware** | Minimize `queue × recent_latency` | ✅ | O(N) |
| **Learned (ML)** | Predict latency per node, pick argmin | ✅ | O(N) + inference |

The learned router is trained offline on traces collected from three teacher policies (JSQ, Latency-Aware, Power-of-2) across all four workloads, using a `DecisionTreeRegressor` over a 6-dimensional per-node feature vector.

---

## The Workloads

| Workload | Inter-arrival | Key distribution | Stresses |
|---|---|---|---|
| **Uniform** | Poisson | Uniform | Steady-state baseline |
| **Zipf (mild)** | Poisson | Zipf, *s=0.8* | Mild popularity skew |
| **Zipf (heavy)** | Poisson | Zipf, *s=1.4* | Severe hot-spot, near-saturation |
| **Bursty** | Poisson with phase changes | Mildly skewed | Adaptability under traffic spikes |

> **Why hot keys matter.** Without modeling per-key cost, key skew is just bookkeeping; Zipf would only change which integer ID gets attached to each request, with zero load impact. LoadSim's `key_size_fn` hook scales service time for the top 1–2% of keys, mirroring real-world phenomena like cache misses on popular objects, large value payloads, or expensive computations on viral content. *That's* what turns Zipf into actual hotspots.

---

## Results

Run a full sweep and you'll get six plots that tell a complete story:

| Plot | What it shows |
|---|---|
| `tail_comparison.png` | **p99 latency across all routers** — the headline plot |
| `latency_bars_homogeneous.png` | Mean / p95 / p99 on identical servers |
| `latency_bars_heterogeneous.png` | Same, on mixed fast/slow servers |
| `throughput.png` | Sustained requests/sec per router |
| `fairness.png` | Jain's fairness index — load distribution across nodes |
| `utilization_heatmap.png` | Per-node busy fraction on heterogeneous clusters |

### What the data reveals

 **On a homogeneous cluster with uniform traffic:**
Every state-aware router (JSQ, Power-of-2, Latency-Aware) clusters tightly. Even Round-Robin is competitive on mean latency. *Routing barely matters here.*

 **On a heterogeneous cluster (mixed fast/slow nodes):**
Round-Robin and Random collapse, p99 latencies of **10–30 seconds**. State-aware routers stay under a second. Static routers don't know which nodes are fast, so they jam slow nodes while fast ones idle.

 **Under heavy Zipf (1% of keys = bulk of work):**
Even good routers struggle as the system approaches saturation. The **learned router fails outside its training distribution**, a finding worth keeping rather than hiding, because it captures a real lesson: ML routers need training data that covers stressed regimes, not just nominal ones.

 **The headline takeaway:**
> Static routing is fine when reality cooperates. The moment hardware is uneven, traffic is skewed, or load spikes, and *all three happen in production*, state-aware routing wins by an order of magnitude on tail latency.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    experiments/run_experiments.py            │
│            (sweeps 6 routers × 4 workloads × 2 clusters)     │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                      src/simulator.py                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  Workload   │─▶│   Router     │─▶│      Nodes           │ │
│  │ generator   │  │              │  │  (queues + servers)  │ │
│  └─────────────┘  └──────────────┘  └──────────────────────┘ │
│         │                │                     │             │
│         └────────────────┴─────────────────────┘             │
│                          ▼                                   │
│              ┌──────────────────────┐                        │
│              │   Event Engine       │                        │
│              │  (priority queue)    │                        │
│              └──────────────────────┘                        │
│                          ▼                                   │
│              ┌──────────────────────┐                        │
│              │ Metrics Collector    │                        │
│              └──────────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  results/*.csv  →  experiments/plot_results.py
                                            │
                                            ▼
                                       plots/*.png
```

### Module breakdown

```
loadsim/
├── src/
│   ├── event_engine.py      # Priority-queue event loop (the clock)
│   ├── node.py              # Node, Network, Request models
│   ├── routers.py           # All 6 routing policies
│   ├── workload.py          # Uniform / Zipf / Bursty generators
│   ├── simulator.py         # Wires engine + nodes + router + workload
│   ├── learned_model.py     # Trains the regressor for the learned router
│   └── metrics.py           # MetricsCollector + RequestRecord
├── experiments/
│   ├── run_experiments.py   # Full sweep driver
│   └── plot_results.py      # Generates all PNG plots
├── tests/
│   └── smoke.py             # Fast sanity check (~2s)
├── results/                 # CSV outputs (created by sweep)
├── plots/                   # PNG outputs (created by plot script)
└── README.md
```

---

## Use the Simulator Directly

```python
from src.simulator import Simulator, SimConfig
from src.node import Node, Network
from src.routers import LeastQueueRouter
from src.workload import ZipfianWorkload

# Build a 4-node cluster
nodes   = [Node(i, service_rate=100.0) for i in range(4)]
network = Network(mean_delay=0.001, jitter=0.0005)
router  = LeastQueueRouter()
wl      = ZipfianWorkload(arrival_rate=300.0, n_keys=500, skew=1.2)

# Simulate 30 seconds of traffic
sim = Simulator(nodes, network, router, wl,
                SimConfig(duration=30.0, seed=42))
sim.run()

print(sim.metrics.summary())
# {'n_requests': 8957, 'mean_latency': 0.024, 'p99_latency': 0.087,
#  'throughput': 298.6, 'fairness': 0.998, ...}
```

### Train your own learned router

```python
from src.learned_model import build_training_set, train_model
from src.routers import LearnedRouter

# Run a "teacher" policy first to collect traces
# (records is sim.metrics.records from a teacher run)
X, y = build_training_set(records, n_nodes=4)
model = train_model(X, y, kind="tree")   # "tree", "ridge", or "mlp"

router = LearnedRouter(model=model, warmup=200)
```

---

## Metrics

Every run reports:

- **Latency** — mean, median, p90, p95, p99, max
- **Throughput** — sustained req/sec
- **Drop rate** — when queues are bounded
- **Jain's fairness index** — load distribution (1.0 = perfectly balanced)
- **Per-node utilization** — fraction of time each node was busy

> **Why Jain's index?** Because raw load counts hide the structure. A perfectly fair distribution on uneven hardware is *worse* than an unfair one — what you actually want is fairness *proportional to capacity*. Jain's index combined with the utilization heatmap reveals that distinction.

---

## Extending

| Want to add... | Do this |
|---|---|
| A new router | Subclass `Router` in `src/routers.py`, register in `make_router()` |
| A new workload | Subclass `WorkloadGenerator` in `src/workload.py`, register in `make_workload()` |
| A new metric | Extend `MetricsCollector.summary()` in `src/metrics.py` |
| A new ML model | Add to `train_model()` in `src/learned_model.py` |
| A new plot | Add a function in `experiments/plot_results.py` and call from `make_all_plots()` |

Each module is under 250 lines. The whole simulator is under 1500.

---

## CLI Reference

**`run_experiments.py`**

```
--out          DIR      Output directory for CSVs            [default: results/]
--arrival-rate FLOAT    Aggregate arrival rate (req/s)       [default: 320.0]
--duration     FLOAT    Simulated seconds per run            [default: 30.0]
--seeds        INT...   Seeds to average over                [default: 1 2 3]
--quick                 Shorthand for duration=10, seeds=(1,)
```

**`plot_results.py`**

```
--results DIR    Where to read CSVs from    [default: results/]
--plots   DIR    Where to write PNGs        [default: plots/]
```

---

## Background

This project draws from three threads of distributed systems research:

- **Tail-latency analysis** — *"The Tail at Scale"* (Dean & Barroso, CACM 2013)
- **Power-of-choices load balancing** — Mitzenmacher's classic queueing results
- **Learned systems** — applying lightweight ML models to traditional systems decisions (think learned indexes, learned cache policies, but for routing)

The simulator approach trades off realism for control: you can't catch hardware-specific quirks, but you *can* run thousands of clean comparisons in minutes, with full reproducibility from a seed.

---

## Requirements

```
Python 3.9+
numpy
pandas
matplotlib
scikit-learn   (optional — pure-numpy fallback included)
```

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## FAQ

<details>
<summary><b>Why simulation instead of running on real hardware?</b></summary>

Real clusters are noisy, expensive, and slow to iterate on. A simulator lets you change one variable at a time, run hundreds of comparisons in minutes, and reproduce results bit-for-bit from a seed. The trade-off is that you miss low-level effects (NUMA, GC pauses, kernel scheduling), but for studying *routing policies*, which are about queue dynamics and information flow, simulation is the right tool.
</details>

<details>
<summary><b>Why does the learned router lose on Zipf-heavy?</b></summary>

The training data was collected under teacher policies that kept the system out of overload. When the evaluation pushes the system *into* overload, the model has to extrapolate beyond what it saw, and its argmin becomes unreliable. This isn't a bug, it's a real lesson about ML in systems: **training distribution coverage is everything**. Fixing it would mean either (a) intentionally including overload regimes in training, or (b) falling back to a classical policy when the model's prediction confidence is low.
</details>

<details>
<summary><b>How realistic is the network model?</b></summary>

Modest. We model one-way delay with optional Gaussian jitter. We don't model packet loss, TCP slow start, head-of-line blocking, or congestion. For the kind of routing-policy questions this project asks, that's fine, but if you wanted to study, say, the impact of HOL blocking on load distribution, you'd need to extend `Network`.
</details>

<details>
<summary><b>Can I plug in real-world traces?</b></summary>

Yes. Subclass `WorkloadGenerator` and `yield (inter_arrival, key)` from your trace file. As long as you produce the same `(float, int)` tuples, the simulator doesn't care whether they came from Zipf or from a real production log.
</details>

---
