"""
Metrics collection.

MetricsCollector stores a trace of per-request outcomes and computes summary
statistics on demand. Using numpy arrays keeps this cheap even for millions
of requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RequestRecord:
    """A single completed-request observation used for training + evaluation."""
    req_id: int
    key: int
    arrival_time: float
    completion_time: float
    latency: float
    queue_wait: float
    service_time: float
    node_id: int
    # The node feature snapshots at dispatch time - used as X for the ML model.
    snapshot: np.ndarray


class MetricsCollector:
    """Aggregates per-request records and computes evaluation metrics."""

    def __init__(self) -> None:
        self.records: List[RequestRecord] = []
        self.node_busy: Dict[int, float] = {}
        self.dropped: int = 0
        self.total_arrivals: int = 0
        self.sim_duration: float = 0.0

    # ------------------------------------------------------------ ingest

    def add_record(self, rec: RequestRecord) -> None:
        self.records.append(rec)

    def add_drop(self) -> None:
        self.dropped += 1

    def set_sim_duration(self, t: float) -> None:
        self.sim_duration = t

    def set_node_busy(self, node_id: int, busy_time: float) -> None:
        self.node_busy[node_id] = busy_time

    # ----------------------------------------------------------- summary

    def latencies(self) -> np.ndarray:
        if not self.records:
            return np.array([])
        return np.array([r.latency for r in self.records])

    def summary(self) -> Dict[str, float]:
        """Return a dictionary of summary statistics suitable for a results
        table."""
        lats = self.latencies()
        if lats.size == 0:
            return {
                "n_requests": 0,
                "mean_latency": 0.0,
                "median_latency": 0.0,
                "p90_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "max_latency": 0.0,
                "throughput": 0.0,
                "drop_rate": 0.0,
                "fairness": 0.0,
            }
        throughput = lats.size / self.sim_duration if self.sim_duration > 0 else 0.0
        total_requests = self.total_arrivals if self.total_arrivals > 0 else lats.size + self.dropped
        drop_rate = self.dropped / total_requests if total_requests > 0 else 0.0
        per_node = self._requests_per_node()
        return {
            "n_requests": int(lats.size),
            "mean_latency": float(lats.mean()),
            "median_latency": float(np.median(lats)),
            "p90_latency": float(np.percentile(lats, 90)),
            "p95_latency": float(np.percentile(lats, 95)),
            "p99_latency": float(np.percentile(lats, 99)),
            "max_latency": float(lats.max()),
            "throughput": float(throughput),
            "drop_rate": float(drop_rate),
            "fairness": self._jain_fairness(per_node),
        }

    def _requests_per_node(self) -> np.ndarray:
        if not self.records:
            return np.array([])
        nodes = {}
        for r in self.records:
            nodes[r.node_id] = nodes.get(r.node_id, 0) + 1
        return np.array(list(nodes.values()), dtype=np.float64)

    @staticmethod
    def _jain_fairness(counts: np.ndarray) -> float:
        """Jain's fairness index: 1 means perfectly balanced load, 1/N means
        all traffic went to a single node."""
        if counts.size == 0:
            return 0.0
        num = float(counts.sum()) ** 2
        den = float(counts.size) * float(np.square(counts).sum())
        if den == 0:
            return 0.0
        return num / den

    # ----------------------------------------------------------- training set

    def to_training_arrays(self):
        """Return (X, y) where X is the stacked feature snapshot at dispatch
        time and y is the observed response latency."""
        if not self.records:
            return np.zeros((0, 1)), np.zeros(0)
        X = np.stack([r.snapshot for r in self.records], axis=0)
        y = np.array([r.latency for r in self.records])
        return X, y
