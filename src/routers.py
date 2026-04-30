"""
Routing policies.

Every router implements a single method, `route(request, nodes, now) -> int`,
which returns the index of the chosen node. Routers are pure Python objects
with no side effects other than updating their own internal state (e.g. a
round-robin counter).

Classical routers:
  * RandomRouter       - uniform random assignment (baseline)
  * RoundRobinRouter   - cyclic assignment
  * LeastQueueRouter   - shortest-queue-first
  * Power2ChoicesRouter- "the power of two choices" (pick 2, take less loaded)
  * LatencyAwareRouter - minimise expected latency based on EMA
  * JSQRouter          - join-the-shortest-queue (alias of LeastQueue, kept for clarity)

Learned router:
  * LearnedRouter      - predicts per-node response time from a feature vector
                         and greedily picks the node with the smallest predicted
                         latency. Supports a decision-tree regressor, ridge
                         regression, or a small MLP.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from .node import Node, Request


class Router(ABC):
    """Abstract base class for all routing policies."""

    name: str = "base"

    @abstractmethod
    def route(self, request: Request, nodes: List[Node], now: float) -> int:
        """Return the index of the node to dispatch the request to."""
        raise NotImplementedError

    def observe(self, request: Request, nodes: List[Node]) -> None:  # pragma: no cover
        """Optional hook for routers that learn online."""
        return None


# --------------------------------------------------------------------- classical

class RandomRouter(Router):
    """Uniform random assignment - the dumbest possible baseline."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def route(self, request: Request, nodes: List[Node], now: float) -> int:
        return int(self.rng.integers(0, len(nodes)))


class RoundRobinRouter(Router):
    """Cyclic assignment. Ignores node state entirely."""

    name = "round_robin"

    def __init__(self) -> None:
        self._i = 0

    def route(self, request: Request, nodes: List[Node], now: float) -> int:
        idx = self._i % len(nodes)
        self._i += 1
        return idx


class LeastQueueRouter(Router):
    """Pick the node with the shortest queue (ties broken by lowest id).

    This is "join the shortest queue" (JSQ) - a classical, surprisingly strong
    heuristic in the M/M/k literature.
    """

    name = "least_queue"

    def route(self, request: Request, nodes: List[Node], now: float) -> int:
        best_idx = 0
        best_load = nodes[0].load()
        for i in range(1, len(nodes)):
            load = nodes[i].load()
            if load < best_load:
                best_idx = i
                best_load = load
        return best_idx


# JSQRouter is just an alias; many papers call this algorithm JSQ.
JSQRouter = LeastQueueRouter


class Power2ChoicesRouter(Router):
    """Mitzenmacher's "power of two choices": sample 2 random nodes and pick the
    one with the shorter queue. Achieves almost-JSQ performance with O(1)
    inspection cost."""

    name = "power2"

    def __init__(self, seed: int = 0, d: int = 2) -> None:
        self.rng = np.random.default_rng(seed)
        self.d = d

    def route(self, request: Request, nodes: List[Node], now: float) -> int:
        n = len(nodes)
        d = min(self.d, n)
        candidates = self.rng.choice(n, size=d, replace=False)
        best = int(candidates[0])
        best_load = nodes[best].load()
        for c in candidates[1:]:
            c = int(c)
            if nodes[c].load() < best_load:
                best = c
                best_load = nodes[c].load()
        return best


class LatencyAwareRouter(Router):
    """Pick the node with the smallest expected completion time, where the
    expectation is:
        load * EMA_latency
    i.e. number of requests ahead of us multiplied by each request's recent
    average service time."""

    name = "latency_aware"

    def route(self, request: Request, nodes: List[Node], now: float) -> int:
        best_idx = 0
        best_score = nodes[0].load() * nodes[0].ema_latency
        for i in range(1, len(nodes)):
            score = nodes[i].load() * nodes[i].ema_latency
            if score < best_score:
                best_idx = i
                best_score = score
        return best_idx


# --------------------------------------------------------------------- learned

class LearnedRouter(Router):
    """Predictive router using a lightweight ML model.

    The router gathers feature vectors for every node and asks a regressor to
    predict the response latency if the request were sent there. It then picks
    the node with the lowest predicted latency (argmin). This mirrors the
    design used in several "learned systems" papers where a tiny model is
    trained offline on simulation traces and used online as a steering policy.

    To avoid catastrophic cold-start behaviour, the router falls back to
    LatencyAware for the first `warmup` requests (or until the model is
    loaded/trained).
    """

    name = "learned"

    def __init__(self,
                 model=None,
                 warmup: int = 200,
                 epsilon: float = 0.02,
                 seed: int = 0) -> None:
        self.model = model
        self.warmup = warmup
        self.epsilon = epsilon               # probability of random exploration
        self._count = 0
        self._fallback = LatencyAwareRouter()
        self._rng = np.random.default_rng(seed)

    def attach_model(self, model) -> None:
        self.model = model

    def route(self, request: Request, nodes: List[Node], now: float) -> int:
        self._count += 1
        if self.model is None or self._count < self.warmup:
            return self._fallback.route(request, nodes, now)

        # Occasional exploration keeps the training distribution diverse if the
        # router is retrained on its own traces.
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, len(nodes)))

        feats = np.stack([n.feature_vector() for n in nodes], axis=0)
        try:
            preds = self.model.predict(feats)
        except Exception:
            return self._fallback.route(request, nodes, now)
        return int(np.argmin(preds))


# --------------------------------------------------------------------- factory

def make_router(name: str, n_nodes: int, seed: int = 0,
                model=None) -> Router:
    """Factory function used by the experiment driver."""
    name = name.lower()
    if name == "random":
        return RandomRouter(seed=seed)
    if name == "round_robin":
        return RoundRobinRouter()
    if name in ("least_queue", "jsq"):
        return LeastQueueRouter()
    if name == "power2":
        return Power2ChoicesRouter(seed=seed)
    if name == "latency_aware":
        return LatencyAwareRouter()
    if name == "learned":
        return LearnedRouter(model=model, seed=seed)
    raise ValueError(f"Unknown router: {name}")
