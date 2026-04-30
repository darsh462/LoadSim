"""
Node and network models.

A Node models a single back-end server in the distributed store. It has:
  * a configurable service rate (mean requests processed per second),
  * an internal FIFO queue,
  * a processing-latency distribution (exponential by default, but any callable
    that returns a positive float is accepted).

The Network model adds a configurable delay on every hop (client -> node and
node -> client). Delay can be fixed or sampled from a distribution.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional

import numpy as np


@dataclass
class Request:
    """A single client request flowing through the system."""
    req_id: int
    key: int                       # the key being accessed (matters for hotspots)
    arrival_time: float            # time the LB saw the request
    dispatch_time: float = 0.0     # time dispatched to a node
    service_start_time: float = 0.0
    completion_time: float = 0.0
    response_time: float = 0.0     # time back at the client
    assigned_node: int = -1
    size: float = 1.0              # work units (used for heterogeneous requests)


class Node:
    """A single server with a FIFO queue.

    The service time for each request is drawn from `service_dist()`. If the
    node is idle when a request arrives, it begins processing immediately;
    otherwise the request is enqueued.
    """

    def __init__(self,
                 node_id: int,
                 service_rate: float,
                 service_dist: Optional[Callable[[], float]] = None,
                 capacity: Optional[int] = None) -> None:
        if service_rate <= 0:
            raise ValueError("service_rate must be positive")
        self.node_id = node_id
        self.service_rate = service_rate         # mean service rate (req/s)
        self.service_dist = service_dist         # callable -> service time in seconds
        self.capacity = capacity                 # None = unbounded queue
        self.queue: Deque[Request] = deque()
        self.busy: bool = False
        self.current: Optional[Request] = None

        # Running metrics
        self.total_served: int = 0
        self.total_busy_time: float = 0.0
        self.total_wait_time: float = 0.0
        self.dropped: int = 0

        # Exponential-moving-average latency used by the latency-aware router
        self.ema_latency: float = 1.0 / service_rate
        self.ema_alpha: float = 0.1

        # Short history of the last `hist_len` completed-request latencies.
        # The ML router consumes summary statistics over this window.
        self.hist_len: int = 32
        self.latency_history: Deque[float] = deque(maxlen=self.hist_len)

    def sample_service_time(self, rng: np.random.Generator,
                            size: float = 1.0) -> float:
        """Sample a service time. Defaults to exponential(1/rate), scaled
        multiplicatively by the request's `size`. A `size` > 1 represents a
        heavier request (e.g. a hot key with a larger value payload, or an
        expensive scan) and proportionally increases service time."""
        if self.service_dist is not None:
            return self.service_dist() * size
        return float(rng.exponential(1.0 / self.service_rate)) * size

    def queue_length(self) -> int:
        """Number of requests waiting (does not include the one being served)."""
        return len(self.queue)

    def load(self) -> int:
        """Total work units at this node: queue + currently serving."""
        return len(self.queue) + (1 if self.busy else 0)

    def update_ema_latency(self, latency: float) -> None:
        """Update the exponential moving average of recent request latency."""
        self.ema_latency = (
            self.ema_alpha * latency + (1.0 - self.ema_alpha) * self.ema_latency
        )
        self.latency_history.append(latency)

    def utilization(self, sim_time: float) -> float:
        """Fraction of simulation time this node was busy."""
        if sim_time <= 0:
            return 0.0
        return min(1.0, self.total_busy_time / sim_time)

    def feature_vector(self) -> np.ndarray:
        """Return a compact feature vector describing this node's state.

        These features are consumed by the learned router and were deliberately
        chosen to be cheap to compute in a real system:
          0. queue length
          1. 1.0 if currently busy else 0.0
          2. EMA of recent latency
          3. mean of last-`hist_len` latencies (0 if empty)
          4. std of last-`hist_len` latencies (0 if empty)
          5. nominal service rate (constant per node, lets the model know fast
             vs. slow machines)
        """
        if self.latency_history:
            arr = np.asarray(self.latency_history)
            mean_lat = float(arr.mean())
            std_lat = float(arr.std())
        else:
            mean_lat = 0.0
            std_lat = 0.0
        return np.array([
            float(self.queue_length()),
            1.0 if self.busy else 0.0,
            self.ema_latency,
            mean_lat,
            std_lat,
            self.service_rate,
        ], dtype=np.float64)


class Network:
    """Models per-hop network delay.

    `mean_delay` is the average one-way delay in seconds. If `jitter` > 0,
    each sample is drawn from a normal distribution truncated at 0.
    """

    def __init__(self, mean_delay: float = 0.001, jitter: float = 0.0) -> None:
        self.mean_delay = mean_delay
        self.jitter = jitter

    def sample(self, rng: np.random.Generator) -> float:
        if self.jitter <= 0:
            return self.mean_delay
        # Truncated normal: resample if we get a negative delay
        for _ in range(8):
            v = rng.normal(self.mean_delay, self.jitter)
            if v >= 0:
                return float(v)
        return max(0.0, self.mean_delay)
