"""
Top-level simulator.

A `Simulator` owns:
  * an `EventEngine`
  * a list of `Node`s
  * a `Network` model
  * a `Router`
  * a `WorkloadGenerator`
  * a `MetricsCollector`

Wiring of events
----------------
    ARRIVAL   -> router chooses a node, schedules DISPATCH after net delay
    DISPATCH  -> request reaches the node, enqueued or immediately started
    SERVICE_START -> service time sampled, COMPLETION scheduled
    COMPLETION -> node updates running stats, pops next from queue,
                  RESPONSE scheduled after net delay
    RESPONSE  -> record the final latency in metrics

The router sees the request at ARRIVAL time with the full current state of all
nodes, so there's no information leakage from the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .event_engine import EventEngine, Event, EventType
from .metrics import MetricsCollector, RequestRecord
from .node import Network, Node, Request
from .routers import Router
from .workload import WorkloadGenerator


@dataclass
class SimConfig:
    """All knobs needed to run a simulation."""
    duration: float = 60.0           # simulated seconds
    seed: int = 42
    record_snapshots: bool = True    # retain per-request feature vectors
    queue_capacity: Optional[int] = None  # per-node max queue (None = unbounded)
    # Optional map key_id -> service-time multiplier. A value > 1.0 makes a key
    # "hot" (heavy payload, cache miss, bigger value blob). The default makes
    # all keys equal-cost so key distribution alone can't distort load.
    key_size_fn: Optional[Callable[[int], float]] = None


class Simulator:
    """Discrete-event simulator for a cluster + load balancer.

    Example
    -------
    >>> nodes = [Node(i, service_rate=100) for i in range(4)]
    >>> sim = Simulator(nodes, Network(mean_delay=0.001),
    ...                 RoundRobinRouter(), UniformWorkload(250),
    ...                 SimConfig(duration=30))
    >>> sim.run()
    >>> print(sim.metrics.summary())
    """

    def __init__(self,
                 nodes: List[Node],
                 network: Network,
                 router: Router,
                 workload: WorkloadGenerator,
                 config: SimConfig) -> None:
        self.nodes = nodes
        self.network = network
        self.router = router
        self.workload = workload
        self.config = config

        self.rng = np.random.default_rng(config.seed)
        self.engine = EventEngine()
        self.metrics = MetricsCollector()
        self._req_id = 0
        self._inflight: dict = {}  # req_id -> Request

        self.engine.register(EventType.ARRIVAL, self._on_arrival)
        self.engine.register(EventType.DISPATCH, self._on_dispatch)
        self.engine.register(EventType.SERVICE_START, self._on_service_start)
        self.engine.register(EventType.COMPLETION, self._on_completion)
        self.engine.register(EventType.RESPONSE, self._on_response)

    # ----------------------------------------------------------- run

    def run(self) -> None:
        """Populate the event queue from the workload and drain it."""
        # Schedule all arrivals up front. This is a valid approach because
        # arrivals don't depend on system state (Poisson is exogenous).
        t = 0.0
        for dt, key in self.workload.stream(self.config.duration, self.rng):
            t += dt
            if t >= self.config.duration:
                break
            self._req_id += 1
            self.metrics.total_arrivals += 1
            size = 1.0
            if self.config.key_size_fn is not None:
                size = float(self.config.key_size_fn(key))
            req = Request(req_id=self._req_id, key=key,
                          arrival_time=t, size=size)
            self.engine.schedule_at(t, EventType.ARRIVAL,
                                    {"request": req})

        self.engine.run(stop_time=self.config.duration * 10)  # drain tail

        # Record final per-node busy time for utilisation computation.
        self.metrics.set_sim_duration(
            max(self.engine.now, self.config.duration))
        for n in self.nodes:
            self.metrics.set_node_busy(n.node_id, n.total_busy_time)

    # ----------------------------------------------------------- handlers

    def _on_arrival(self, eng: EventEngine, ev: Event) -> None:
        req: Request = ev.payload["request"]

        # Record a snapshot of the node features *before* we route. This is
        # what the ML router would see, and what we use as X for training.
        if self.config.record_snapshots:
            snapshot = np.concatenate(
                [n.feature_vector() for n in self.nodes], axis=0)
        else:
            snapshot = np.zeros(1)

        idx = self.router.route(req, self.nodes, eng.now)
        req.assigned_node = idx
        self._inflight[req.req_id] = (req, snapshot)

        delay = self.network.sample(self.rng)
        eng.schedule(delay, EventType.DISPATCH,
                     {"request": req, "node_idx": idx})

    def _on_dispatch(self, eng: EventEngine, ev: Event) -> None:
        req: Request = ev.payload["request"]
        idx: int = ev.payload["node_idx"]
        node = self.nodes[idx]
        req.dispatch_time = eng.now

        # If the node is capped and full, drop the request.
        if node.capacity is not None and node.load() >= node.capacity:
            node.dropped += 1
            self.metrics.add_drop()
            self._inflight.pop(req.req_id, None)
            return

        if not node.busy:
            node.busy = True
            node.current = req
            eng.schedule(0.0, EventType.SERVICE_START,
                         {"request": req, "node_idx": idx})
        else:
            node.queue.append(req)

    def _on_service_start(self, eng: EventEngine, ev: Event) -> None:
        req: Request = ev.payload["request"]
        idx: int = ev.payload["node_idx"]
        node = self.nodes[idx]
        req.service_start_time = eng.now
        service_time = node.sample_service_time(self.rng, size=req.size)
        eng.schedule(service_time, EventType.COMPLETION,
                     {"request": req, "node_idx": idx,
                      "service_time": service_time})

    def _on_completion(self, eng: EventEngine, ev: Event) -> None:
        req: Request = ev.payload["request"]
        idx: int = ev.payload["node_idx"]
        service_time: float = ev.payload["service_time"]
        node = self.nodes[idx]
        req.completion_time = eng.now

        # Accounting
        node.total_served += 1
        node.total_busy_time += service_time
        # End-to-end response latency tracked at RESPONSE, but update EMA here
        # so the router sees fresh info quickly.
        observed_latency = eng.now - req.arrival_time
        node.update_ema_latency(observed_latency)

        # Let the next queued request (if any) start serving.
        if node.queue:
            next_req = node.queue.popleft()
            node.current = next_req
            # SERVICE_START fires immediately - but still goes through the engine
            # so timestamps stay globally ordered.
            eng.schedule(0.0, EventType.SERVICE_START,
                         {"request": next_req, "node_idx": idx})
        else:
            node.busy = False
            node.current = None

        # Schedule the response back to the client.
        delay = self.network.sample(self.rng)
        eng.schedule(delay, EventType.RESPONSE,
                     {"request": req, "service_time": service_time,
                      "node_idx": idx})

    def _on_response(self, eng: EventEngine, ev: Event) -> None:
        req: Request = ev.payload["request"]
        service_time: float = ev.payload["service_time"]
        idx: int = ev.payload["node_idx"]
        req.response_time = eng.now
        latency = req.response_time - req.arrival_time
        queue_wait = req.service_start_time - req.dispatch_time

        snapshot = self._inflight.pop(req.req_id, (req, None))[1]
        self.metrics.add_record(RequestRecord(
            req_id=req.req_id,
            key=req.key,
            arrival_time=req.arrival_time,
            completion_time=req.completion_time,
            latency=latency,
            queue_wait=queue_wait,
            service_time=service_time,
            node_id=idx,
            snapshot=snapshot if snapshot is not None else np.zeros(1),
        ))
