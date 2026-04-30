"""
Discrete-event simulation engine.

The EventEngine provides a time-ordered priority queue of events. Events are
processed in non-decreasing timestamp order. Ties are broken by a monotonically
increasing sequence number so that the engine is deterministic given a fixed
random seed.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class EventType(Enum):
    """Types of events the simulator can process."""
    ARRIVAL = "arrival"          # A new request arrives at the load balancer
    DISPATCH = "dispatch"        # A request is dispatched to a node (after network delay)
    SERVICE_START = "start"      # A node begins serving a request
    COMPLETION = "completion"    # A node finishes a request
    RESPONSE = "response"        # Response returns to the client (after network delay)
    METRIC_SAMPLE = "sample"     # Periodic metric sample


@dataclass(order=True)
class Event:
    """A scheduled event in the simulation.

    Events are ordered first by timestamp, then by a sequence number, so the
    priority queue is fully deterministic.
    """
    time: float
    seq: int = field(compare=True)
    type: EventType = field(compare=False)
    payload: dict = field(default_factory=dict, compare=False)


class EventEngine:
    """Minimal discrete-event simulation loop.

    Handlers are functions that receive (engine, event) and may schedule further
    events via engine.schedule(). The engine keeps a global clock `now` and
    processes events until the queue is empty or `stop_time` is reached.
    """

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter = itertools.count()
        self._handlers: dict[EventType, Callable[["EventEngine", Event], None]] = {}
        self.now: float = 0.0
        self._stop_time: Optional[float] = None
        self._processed: int = 0

    # ------------------------------------------------------------------ API

    def register(self, event_type: EventType,
                 handler: Callable[["EventEngine", Event], None]) -> None:
        """Register a handler for a given event type."""
        self._handlers[event_type] = handler

    def schedule(self, delay: float, event_type: EventType,
                 payload: Optional[dict] = None) -> None:
        """Schedule an event `delay` seconds after the current time."""
        if delay < 0:
            raise ValueError(f"Negative delay: {delay}")
        ev = Event(
            time=self.now + delay,
            seq=next(self._counter),
            type=event_type,
            payload=payload or {},
        )
        heapq.heappush(self._heap, ev)

    def schedule_at(self, time: float, event_type: EventType,
                    payload: Optional[dict] = None) -> None:
        """Schedule an event at absolute simulation time `time`."""
        if time < self.now:
            raise ValueError(f"Cannot schedule event in the past: {time} < {self.now}")
        ev = Event(
            time=time,
            seq=next(self._counter),
            type=event_type,
            payload=payload or {},
        )
        heapq.heappush(self._heap, ev)

    def run(self, stop_time: Optional[float] = None,
            max_events: Optional[int] = None) -> None:
        """Run the simulation until the queue empties or a stopping
        condition is met."""
        self._stop_time = stop_time
        while self._heap:
            if self._stop_time is not None and self._heap[0].time > self._stop_time:
                break
            if max_events is not None and self._processed >= max_events:
                break
            ev = heapq.heappop(self._heap)
            self.now = ev.time
            handler = self._handlers.get(ev.type)
            if handler is None:
                raise RuntimeError(f"No handler registered for event {ev.type}")
            handler(self, ev)
            self._processed += 1

    @property
    def processed(self) -> int:
        """Number of events processed so far."""
        return self._processed

    @property
    def pending(self) -> int:
        """Number of events still waiting in the queue."""
        return len(self._heap)
