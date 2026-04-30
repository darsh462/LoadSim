"""
Synthetic workload generators.

A WorkloadGenerator yields request arrivals. Each yield produces a tuple
(inter_arrival_seconds, key) which the simulator uses to schedule ARRIVAL events.

Supported patterns
------------------
UniformWorkload     - exponential inter-arrivals + uniform key distribution.
                      This is the classic M/M/k driver.

ZipfianWorkload     - exponential inter-arrivals + Zipf key distribution. A
                      small number of keys dominate traffic, creating hotspots.

BurstyWorkload      - alternates between low-rate and high-rate phases to
                      simulate flash crowds or bursty application traffic.

MixedWorkload       - concatenates multiple workloads; useful for "normal ->
                      sudden spike -> recovery" style experiments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


Arrival = Tuple[float, int]   # (inter-arrival delay, key)


class WorkloadGenerator(ABC):
    """Abstract workload generator."""

    @abstractmethod
    def stream(self, duration: float,
               rng: np.random.Generator) -> Iterator[Arrival]:
        """Yield (inter_arrival, key) pairs until `duration` seconds of
        simulated time has been emitted."""
        raise NotImplementedError


class UniformWorkload(WorkloadGenerator):
    """Poisson arrivals (exponential inter-arrivals) with uniform keys."""

    def __init__(self, arrival_rate: float, n_keys: int = 1000) -> None:
        if arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        self.arrival_rate = arrival_rate
        self.n_keys = n_keys

    def stream(self, duration: float,
               rng: np.random.Generator) -> Iterator[Arrival]:
        t = 0.0
        while t < duration:
            dt = float(rng.exponential(1.0 / self.arrival_rate))
            t += dt
            if t >= duration:
                return
            key = int(rng.integers(0, self.n_keys))
            yield dt, key


class ZipfianWorkload(WorkloadGenerator):
    """Exponential inter-arrivals, Zipfian key popularity.

    `skew` = 0 is uniform; `skew` = 1 is the classic 80/20 distribution; larger
    values concentrate more traffic onto a handful of keys.
    """

    def __init__(self, arrival_rate: float, n_keys: int = 1000,
                 skew: float = 1.1) -> None:
        self.arrival_rate = arrival_rate
        self.n_keys = n_keys
        self.skew = skew
        # Precompute the normalised Zipf CDF once for O(log n) sampling.
        ranks = np.arange(1, n_keys + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, skew)
        weights /= weights.sum()
        self._cdf = np.cumsum(weights)

    def stream(self, duration: float,
               rng: np.random.Generator) -> Iterator[Arrival]:
        t = 0.0
        while t < duration:
            dt = float(rng.exponential(1.0 / self.arrival_rate))
            t += dt
            if t >= duration:
                return
            u = rng.random()
            key = int(np.searchsorted(self._cdf, u))
            if key >= self.n_keys:
                key = self.n_keys - 1
            yield dt, key


class BurstyWorkload(WorkloadGenerator):
    """Alternating low/high arrival rate phases.

    Phases alternate deterministically: `low_duration` seconds at `low_rate`,
    then `high_duration` seconds at `high_rate`, and so on.
    """

    def __init__(self,
                 low_rate: float,
                 high_rate: float,
                 low_duration: float,
                 high_duration: float,
                 n_keys: int = 1000,
                 skew: float = 0.0) -> None:
        self.low_rate = low_rate
        self.high_rate = high_rate
        self.low_duration = low_duration
        self.high_duration = high_duration
        self.n_keys = n_keys
        self.skew = skew

        if skew > 0:
            ranks = np.arange(1, n_keys + 1, dtype=np.float64)
            weights = 1.0 / np.power(ranks, skew)
            weights /= weights.sum()
            self._cdf = np.cumsum(weights)
        else:
            self._cdf = None

    def _sample_key(self, rng: np.random.Generator) -> int:
        if self._cdf is None:
            return int(rng.integers(0, self.n_keys))
        u = rng.random()
        key = int(np.searchsorted(self._cdf, u))
        return min(key, self.n_keys - 1)

    def stream(self, duration: float,
               rng: np.random.Generator) -> Iterator[Arrival]:
        t = 0.0
        in_burst = False
        phase_start = 0.0
        while t < duration:
            phase_len = self.high_duration if in_burst else self.low_duration
            rate = self.high_rate if in_burst else self.low_rate
            dt = float(rng.exponential(1.0 / rate))
            t += dt
            if t >= duration:
                return
            if t - phase_start >= phase_len:
                in_burst = not in_burst
                phase_start = t
            yield dt, self._sample_key(rng)


class MixedWorkload(WorkloadGenerator):
    """Concatenate several workloads into a single stream.

    Each entry in `phases` is a (duration, workload) pair. The phases run
    sequentially so you can compose scenarios such as:
        30s uniform -> 20s zipfian(1.4) -> 30s bursty
    """

    def __init__(self, phases: List[Tuple[float, WorkloadGenerator]]) -> None:
        self.phases = phases

    def stream(self, duration: float,
               rng: np.random.Generator) -> Iterator[Arrival]:
        for phase_dur, wl in self.phases:
            for dt, k in wl.stream(phase_dur, rng):
                yield dt, k


# --------------------------------------------------------------------- factory

def make_workload(name: str, arrival_rate: float,
                  n_keys: int = 1000, **kwargs) -> WorkloadGenerator:
    name = name.lower()
    if name == "uniform":
        return UniformWorkload(arrival_rate, n_keys=n_keys)
    if name == "zipfian":
        skew = float(kwargs.get("skew", 1.1))
        return ZipfianWorkload(arrival_rate, n_keys=n_keys, skew=skew)
    if name == "bursty":
        low = float(kwargs.get("low_rate", arrival_rate * 0.3))
        high = float(kwargs.get("high_rate", arrival_rate * 2.0))
        low_dur = float(kwargs.get("low_duration", 5.0))
        high_dur = float(kwargs.get("high_duration", 2.0))
        skew = float(kwargs.get("skew", 0.0))
        return BurstyWorkload(low, high, low_dur, high_dur,
                              n_keys=n_keys, skew=skew)
    raise ValueError(f"Unknown workload: {name}")
