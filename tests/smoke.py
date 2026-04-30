"""Tiny smoke test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import Simulator, SimConfig
from src.node import Node, Network
from src.routers import RoundRobinRouter, LeastQueueRouter, LatencyAwareRouter
from src.workload import UniformWorkload, ZipfianWorkload

def test_basic():
    nodes = [Node(i, service_rate=100.0) for i in range(4)]
    net = Network(mean_delay=0.001)
    router = LeastQueueRouter()
    wl = UniformWorkload(arrival_rate=300.0, n_keys=100)
    sim = Simulator(nodes, net, router, wl, SimConfig(duration=20.0, seed=1))
    sim.run()
    s = sim.metrics.summary()
    print("LeastQueue summary:", s)
    assert s["n_requests"] > 1000, f"Too few requests: {s['n_requests']}"
    assert s["mean_latency"] > 0
    assert 0.0 < s["fairness"] <= 1.0

def test_zipf():
    nodes = [Node(i, service_rate=100.0) for i in range(4)]
    net = Network(mean_delay=0.001)
    router = RoundRobinRouter()
    wl = ZipfianWorkload(arrival_rate=300.0, n_keys=100, skew=1.2)
    sim = Simulator(nodes, net, router, wl, SimConfig(duration=10.0, seed=2))
    sim.run()
    print("Zipf+RR summary:", sim.metrics.summary())

if __name__ == "__main__":
    test_basic()
    test_zipf()
    print("OK")
