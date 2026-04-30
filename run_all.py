"""
One-shot driver: runs the full pipeline and shows the plots.

Usage
-----
    python3 run_all.py                # full sweep (~5 min) + open plots
    python3 run_all.py --quick        # quick sweep (~30 sec) + open plots
    python3 run_all.py --no-show      # run everything but don't open viewers
    python3 run_all.py --skip-sim     # only re-plot existing CSVs

This script:
  1. Sanity-checks the simulator with a tiny smoke test.
  2. Runs the experiment sweep (saves results/*.csv).
  3. Generates all plots (saves plots/*.png).
  4. Opens every plot in your OS's default image viewer.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# --------------------------------------------------------------- helpers

def _banner(msg: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {msg}\n{bar}")


def _open_file(path: str) -> bool:
    """Open a file in the OS's default viewer. Returns True on success."""
    try:
        system = platform.system()
        if system == "Darwin":          # macOS
            subprocess.Popen(["open", path],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        elif system == "Windows":
            os.startfile(path)          # type: ignore[attr-defined]
        else:                           # Linux / other
            subprocess.Popen(["xdg-open", path],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"  could not open {path}: {e}")
        return False


# --------------------------------------------------------------- pipeline steps

def step_smoke() -> None:
    _banner("STEP 1 / 4 — smoke test")
    from tests import smoke  # type: ignore
    smoke.test_basic()
    smoke.test_zipf()
    print("Smoke test passed.")


def step_experiments(quick: bool) -> None:
    _banner("STEP 2 / 4 — running simulation sweep "
            f"({'quick: ~30 sec' if quick else 'full: ~5 min'})")
    from experiments.run_experiments import run_all
    if quick:
        run_all(out_dir=os.path.join(ROOT, "results"),
                duration=10.0, seeds=(1,))
    else:
        run_all(out_dir=os.path.join(ROOT, "results"),
                duration=30.0, seeds=(1, 2, 3))


def step_plots() -> list[str]:
    _banner("STEP 3 / 4 — generating plots")
    from experiments.plot_results import make_all_plots
    plots_dir = os.path.join(ROOT, "plots")
    results_dir = os.path.join(ROOT, "results")
    make_all_plots(results_dir=results_dir, plots_dir=plots_dir)
    pngs = sorted(
        os.path.join(plots_dir, f)
        for f in os.listdir(plots_dir) if f.endswith(".png")
    )
    return pngs


def step_summary() -> None:
    """Print the aggregated results table to the terminal."""
    import pandas as pd
    csv_path = os.path.join(ROOT, "results", "aggregated.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    cols = ["cluster", "workload", "router",
            "mean_latency", "p99_latency", "throughput", "fairness"]
    df = df[[c for c in cols if c in df.columns]].copy()
    for c in ("mean_latency", "p99_latency"):
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{v:.4f}")
    if "throughput" in df.columns:
        df["throughput"] = df["throughput"].map(lambda v: f"{v:.1f}")
    if "fairness" in df.columns:
        df["fairness"] = df["fairness"].map(lambda v: f"{v:.3f}")
    print("\nAggregated results (mean latency / p99 / throughput / fairness):")
    print(df.to_string(index=False))


def step_open(pngs: list[str], show: bool) -> None:
    _banner("STEP 4 / 4 — opening plots")
    if not pngs:
        print("No PNGs found in plots/.")
        return
    for p in pngs:
        print(f"  {os.path.basename(p)}")
    if not show:
        print("(--no-show given; not opening viewers)")
        return
    print()
    for p in pngs:
        if _open_file(p):
            time.sleep(0.4)   # give the OS time to launch each window


# --------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(
        description="Run the LoadSim pipeline end-to-end and display plots.")
    p.add_argument("--quick", action="store_true",
                   help="Run a small sweep (~30 sec) instead of the full one.")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open the plot files in an image viewer.")
    p.add_argument("--skip-sim", action="store_true",
                   help="Skip the simulation step and only regenerate plots "
                        "from existing CSVs.")
    args = p.parse_args()

    t0 = time.time()
    try:
        if not args.skip_sim:
            step_smoke()
            step_experiments(quick=args.quick)
        else:
            _banner("SKIP-SIM: re-plotting existing results only")
        pngs = step_plots()
        step_summary()
        step_open(pngs, show=not args.no_show)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    dt = time.time() - t0
    _banner(f"DONE in {dt:.1f} sec — plots are in {os.path.join(ROOT, 'plots')}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
