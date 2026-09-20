"""Measure fresh independent experiment throughput under a fixed resource budget.

Run configurations sequentially; each worker-count/repetition gets fresh outputs.
No CPU measurements establish GPU throughput.
"""

import argparse
import json
from pathlib import Path

from erot.launcher import launch_experiments
from erot.runtime.timing import summarize_timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--cpu-budget", required=True, type=int)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or len(set(args.workers)) != len(args.workers):
        parser.error("positive repeats and distinct worker counts required")
    args.output_root.mkdir(parents=True, exist_ok=False)
    configs = [json.loads(path.read_text()) for path in args.configs]
    reports = []
    for workers in args.workers:
        for repeat in range(args.repeats):
            root = args.output_root / f"workers-{workers}-repeat-{repeat}"
            report = launch_experiments(
                configs,
                root,
                workers=workers,
                cpu_budget=args.cpu_budget,
                device=args.device,
            )
            for run in report["runs"]:
                path = root / run["run_id"] / "result.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                run["chunk_timing"] = summarize_timing(
                    path.parent / "timing.jsonl", result["chunks"]
                )
                run["worker_setup_seconds"] = result["worker_setup_seconds"]
                run["worker_session_seconds"] = result["session_wall_seconds"]
                # Includes interpreter/import startup and exit, not pure startup.
                run["outside_worker_session_seconds"] = max(
                    0.0, run["process_wall_seconds"] - result["session_wall_seconds"]
                )
            report["measurement_complete"] = report["completed"] == len(
                configs
            ) and all(
                run.get("chunk_timing", {}).get("complete", False)
                for run in report["runs"]
            )
            reports.append(report)
            (args.output_root / "throughput.json").write_text(
                json.dumps(reports, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "workers": workers,
                        "repeat": repeat,
                        "complete": report["measurement_complete"],
                        "wall_seconds": report["wall_seconds"],
                        "runs_per_second": report["completed_runs_per_second"],
                    }
                ),
                flush=True,
            )
    return 0 if all(report["measurement_complete"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
