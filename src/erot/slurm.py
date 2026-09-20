"""Select one independent experiment from a Slurm array manifest."""

import argparse
import json
import os
from pathlib import Path

from .launcher import launch_experiments


def array_configuration(manifest, index):
    """Resolve manifest-relative configuration paths and a zero-based task ID."""
    manifest = Path(manifest)
    paths = json.loads(manifest.read_text())
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or not path for path in paths)
    ):
        raise ValueError("manifest must contain a nonempty list of configuration paths")
    try:
        selected = int(index)
    except (TypeError, ValueError) as exc:
        raise ValueError("array index must be an integer") from exc
    if not 0 <= selected < len(paths):
        raise ValueError("array index is outside the configuration manifest")
    return json.loads((manifest.parent / paths[selected]).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--index", default=os.environ.get("SLURM_ARRAY_TASK_ID"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = array_configuration(args.configs, args.index)
        profile = json.loads(args.profile.read_text())
        allowed = {"device", "cpu_budget", "threads_per_worker"}
        if not isinstance(profile, dict) or set(profile) - allowed:
            raise ValueError(
                "profile supports only device, cpu_budget, threads_per_worker"
            )
        report = launch_experiments(
            [config], args.output_root, workers=1, resume=args.resume, **profile
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
