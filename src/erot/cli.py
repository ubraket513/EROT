"""Command-line interface for EROT."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .api import solve
from .generate import cost_tensor, gaussian_marginal
from .io import load_array, load_result, save_result
from .plotting import plot_coupling
from .types import SolverConfig


def _add_solver_arguments(parser: argparse.ArgumentParser, problem: str) -> None:
    parser.add_argument("--cost", required=True, help="Cost tensor in .npy format")
    parser.add_argument(
        "--marginal",
        action="append",
        required=True,
        help="Marginal .npy path; repeat once per marginal",
    )
    if problem == "classical":
        parser.add_argument(
            "--regularizer", choices=("shannon", "quadratic"), required=True
        )
        parser.add_argument("--method", choices=("sinkhorn", "cyclic"), required=True)
    else:
        parser.set_defaults(regularizer="quadratic", method="cyclic")
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=50_000)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True, help="Result path ending in .npz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="erot", description="Entropy-regularized optimal transport on JAX"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    solve_parser = commands.add_parser("solve", help="Solve an EROT problem")
    problem_parsers = solve_parser.add_subparsers(dest="problem", required=True)
    _add_solver_arguments(problem_parsers.add_parser("classical"), "classical")
    _add_solver_arguments(problem_parsers.add_parser("quantum"), "quantum")

    generate_parser = commands.add_parser(
        "generate", help="Generate dense example data"
    )
    generate_commands = generate_parser.add_subparsers(
        dest="generate_kind", required=True
    )
    cost_parser = generate_commands.add_parser("cost")
    cost_parser.add_argument("--n", type=int, required=True)
    cost_parser.add_argument("--marginals", type=int, default=2)
    cost_parser.add_argument(
        "--kind",
        choices=("euclidean", "weak-coulomb", "strong-coulomb"),
        default="euclidean",
    )
    cost_parser.add_argument("--diagonal-penalty", type=float, default=1e6)
    cost_parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    cost_parser.add_argument("--output", required=True)

    marginal_parser = generate_commands.add_parser("marginal")
    marginal_parser.add_argument("--n", type=int, required=True)
    marginal_parser.add_argument("--loc", action="append", type=float, default=[])
    marginal_parser.add_argument("--scale", action="append", type=float, default=[])
    marginal_parser.add_argument("--lower", type=float, default=-5.0)
    marginal_parser.add_argument("--upper", type=float, default=5.0)
    marginal_parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    marginal_parser.add_argument("--output", required=True)

    plot_parser = commands.add_parser(
        "plot", help="Plot a matrix coupling from a result"
    )
    plot_parser.add_argument("result")
    plot_parser.add_argument("--output", required=True)
    return parser


def _save_array(path: str, value: np.ndarray) -> Path:
    destination = Path(path)
    if destination.suffix != ".npy":
        raise ValueError("generated arrays must use the .npy extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, value)
    return destination


def _run_solve(args: argparse.Namespace) -> dict[str, object]:
    config = SolverConfig(
        epsilon=args.epsilon,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        dtype=args.dtype,
        device=args.device,
    )
    result = solve(
        load_array(args.cost),
        [load_array(path) for path in args.marginal],
        problem=args.problem,
        regularizer=args.regularizer,
        method=args.method,
        config=config,
    )
    metadata = {
        "problem": args.problem,
        "regularizer": args.regularizer,
        "method": args.method,
        "epsilon": args.epsilon,
        "tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
        "dtype": args.dtype,
        "device": args.device,
    }
    destination = save_result(args.output, result, metadata=metadata)
    return {
        "output": str(destination),
        "error": result.error,
        "iterations": result.iterations,
        "converged": result.converged,
        "elapsed_seconds": result.elapsed_seconds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "solve":
            summary = _run_solve(args)
        elif args.command == "generate" and args.generate_kind == "cost":
            value = cost_tensor(
                args.n,
                args.marginals,
                kind=args.kind,
                dtype=args.dtype,
                diagonal_penalty=args.diagonal_penalty,
            )
            summary = {
                "output": str(_save_array(args.output, value)),
                "shape": value.shape,
            }
        elif args.command == "generate" and args.generate_kind == "marginal":
            locs = args.loc or [0.0]
            scales = args.scale or [1.0]
            value = gaussian_marginal(
                args.n,
                locs=locs,
                scales=scales,
                lower=args.lower,
                upper=args.upper,
                dtype=args.dtype,
            )
            summary = {
                "output": str(_save_array(args.output, value)),
                "shape": value.shape,
            }
        elif args.command == "plot":
            archive = load_result(args.result)
            summary = {"output": str(plot_coupling(archive["coupling"], args.output))}
        else:  # pragma: no cover - argparse prevents this branch
            parser.error("unsupported command")
            return 2
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(summary, sort_keys=True))
    return 0
