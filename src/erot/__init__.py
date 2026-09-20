"""EROT: accelerator-oriented entropy-regularized optimal transport."""

from importlib.metadata import version as _distribution_version

__all__ = ["SolveResult", "SolverConfig", "solve"]
__version__ = _distribution_version("EROT")


def __getattr__(name):
    # Host launchers must set affinity/device visibility before numerical imports.
    if name == "solve":
        from .api import solve

        globals()[name] = solve
        return solve
    if name in ("SolverConfig", "SolveResult"):
        from .types import SolverConfig, SolveResult

        globals().update(SolverConfig=SolverConfig, SolveResult=SolveResult)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
