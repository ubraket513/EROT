"""EROT: accelerator-oriented entropy-regularized optimal transport."""

from importlib.metadata import version as _distribution_version

from .api import solve
from .types import SolverConfig, SolveResult

__all__ = ["SolveResult", "SolverConfig", "solve"]
__version__ = _distribution_version("EROT")
