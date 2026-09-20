"""EROT: accelerator-oriented entropy-regularized optimal transport."""

from .api import solve
from .types import SolverConfig, SolveResult

__all__ = ["SolveResult", "SolverConfig", "solve"]
__version__ = "0.1.0"
