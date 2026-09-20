"""Pure device solver APIs for compiled and batched numerical workflows."""

from .sinkhorn import materialize_plan, solve_sinkhorn
from .state import SinkhornState, SinkhornWarmStart, SolverDiagnostics

__all__ = [
    "SinkhornState",
    "SinkhornWarmStart",
    "SolverDiagnostics",
    "materialize_plan",
    "solve_sinkhorn",
]
