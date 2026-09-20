"""Unstable research solvers excluded from the supported public API."""

from .classical import (
    quadratic_fixed_point,
    quadratic_gradient_descent,
    quadratic_nesterov,
)

__all__ = [
    "quadratic_fixed_point",
    "quadratic_gradient_descent",
    "quadratic_nesterov",
]
