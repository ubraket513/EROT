# SPDX-License-Identifier: Apache-2.0
"""Discrete optimal-transport gradient flows."""

from .functionals import Entropy, Potential, Quadratic
from .jko import solve_entropic_jko
from .pdhg import solve_pdhg_jko
from .trajectory import FlowState, initialize_flow, jko_step, run_flow_chunk

__all__ = [
    "Entropy",
    "Potential",
    "Quadratic",
    "FlowState",
    "initialize_flow",
    "jko_step",
    "run_flow_chunk",
    "solve_entropic_jko",
    "solve_pdhg_jko",
]
