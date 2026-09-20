"""Optional plotting support, kept outside the solver execution path."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_coupling(coupling: np.ndarray, output: str | Path) -> Path:
    if coupling.ndim != 2:
        raise ValueError("plotting currently supports only matrix-valued couplings")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires `pip install EROT[plot]`") from exc

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    display = np.abs(coupling) if np.iscomplexobj(coupling) else coupling
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(display, interpolation="nearest", cmap="inferno")
    axis.set_title(
        "EROT coupling" if not np.iscomplexobj(coupling) else "|EROT coupling|"
    )
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
