# Migrating to EROT 0.1

Version 0.1 is a clean API break. Import `erot`, call the unified `solve`
function, and use NumPy `.npy` inputs and `.npz` results. The legacy `eot`
scripts, their implicit plotting, and mixed JAX/PyTorch result dictionaries are
not part of the new package.

## Convert trusted PyTorch tensors

Only load `.pt` files that you created or trust; PyTorch serialization may
execute code while loading older pickle-based files.

```python
from pathlib import Path

import numpy as np
import torch

source = Path("legacy.pt")
value = torch.load(source, map_location="cpu", weights_only=True)
if isinstance(value, torch.Tensor):
    np.save(source.with_suffix(".npy"), value.detach().numpy())
else:
    raise TypeError("Convert dictionary entries individually after inspection")
```

The conversion dependency is deliberately not included in EROT; PyTorch is no
longer needed to install or run the library.

## Command mapping

```text
Old: python eot/run_classical_eot.py ...
New: erot solve classical ...

Old: python eot/run_quantum_eot.py ...
New: erot solve quantum ...

Old: separate output directory with plots and tensor dictionaries
New: one result.npz; run `erot plot result.npz --output coupling.png` explicitly
```

The unsupported gradient-descent, fixed-point, and Nesterov implementations
formerly under `erot.experimental` were removed during repository cleanup.
Use the supported `erot.solve` methods; historical implementations remain in
Git history before the cleanup commit.
