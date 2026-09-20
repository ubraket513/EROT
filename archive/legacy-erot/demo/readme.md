# EROT examples

The checked-in `.pt` files are legacy 0.0.1 demonstration data retained only
for migration testing. Convert trusted files to `.npy` using
[`docs/migration-0.1.md`](../docs/migration-0.1.md), or generate fresh portable
inputs with:

```bash
erot generate cost --n 100 --output C.npy
erot generate marginal --n 100 --loc 2.1 --output X.npy
erot generate marginal --n 100 --loc -3 --scale 0.75 --output Y.npy
```

Then run the example shown in the project README.
