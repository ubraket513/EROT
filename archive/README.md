# Historical EROT artifacts

`legacy-erot/` preserves the original demonstration tensors, notebook, and
generated test plots/results from revision
`424c4ede55f67aa805c4f7914133a5a7bc1b9155`.
They are historical research material, not inputs to the current test suite.

[manifest.json](manifest.json) records the original and archived paths, sizes,
SHA-256 hashes, and source revision. The cleanup moves preserve file bytes.
The original Git revision also retains every original path.

The notebook and demonstration README describe historical interfaces and
dependencies. Use the current [EROT README](../README.md) for runnable examples
and the [migration guide](../docs/migration-0.1.md) for old PyTorch data.
Archive files are excluded from wheel and source distributions.
