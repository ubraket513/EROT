# Checkpoint format v1

A checkpoint directory contains immutable checkpoint-<uuid> generations and an
atomically replaced LATEST pointer. Each generation contains arrays.npz and
manifest.json. Array data is written without pickle, flushed and fsynced;
the manifest includes a SHA-256 payload checksum, every array shape/dtype,
a structural tree, identity metadata and schema version. The generation is
renamed into place before LATEST is published. An interrupted save can leave
unpublished files but does not modify the preceding complete checkpoint.

Trees support numeric/boolean arrays, JSON scalars, lists, tuples, string-keyed
dictionaries and explicitly registered EROT state/diagnostic named tuples.
Loading never dynamically imports a type named by the file. It rejects unknown
schemas/types, incompatible expected metadata and array corruption. SHA-256
checksums detect accidental changes; they are not signed authenticity proofs.

Required metadata fields are config_digest, input_digest, dtype and topology.
The experiment worker supplies runtime versions, source identity, seed and RNG
state as additional fields. Every expected field is compared exactly.
load_checkpoint returns NumPy-backed arrays without silently downcasting or
selecting a device; callers configure precision and place leaves explicitly.

Callers must own the run directory and serialize writers. This format relies
on POSIX same-filesystem rename, file fsync and directory fsync. Filesystems
without those semantics are not certified. Unpublished generations are ignored;
automatic garbage collection is intentionally absent. State arrays are staged
on host during saving, so checkpoint memory and transfer cost must be included
in capacity planning. A snapshot is not a complete solver checkpoint.
