# TP3 Triton source rebuild

[Complete instructions, exact commands and qualification boundaries](../../docs/AOT-SOURCE-REBUILD.md).

- `tp3_triton_kernels.py`: exact missing TP3 generator named in issue #1.
- `compile_triton.py`: unchanged historical explicit-case compiler.
- `seed_triton_cases.py`: original fixture helper; not full launch coverage.
- `base-cases.json` / `v128-cases.json`: explicit complete launch inventories.
- `rebuild.py`: checked offline compiler and PTX/cubin comparison.
- `debug_paths.py`: static ELF/DWARF reader for reproducible PTX filename labels.
- `SOURCES.json`: byte hashes and target/compiler pin.

No model weights, credentials, operator configuration or service changes are
included. The reference generators are upstream TensorFold work, credited in
the guide and retained licenses; the TP3 compatibility addition is identified
in its source header. Builds never install into a running runtime.
