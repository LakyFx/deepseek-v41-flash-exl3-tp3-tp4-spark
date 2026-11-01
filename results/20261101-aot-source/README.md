# Offline AOT source rebuild verification — 1 November 2026

Scope: the original TP3 kit and the **TP3 1.2.8 / AF attention additions**,
addressing [issue #1](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-tp4-spark/issues/1).
This is a compiler/source audit, **not an inference or speed benchmark**.
No model was loaded, no GPU kernel was executed, and no production API/service,
qualified release asset or historical tag was changed.

| Check | Result |
| --- | ---: |
| Source signatures covering the complete merged native manifest | **566 / 566** |
| Freshly compiled entries, ABI and scratch contract checked | **566 / 566** |
| Recorded historical PTX comparisons in the full set | **212 / 212 match** |
| Distinct recorded PTX hashes covered by those comparisons | **195** |
| Six short-prefill repair PTX hashes specifically requested in the issue | **6 / 6 match** |
| Added 1.2.8 attention entries with exact cubin identity | **24 / 24 match** |
| Original 542 entries with exact cubin identity in this environment | **0 / 542** |
| Targeted source, TP3 package and current release regression tests | **26 / 26 pass** |

The base manifest has 542 entries / 489 referenced cubin hashes; its archive
contains 511 cubin files. The merged 1.2.8 manifest has 566 entries / 508
referenced original hashes. This source rebuild emitted **469 unique cubins**.
Different entries can share a generated binary, and the compatibility treatment
of the retired `TREE=1` metadata uses the pinned, unconditional tree-sum source.
The archive's 22 unreferenced legacy files were not copied into the rebuilt set.

## Environment and method

- Linux ARM64 / aarch64, Python **3.12.3**.
- Triton **3.8.0**, explicit **`cuda:121:32`** target.
- CPU PyTorch **2.11.0+cpu**, used for reference-module imports, not GPU arithmetic
  or RoPE generation.
- `CUDA_VISIBLE_DEVICES=` and `TRITON_ALWAYS_COMPILE=1`; isolated compiler cache.
- Exact generator source bytes and manifest cases from the published input pins.
- Historical PTX filename labels recovered from static ELF/DWARF metadata; no
  supplied cubin loaded or executed through CUDA.

All 566 entries were compiled through the explicit-case compiler. The final
comparison then **rehashed those generated PTX/cubin files without recompiling**,
using the published `compare` command. This final check scopes addition receipts
to their actual explicit-case entries: five older captured entries have matching
shapes but different provenance and are correctly marked with **no recorded PTX
reference**, rather than borrowing a later addition's hash. Seventeen duplicate
manifest references account for the difference between 212 checked entries and
195 distinct recorded PTX hashes.

The full run checked 23 older indexer entries through the explicit `TREE=1`
compatibility lowering documented in [the rebuild guide](../../docs/AOT-SOURCE-REBUILD.md).
No other unknown constexpr is discarded.

## Evidence and interpretation

- [Full checked summary](REBUILD-V128.json) and [566-entry comparison](COMPARISON-V128.json).
- [Six-case summary](REBUILD-HC.json) and [six-case comparison](COMPARISON-HC.json).
- [Source pins and compiler cases](../../tools/dsv41-tp3/SOURCES.json).
- [Commands, dependency paths and debug-filename explanation](../../docs/AOT-SOURCE-REBUILD.md).

**PTX identity does not imply ELF/cubin identity.** This check demonstrates that
all available, applicable recorded PTX hashes match and that all 24 added 1.2.8
attention binaries reproduce exactly. It does **not** establish a byte-identical
rebuild of the older base kit: those cubin hashes differ, and historical PTX
references are unavailable for many original captured entries. Toolchain,
assembler and original-source provenance need to be distinguished when comparing
that older output. We do not assign every difference to one unverified cause.

The published status remains `REBUILD_DIFFERS_NOT_GPU_QUALIFIED` for the full set.
Changed source-built artifacts need independent numerical qualification and
reviewed hashes before installation. Production launchers retain their original
qualified-asset checks.

Regression coverage: five new AOT-source tests, five existing 1.2.8 release tests,
ten existing TensorFold package tests, and six current 4.0.2 release tests.
The original source archives, runtime profiles, benchmark results and posters
remain intact.
