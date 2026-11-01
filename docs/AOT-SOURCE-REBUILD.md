# Rebuild the TP3 Triton AOT kernels from source

This addresses [issue #1](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-tp4-spark/issues/1),
including the original TP3 kit and the attention additions used by **TP3 1.2.8 / AF**.
The historical releases, compiled reference assets and production profiles are unchanged.
Use the current repository checkout for these tools; they were added after the
immutable `v1.2.8` tag. The recipe still selects the original, pinned 1.2.8 source.

## Sources and provenance

| Source | Published location |
| --- | --- |
| TP3 `_hc_mix_pf` generator | [`tools/dsv41-tp3/tp3_triton_kernels.py`](../tools/dsv41-tp3/tp3_triton_kernels.py) |
| Historical explicit-case compiler, unchanged | [`tools/dsv41-tp3/compile_triton.py`](../tools/dsv41-tp3/compile_triton.py) |
| Reference row kernels | [`python-reference/src/tensorfold/families/deepseek_v41/cuda/kernels.py`](../python-reference/src/tensorfold/families/deepseek_v41/cuda/kernels.py) |
| Markov generators | [`python-reference/src/tensorfold/families/deepseek_v41/cuda/markov.py`](../python-reference/src/tensorfold/families/deepseek_v41/cuda/markov.py) |
| EXL3 prompt GEMM generator | [`python-reference/src/tensorfold/cuda/exl3/prefill.py`](../python-reference/src/tensorfold/cuda/exl3/prefill.py) |
| Complete reference package and import dependencies | Existing [`tensorfold-rope-reference.tar.gz`](../tensorfold/artifacts/tensorfold-rope-reference.tar.gz), extracted as `reference/` |
| 1.2.8 masked-attention generator | [`tensorfold/prefill-128/kernels.py`](../tensorfold/prefill-128/kernels.py) |
| 1.2.8 auxiliary CUDA source | [`tensorfold/prefill-128/experts_ragged.cu`](../tensorfold/prefill-128/experts_ragged.cu) |
| All 542 base launch cases | [`base-cases.json`](../tools/dsv41-tp3/base-cases.json) |
| All 24 additional 1.2.8 cases | [`v128-cases.json`](../tools/dsv41-tp3/v128-cases.json) |
| Source hashes / compiler target | [`SOURCES.json`](../tools/dsv41-tp3/SOURCES.json) |

The first requested file has SHA-256
`4301f70b936fd7e8cac6ce4406f6ad372296d626660dedbb015fbc5a3fc85596`.
The reference `kernels.py` has SHA-256
`723c3af1c9c1608f7c6439eb4e05d1cadfc7e633727a3b71f10552a67dd4a530`.
Both match the existing [six-case provenance](../tensorfold/hc-prefill/COMPILE-PROVENANCE.json).
The reference file was already inside the reference archive; the direct copies
make the generators browsable. Import the **complete extracted package**, not
the partial browsing copy in `python-reference/`.

The base case list is a complete enumeration of the published native manifest:
types, constexprs, alignment, warp count and PDL. Compiler stages are explicit:
historical addition-case options take precedence; other baseline attention uses
one stage, EXL3 GEMM four, other baseline kernels three. The seed helper is also
included for provenance, but seeding fixtures is **not** a complete rebuild.

There is one explicit compatibility lowering: 23 older indexer entries retain
the selector constant `TREE=1`. The pinned native `tri_index.zig` documents that
Python reference `8b1fe210` now always uses the tree sum and has removed that
argument. The wrapper compiles that unconditional tree implementation and
retains `TREE=1` in the output manifest. It rejects `TREE=0`; it does not silently
discard unknown constants or claim to have recovered an older Python source.
Those entries are identified individually in the comparison output.

| Set | Launch entries | Distinct referenced cubins | Cubin files shipped |
| --- | ---: | ---: | ---: |
| Original TP3 base | 542 | 489 | 511 |
| TP3 1.2.8 merged set | 566 | 508 | Base archive plus overlay |
| Original short-prefill repair only | 6 | 6 | Already included in the base |

The issue's count of **511** is correct. Twenty-two files in that archive are
not referenced by the final base manifest. The full rebuild compiles every
**reachable launch entry**; it does not copy unused legacy binaries into its
output. Repeated entries can legitimately produce the same cubin. The 1.2.8
overlay has 24 entries but 19 new cubins; some new entries also share existing
binary code. These counts describe this pinned kit, not all future request shapes.

## Offline compiler environment

Use Linux, Python **3.12**, Triton **3.8.0** and explicit target
`GPUTarget('cuda', 121, 32)`. A GPU is not needed for AOT compilation.
CPU PyTorch supplies import-time types for the reference wrappers; no tensors
are allocated on a GPU, no model is loaded and no CUDA kernel is launched.

One isolated setup, matching the CPU import environment used for the new check:

```sh
python3.12 -m venv /srv/ds41f-aot-venv
. /srv/ds41f-aot-venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install triton==3.8.0
```

This environment is separate from serving and from the CPU **RoPE generation**
environment. Its torch version does not replace the qualified RoPE generator.
The target's assembler comes from the Triton toolchain; a matching Triton
version alone does not promise identical assembler output across wheel builds.

From the repository root, extract the hash-verified inputs into a **new** directory:

```sh
python tensorfold/extract.py --out /srv/ds41f-aot-inputs
python tools/dsv41-tp3/rebuild.py audit
export CUDA_VISIBLE_DEVICES=
export TRITON_ALWAYS_COMPILE=1
export TRITON_CACHE_DIR=/srv/ds41f-aot-compiler-cache
```

`TRITON_ALWAYS_COMPILE=1` prevents a cached cubin from masquerading as a fresh
source build. The tools require the GPU to be hidden and refuse an existing
output directory. They never install artifacts or change running services.

## Reproduce the six cases requested in the issue

The portable invocation of the **unchanged historical compiler** is:

```sh
PYTHONPATH="/srv/ds41f-aot-inputs/reference/src:$PWD/tools/dsv41-tp3" \
  python tools/dsv41-tp3/compile_triton.py \
  --cases tensorfold/hc-prefill/COMPILE-CASES.json \
  --out /srv/ds41f-aot-hc-original
```

This produces `aot.json`, `cubins/` and `provenance.json`, including each
`ptx_sha256`. Its exact cases retain their published SHA-256. This is the six-case
addition manifest; the historical full kit appended those entries to the
preceding 536 entries. It does not by itself regenerate the full kit.

**For comparison to the recorded PTX hashes, account for debug filenames.**
Triton emits absolute `.file` labels into PTX, including its own `standard.py`.
A new checkout or virtual-environment path therefore changes the PTX hash even
with the identical kernel source. Use the checked wrapper to reproduce those
labels without creating historical paths:

```sh
python tools/dsv41-tp3/rebuild.py build --recipe hc-prefill \
  --reference /srv/ds41f-aot-inputs/reference \
  --out /srv/ds41f-aot-hc-checked --historical-debug-paths
```

That option reads the existing ELF's DWARF as **static metadata**, then changes
only Triton's source-filename labels. It neither imports code from those paths,
executes the supplied cubin, nor modifies generated PTX instructions. Omit the
option if you prefer local debug paths; expect historical raw PTX hashes to differ.

## Compile the complete base or 1.2.8 manifest

Choose one output set, or use separate new directories for both:

```sh
python tools/dsv41-tp3/rebuild.py build --recipe base \
  --reference /srv/ds41f-aot-inputs/reference \
  --out /srv/ds41f-aot-base --historical-debug-paths

python tools/dsv41-tp3/rebuild.py build --recipe v128 \
  --reference /srv/ds41f-aot-inputs/reference \
  --out /srv/ds41f-aot-128 --historical-debug-paths
```

`v128` compiles all **566** entries: 542 original plus 24 masked-attention
variants. It does not reuse the supplied cubins as output. Each result includes:

- `aot.json`: rebuilt native manifest with new binary hashes and the original
  runtime matching metadata (`nospec`, alignment, types and constexprs).
- `cubins/` and `ptx/`: freshly generated binary and intermediate files.
- `comparison.json`: per-entry generated/expected cubin and available PTX hashes.
- `BUILD.json`: totals, toolchain, debug-path mode and explicit qualification status.

The compiler validates the native ABI, including the two Triton scratch-pointer
arguments and scratch contract. A mismatch against a **recorded PTX** hash makes
the command fail. It records cubin differences separately; add
`--require-exact-cubins` to fail on any binary difference as well. Historical PTX
receipts exist for selected additions, **not every original baseline entry**;
missing references are `null`, never counted as passing comparisons.

To rehash/recompare a completed base or 1.2.8 source build without compiling it
again, write the new comparison to a separate directory:

```sh
python tools/dsv41-tp3/rebuild.py compare \
  --built /srv/ds41f-aot-128 --out /srv/ds41f-aot-128-recheck
```

The addition receipts apply to the explicit-case compiler's entries. They are
not attached to older captured upstream entries merely because their shapes
match; runtime matching metadata and provenance can differ.

### 1.2.8 auxiliary CUDA kernel

The existing measured CUDA build command is unchanged:

```sh
nvcc --cubin -arch=sm_121 -O3 -std=c++17 --fmad=true --ftz=false \
  --prec-div=true --prec-sqrt=true -Xptxas=-v \
  tensorfold/prefill-128/experts_ragged.cu \
  -o /srv/ds41f-aot-128/experts_ragged.cubin
```

The base CUDA fatbins are separate from Triton. Their CUDA source remains in
the pinned native source archive; this Triton command does not rebuild them.

## Scope of reproducibility and installation

See [the actual offline rebuild results](../results/20261101-aot-source/README.md).
PTX identity and cubin identity are distinct checks. A different assembler build
can emit different ELF bytes from identical PTX. A successful compile/ABI check
is also distinct from numerical qualification on your own GB10 cluster.

The existing production builder/installers intentionally continue to verify
their **qualified artifact hashes**. This addition does not weaken those checks
or silently install new binaries. Source-built output with different hashes
needs its own reviewed artifact manifest and numerical qualification before
being substituted into a runtime. In particular, the original 1.2.8 asset
manifest includes the unused historical files too; a reachable-kernel rebuild
is not a byte-for-byte replacement of that whole archive.

## Attribution

The reference kernels and engine are Ashley Hart's **TensorFold**, under their
original Apache-2.0 license, retained at [`python-reference/LICENSE`](../python-reference/LICENSE).
The TP3 generator is our adaptation of TensorFold's `_hc_mix_rows` arithmetic;
its upstream reference remains in the source. EXL3/Turboderp and the additional
1.2.8 contributors retain their existing notices and [credits](SOURCES.md#128-tp3-prefill-work).
Publishing build tooling does not claim authorship of the upstream kernels.
