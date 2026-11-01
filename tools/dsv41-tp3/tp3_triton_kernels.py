"""TP3 additions absent from the pinned public Python kernel source.

_hc_mix_pf implements the native prompt launch ABI using the upstream
_hc_mix_rows arithmetic, with its original rank-ordered FP32 sum and BF16
rounding. This is a compatibility implementation, NOT a claim that it is the
unpublished prompt kernel or that its GPU performance/rounding is qualified.
Reference dependency: TensorFold deepseek-v41-tp2 8b1fe210d5e0d68e0cdb3b14755c135465c9ffa8.
"""
import triton
import triton.language as tl
from tensorfold.families.deepseek_v41.cuda.kernels import _hc_mix_rows
from tensorfold.families.deepseek_v41.cuda import kernels as reference_kernels

SOURCE_DEPENDENCIES = (reference_kernels,)


@triton.jit
def _hc_mix_pf(X, XO, G, RS, POST, COMB, FN, PART, rows,
               WIDE: tl.constexpr, D: tl.constexpr, NB: tl.constexpr,
               SUB: tl.constexpr, MB: tl.constexpr, WORLD: tl.constexpr,
               POSTED: tl.constexpr, RB: tl.constexpr):
    # Native grid: (ceil(rows/RB) * NB, 1, 1). Each program owns one
    # row block / K block and all 24 mixes, so only mg=0 writes XO and SS.
    b = tl.program_id(0) % NB
    r0 = (tl.program_id(0) // NB) * RB
    KB: tl.constexpr = WIDE // NB
    tl.static_assert(MB == 8)
    tl.static_assert(KB % SUB == 0)
    tl.static_assert(D % KB == 0)
    k = tl.arange(0, SUB)
    for mg in tl.static_range(3):
        m = mg * MB + tl.arange(0, MB)
        ws = ()
        for t in tl.static_range(KB // SUB):
            ws = ws + (tl.load(FN + m[:, None] * WIDE + b * KB + t * SUB + k[None, :]),)
        for i in tl.static_range(RB):
            if r0 + i < rows:
                _hc_mix_rows(X, XO, G, RS, POST, COMB, PART, ws,
                             r0 + i, b, mg, m, k, WIDE, D, NB, KB,
                             SUB, MB, WORLD, POSTED)
