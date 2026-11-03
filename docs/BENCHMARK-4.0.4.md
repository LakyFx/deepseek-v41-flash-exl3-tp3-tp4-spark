# TensorFold TP4 4.0.4 - qualification and measurements

**Production 4.0.4 is the unchanged selected 4.0.4-B compute payload.**
DeepSeek-V4.1-Flash-1017, mixed EXL3 ~3.51 bpw routed experts, four DGX
Sparks, switchless TP4, balanced401a-v1, C16/adaptive R48/9M KV/3.5 GiB.
Publication reuses these measurements; no new inference or service changes.

## Flyer figures and definitions

| Concurrency | Code sum tok/s | Prose sum tok/s | MAX median/task tok/s | MAX median total tok/s |
| --- | ---: | ---: | ---: | ---: |
| **C1** | **115.4** | **88.6** | 86.5 | **86.5** |
| **C3** | **186.6** | **152.6** | 48.8 | **146.5** |
| **C6** | **245.9** | **200.5** | 29.6 | **177.8** |
| **C12** | **310.1** | **265.2** | 19.1 | **228.8** |
| **C16** | **331.7** | **287.5** | 16.3 | **260.3** |

Code/prose are sums of native request decode rates, thinking OFF, temperature 0,
top_p 1, seed 42, 512 tokens/request. Two consecutive passes; only pass 2 is
shown. The authored diverse fixtures and body hashes match across candidates.
They are our short screen, **not SparkDash or RigMark**. The portable runner
is [benchmarks/short_decode](../benchmarks/short_decode/) with its C16 option.

MAX workflow uses the unchanged frozen agent/tool task factory, thinking MAX,
temperature 0.6, top_p 0.95, top_k 20, seed 42 and the original evaluators.
The poster uses time-sampled native rolling two-second rates only while
all requested tasks are actually streaming. Require the whole sample window
inside concurrent streaming intervals, with 0.5-second edge margins; omit
prefill, tool gaps and completed-task tails. Select all qualifying samples
without choosing fast samples. Per task is total/C before taking the median.
The two columns share the exact same samples. Half-up one-decimal rounding.
Counts at C1/C3/C6/C12/C16: 13/7/10/14/16. Small samples are descriptive;
these medians are not whole-workflow completion throughput or peaks.

## Matched short screen - full-cell wall throughput

The tables below use completion tokens / whole cell wall seconds, including
fill and drain. This differs from the native decode sum on the flyer.

### Code

| Variant | C1 | C3 | C6 | C12 | C16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4.0.3 | 113.68 | 180.23 | 215.09 | 266.97 | 292.51 |
| 4.0.4-A | 115.39 | 184.50 | 219.71 | 271.79 | 289.05 |
| 4.0.4-B | 114.90 | 181.80 | 223.53 | 273.78 | 295.55 |
| 4.0.4-C | 112.04 | 178.35 | 213.60 | 265.48 | 280.77 |
| 4.0.4-BC | 113.57 | 181.91 | 221.65 | 267.86 | 293.44 |
| 4.0.4-D | 114.86 | 184.01 | 218.90 | 270.31 | 292.37 |
| 4.0.4-BCD | 112.34 | 179.99 | 219.48 | 269.55 | 292.94 |

### Prose

| Variant | C1 | C3 | C6 | C12 | C16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4.0.3 | 87.92 | 148.22 | 189.92 | 241.66 | 268.45 |
| 4.0.4-A | 87.47 | 149.07 | 190.13 | 238.44 | 265.41 |
| 4.0.4-B | 88.51 | 147.75 | 192.87 | 242.98 | 268.89 |
| 4.0.4-C | 87.38 | 147.16 | 188.31 | 234.24 | 258.03 |
| 4.0.4-BC | 87.23 | 146.04 | 186.56 | 240.63 | 266.16 |
| 4.0.4-D | 87.22 | 148.29 | 188.97 | 240.93 | 260.17 |
| 4.0.4-BCD | 86.71 | 148.49 | 190.11 | 238.97 | 266.66 |

## Prefill, elapsed time and speculation

| Variant | Cold 64K median tok/s | Short pass 2 seconds | Accepted / drafted |
| --- | ---: | ---: | ---: |
| 4.0.3 | 4278.6 | 166.63 | 82.54% |
| 4.0.4-A | 4259.7 | 166.64 | 82.55% |
| 4.0.4-B | 4276.6 | 164.67 | 83.21% |
| 4.0.4-C | 4291.8 | 170.47 | 82.70% |
| 4.0.4-BC | 4291.3 | 166.83 | 83.17% |
| 4.0.4-D | 4291.0 | 167.06 | 83.30% |
| 4.0.4-BCD | 4301.2 | 166.69 | 83.41% |

Each cold test has three exactly 65,536-token inputs and zero cached tokens.
Prefill rate is computed tokens / native prefill duration, separately from
TTFT. Post-prefill first-token event gaps are also exported. A first token
can come from final prefill logits, so a microsecond gap is not an independent
GPU decode-step latency. No TTFT or fill/drain is hidden inside a decode label.

**Selected B:** 164.67 seconds versus the freshly measured 4.0.3 reference
166.63 seconds: 1.18% less elapsed, 1.19% higher fixed-work throughput.
Code wall throughput rose 3.9% at C6 and 2.6% at C12; prose was broadly
unchanged. These screening differences do not establish a universal gain.
Acceptance alone does not rank speed: BCD has higher acceptance but is slower.

A adds amortized histories, thinking compatibility and reported-CUDA-error
supervision. B adds guarded mid-M/WC EXL3 kernels to A. C adds an alternative
weight-read cache policy to A; its cubin compiler also differs, so a pure L1
effect is not isolated. BC combines B/C. D adds one-time file-cache advice
to A; those pages were already reclaimable, with no MemAvailable gain.
BCD combines all. C-control and BC-control are retained diagnostics, not
additional measured rows. Only A+B is enabled in production 4.0.4.

## Full frozen workflow, C1-C16

| C | Correct | Wall seconds | Native request median tok/s | Whole workflow tok/s | Correction events |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1/1 | 48.56 | 107.27 | 96.18 | 4 |
| 2 | 2/2 | 35.49 | 66.56 | 118.49 | 1 |
| 3 | 3/3 | 40.82 | 52.48 | 131.50 | 0 |
| 4 | 4/4 | 59.28 | 41.48 | 138.56 | 1 |
| 5 | 5/5 | 91.49 | 37.38 | 135.87 | 6 |
| 6 | 6/6 | 91.24 | 32.70 | 156.09 | 6 |
| 7 | 7/7 | 73.65 | 28.18 | 167.36 | 0 |
| 8 | 8/8 | 90.15 | 26.46 | 173.31 | 3 |
| 9 | 9/9 | 144.05 | 23.77 | 151.94 | 5 |
| 10 | 10/10 | 103.14 | 22.72 | 174.34 | 3 |
| 11 | 11/11 | 120.76 | 19.85 | 180.58 | 2 |
| 12 | 12/12 | 129.29 | 18.75 | 191.38 | 7 |
| 13 | 13/13 | 189.70 | 18.53 | 163.66 | 4 |
| 14 | 14/14 | 132.72 | 17.96 | 200.11 | 3 |
| 15 | 15/15 | 205.31 | 16.63 | 183.95 | 11 |
| 16 | 16/16 | 175.07 | 16.40 | 187.43 | 8 |

**136/136 correct, 1730.726 seconds total, 64 original tool/task correction events.**
No outer inference retries. Corrections are part of the original workflow;
all tasks eventually passed. Whole-workflow throughput includes tools and
varying output lengths. It is not the full-load median shown on the flyer.
No fresh paired 4.0.3 heavy run was performed; this is winner qualification.

## Memory, correctness and vision

- 579 samples per rank, approximately every three seconds. Model VmSwap and
  model cgroup swap stayed zero, one PID per rank, full memory PSI zero.
  Minimum MemAvailable across ranks was 8.84 GiB. Small host-wide swapout
  increments of 1-14 pages are separate from model paging.
- 1,608 exact-bit GPU kernel checks passed, including graph replay, split-K
  and unpack cases. Unsupported/fused shapes retain the original route.
- Fixed gateway regressions: 233 passed, one skipped. Live standalone jobs,
  inline Chat image history, Responses images, internal JSON and cancellation
  followed by a successful answer passed against the selected backend.
- Main image routes retain MAX/8192/32768/C2, including when callers send
  thinking=false. The bounded internal visual-note extractor retains its
  existing thinking-OFF policy. The red-square/blue-circle fixture checks
  routing and basic correctness, not all possible vision-analysis quality.
- Reported fatal-CUDA supervision was checked offline; no deliberate fatal
  CUDA fault was injected in production. A silent hang is a separate limit.

## Reproduce and audit

[Install the exact payload](DS41F-4.0.4.md). [Numeric exports](../results/20261103-4.0.4/)
contain both short passes for all seven measured versions, per-request
counters, five speculation depths, zero-hit prefill samples, full workflow
metrics, memory summary, flyer sample values and production hashes.
Prompts, answers, image bodies, job/request identifiers, private addresses
and operator configuration are excluded. Public short/workflow fixtures
remain available in their existing benchmark directories.
Run `python3 tools/audit_404.py` offline. It issues no model requests.
