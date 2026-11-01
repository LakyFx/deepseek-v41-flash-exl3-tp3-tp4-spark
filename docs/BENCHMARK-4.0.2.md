# TensorFold 4.0.2 - benchmark results

Public release name for the unchanged combined **4.0.1a** production payload,
measured 1 November 2026. Four DGX Sparks/GB10, switchless RoCE, 2200MHz GPU cap,
Linux ARM64, native TensorFold. DeepSeek-V4.1-Flash-1017 source revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`; LakyFx mixed EXL3 ~3.510072157 bpw
routed experts, weight revision `8621801c1fc74aece1fa83e67c6ade3fde1e7894`.
C16, adaptive R48, 9,000,000 physical KV positions, 3.5GiB head checkpoints,
8192-token compute chunks, 2048-token KV quantum, 401408-token context.

## Flyer results

| Concurrency | Code sum (tok/s) | Prose sum (tok/s) | MAX per task (tok/s) | MAX total (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| **C1** | **112.9** | **88.1** | 86.5 | **86.5** |
| **C3** | **184.9** | **151.3** | 44.8 | **134.5** |
| **C6** | **244.5** | **198.4** | 29.7 | **178.0** |
| **C12** | **302.2** | **258.7** | 18.9 | **226.5** |
| **C16** | **320.1** | **285.4** | 15.4 | **246.0** |

The two decode protocols are different. Code/prose uses the short authored
diverse-prompt screen, **not SparkDash**: greedy, thinking OFF, temperature0,
top_p1, seed42, 512 tokens/request; 2 consecutive passes, display only pass2.
The reported sum adds each request's native `tokens_per_second`, excludes
prefill and is not an instantaneous peak or a common shared-window aggregate.

MAX uses the frozen agent/tool workflow: thinking MAX, exact non-greedy,
temperature 0.6, top_p 0.95, top_k 20, seed 42. The request explicitly sends
temperature/seed and leaves top_p/top_k to this version's native served-lane
defaults; those defaults are pinned in the published source. It performs real tool calls, code,
tests and repairs. The original C1-C12 fixtures are unchanged; C13-C16 extend
the same factory. It is an authored reproducible workflow, not captured private
work and not a same-token deterministic decode microbenchmark.

## Full-concurrency medians

The MAX total is the ordinary p50 of native aggregate live decode samples at
actual full C. Per-task is p50 of the same-window aggregate/C, a temporal median
of the concurrent per-task average, **not** a cross-request distribution median.
The native rolling rate covers the trailing2s and is observed about every2s.
Every selected entire window must be inside an already streaming reply of
every task, with a0.5s boundary exclusion. Current+previous observed C must match;
finished/input/output counters bind the sample to the exact measured wave.
Prefill, tool gaps crossing a window, idle time and lower-C completion tails
are excluded. All eligible slow samples are kept; no fastest-window selection.
Small independent rounding differences are expected. Single-run valid coverage:

| Load | Valid samples | Median per task | Median total |
| --- | ---: | ---: | ---: |
| C1 | 13 | 86.500 | 86.5 |
| C3 | 7 | 44.833 | 134.5 |
| C6 | 9 | 29.667 | 178.0 |
| C12 | 15 | 18.875 | 226.5 |
| C16 | 16 | 15.375 | 246.0 |

Sample counts are limited. These measurements characterize observed full-load
decode intervals; they are not the throughput of the complete workflow.
[All C1-C16 selected samples and stream boundaries](../results/20261101-4.0.2/HEAVY-FULL-CONCURRENCY.json).

## Complete workflow, including prefill, tools and repairs

136/136 final tasks correct; 606 model requests, 51 recoverable tool errors,
zero inference retries. All four model processes remained stable and had no
model/cgroup swap during the test. Full campaign duration: about30.41minutes.
Tool-error count is stochastic and is not a demonstrated quality improvement
caused by memory management or speculative decoding.

| C | Request decode median | Whole-wave workflow tok/s | Seconds | Correct | Tool errors |
| --- | ---: | ---: | ---: | --- | ---: |
| C1 | 106.99 | 96.83 | 48.2 | 1/1 | 4 |
| C2 | 73.01 | 119.77 | 35.1 | 2/2 | 1 |
| C3 | 51.76 | 117.10 | 58.6 | 3/3 | 2 |
| C4 | 40.74 | 135.51 | 59.7 | 4/4 | 1 |
| C5 | 36.30 | 145.08 | 77.5 | 5/5 | 4 |
| C6 | 31.72 | 159.17 | 67.0 | 6/6 | 0 |
| C7 | 28.04 | 165.12 | 85.9 | 7/7 | 1 |
| C8 | 25.56 | 170.29 | 89.7 | 8/8 | 1 |
| C9 | 24.06 | 175.51 | 107.6 | 9/9 | 3 |
| C10 | 21.79 | 171.02 | 92.7 | 10/10 | 1 |
| C11 | 20.32 | 188.32 | 108.2 | 11/11 | 1 |
| C12 | 19.57 | 177.15 | 147.7 | 12/12 | 5 |
| C13 | 17.89 | 177.23 | 163.2 | 13/13 | 5 |
| C14 | 17.23 | 191.88 | 149.6 | 14/14 | 4 |
| C15 | 16.51 | 134.19 | 378.0 | 15/15 | 14 |
| C16 | 15.68 | 195.81 | 155.5 | 16/16 | 4 |

The native per-request median above includes requests at lower load as tasks
finish. Whole-wave tokens/time includes prefill, tools and tails. Neither is
the full-C median on the flyer, and multiplying this request median by nominal
C would be incorrect. C15 took ~378s, including one long repair/reasoning path;
it remains in the results. [All metric-only requests](../results/20261101-4.0.2/HEAVY-WORKFLOW.json).

## Short screen - both passes

| Pass | C | Code native sum | Prose native sum | Code whole-wave | Prose whole-wave |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | C1 | 111.41 | 86.63 | 107.08 | 84.25 |
| 1 | C3 | 177.82 | 147.82 | 167.60 | 142.81 |
| 1 | C6 | 228.69 | 190.99 | 204.88 | 179.12 |
| 1 | C12 | 285.82 | 248.49 | 244.94 | 226.15 |
| 1 | C16 | 307.42 | 275.80 | 271.19 | 257.02 |
| 2 | C1 | 112.90 | 88.07 | 112.68 | 87.85 |
| 2 | C3 | 184.88 | 151.25 | 182.45 | 147.29 |
| 2 | C6 | 244.52 | 198.40 | 222.72 | 191.35 |
| 2 | C12 | 302.19 | 258.66 | 263.85 | 241.15 |
| 2 | C16 | 320.09 | 285.36 | 288.55 | 263.91 |

Pass2 warms the adaptive controller, KV, CUDA and disk caches together; this
does not isolate controller learning. The same public short fixtures/formulas
are in [benchmarks/short_decode](../benchmarks/short_decode/); use `--c16`
to include the final cell. Historical default concurrency lists are unchanged.
The workflow runner takes `--max-concurrency 16 --world-size 4` for this release;
its receipt must include verified ranks `[0,1,2,3]`. The original 78 input hashes
are checked before adding the 58 C13-C16 tasks from the same factory.

## Cold prefill - no cached input

| Input tokens | Median computed tok/s | Samples | Cache hits |
| --- | ---: | ---: | ---: |
| 1,024 | 2152.34 | 3 | 0 |
| 2,048 | 2777.05 | 3 | 0 |
| 4,096 | 3238.30 | 3 | 0 |
| 8,192 | 3636.21 | 3 | 0 |
| 32,768 | 4205.56 | 3 | 0 |
| 65,536 | 4263.53 | 3 | 0 |
| 131,072 | 4231.87 | 3 | 0 |

Computed-input tokens divided by native prefill seconds; zero-hit samples,
three independent inputs per size. This differs from small warm workflow
prefills and effective cache-inclusive input throughput. [All21 samples](../results/20261101-4.0.2/COLD-PREFILL.json).

## Latency and qualification

Native post-prefill first-token time is the host `prefilled` event to first
nonempty token event; about3microseconds in many samples, not isolated GPU kernel
latency and not user-visible TTFT. Preparation, total TTFT and SSE delivery are
separate fields in the request evidence; do not compare it to end-to-end TTFT.
The old upper bound includes server preparation and remains separately recorded.

Two concurrent synthetic shape/color image questions passed at MAX/8192 thinking
and32768 final budget; duplicate submission joined the same job. Both succeeded
on their first attempt, so live repair quality was not exercised by that smoke.
Recovery/cancellation/deadline protocols have separate CPU regression coverage.

Published evidence is allowlisted metrics, hashes and timing boundaries. It
contains no generated answers, user prompts, runtime job IDs, tokens, private
addresses or operator paths. Original historical recipes/reports are preserved.
The public release itself ran no new GPU benchmark or live restart.

Run `python3 tools/audit_402.py` offline to reproduce sums/medians, sample-window
selection invariants, zero-hit prefill and source/payload identity.
