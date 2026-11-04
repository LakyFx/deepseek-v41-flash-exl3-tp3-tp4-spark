# TP4 TensorFold 4.0.5 qualification

**Qualification was performed with public ingress OFF. This exact candidate was subsequently approved as production 4.0.5.**

Functional JSON/vision suites pass and the measured native model remained stable. The strict all-tasks-correct heavy gate does **not** pass: each heavy run finished 135/136 tasks correctly, with different failed tasks. Those failures are preserved and described below. Completion of the campaign is separate from a perfect correctness result.

Base: exact production 4.0.4/B; add the image-aware prefix-cache integration and TP3 1.3.0 JSON gateway compatibility. All three smart controllers, B kernels, balanced401a weights, C16/adaptive R48 ceiling, 9M KV, 3.5 GiB checkpoints, 320 MiB relocation and 8K logical/2K physical prefill remain unchanged. No 4.0.4 baseline rerun.

## Fast code/prose screening

Same historical diverse prompts; thinking OFF, greedy, temperature 0, top_p 1, seed 42, 512 output tokens. This is our native short screen, not SparkDash/RigMark. All input-body hashes match 4.0.4/B. All three passes are retained; no model reload or learner/cache reset between passes.

Aggregate = completion tokens divided by full cell wall time, including fill/drain. Request median = median of native per-request decode rates. Neither is a full-overlap aggregate median; the nonstreaming screen does not export token event timestamps.

| Version/pass | C | Code aggregate tok/s | Prose aggregate tok/s | Code request median tok/s | Prose request median tok/s |
|---|---:|---:|---:|---:|---:|
| 4.0.4 historical pass 2 | 1 | 114.90 | 88.51 | 115.41 | 88.61 |
| 4.0.5 pass 1 | 1 | 105.46 | 86.49 | 111.20 | 89.08 |
| 4.0.5 pass 2 | 1 | 113.60 | 87.52 | 114.51 | 87.73 |
| 4.0.5 pass 3 | 1 | 113.34 | 87.89 | 113.57 | 88.37 |
| 4.0.4 historical pass 2 | 3 | 181.80 | 147.75 | 61.65 | 50.81 |
| 4.0.5 pass 1 | 3 | 168.86 | 145.69 | 59.80 | 50.59 |
| 4.0.5 pass 2 | 3 | 179.20 | 146.25 | 62.39 | 50.29 |
| 4.0.5 pass 3 | 3 | 183.20 | 150.29 | 62.02 | 50.58 |
| 4.0.4 historical pass 2 | 6 | 223.53 | 192.87 | 40.26 | 33.41 |
| 4.0.5 pass 1 | 6 | 207.19 | 185.96 | 37.63 | 32.79 |
| 4.0.5 pass 2 | 6 | 221.94 | 192.14 | 39.23 | 33.11 |
| 4.0.5 pass 3 | 6 | 216.23 | 191.71 | 38.84 | 33.45 |
| 4.0.4 historical pass 2 | 12 | 273.78 | 242.98 | 24.92 | 21.63 |
| 4.0.5 pass 1 | 12 | 249.96 | 234.32 | 23.38 | 21.56 |
| 4.0.5 pass 2 | 12 | 269.30 | 248.13 | 24.49 | 21.58 |
| 4.0.5 pass 3 | 12 | 274.89 | 248.41 | 25.02 | 22.11 |
| 4.0.4 historical pass 2 | 16 | 295.55 | 268.89 | 19.91 | 17.65 |
| 4.0.5 pass 1 | 16 | 284.87 | 266.84 | 19.49 | 17.64 |
| 4.0.5 pass 2 | 16 | 296.86 | 270.93 | 19.96 | 17.78 |
| 4.0.5 pass 3 | 16 | 299.65 | 273.18 | 19.95 | 17.81 |

| Fast pass | Summed cell seconds | Accepted / proposed |
|---|---:|---:|
| 1 | 172.02 | 21680/26373 (82.21%) |
| 2 | 164.67 | 21302/25808 (82.54%) |
| 3 | 163.59 | 21245/25490 (83.35%) |

Warming includes KV, CUDA graphs, disk caches and adaptation together. A faster later pass alone does not establish learning as the cause.

## Cold prefill

Three frozen 65,536-token inputs, all cached_tokens=0. Median **4288.7 tok/s**; historical 4.0.4/B **4276.6 tok/s**. Token hashes are preserved in raw evidence.

Post-prefill first-token event latency excludes prefill and queueing; a token may come directly from the final prefill logits. Microsecond event gaps are not a separate GPU decode-step benchmark. Complete native event/delivery/TTFT timings are retained.

| Sample | Uncached input | Prefill seconds | Prefill tok/s | Post-prefill event ms |
|---|---:|---:|---:|---:|
| 65536-1 | 65536 | 15.2812 | 4288.7 | 0.002720 |
| 65536-2 | 65536 | 15.2824 | 4288.3 | 0.001952 |
| 65536-3 | 65536 | 15.2798 | 4289.0 | 0.003408 |

## Frozen heavy workflow

Unchanged historical task factory, initial inputs, tools and evaluators; MAX thinking, temperature 0.6, top_p 1, top_k 0, min_p 0, seed 42, max_tokens 65,536, C1-C16, 136 tasks per pass. Original within-task correction steps are counted; no outer inference retries. Both passes run on the same model generation. Different generated work and corrections affect task time independently of decode rate.

### Pass 1

| C | Correct tasks | Wall s | Request decode median tok/s | Aggregate workflow tok/s | Output tokens | Corrections |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 45.85 | 109.75 | 97.84 | 4486 | 4 |
| 2 | 2/2 | 34.26 | 71.65 | 117.85 | 4037 | 1 |
| 3 | 3/3 | 158.21 | 56.50 | 104.23 | 16491 | 5 |
| 4 | 4/4 | 63.26 | 42.76 | 128.25 | 8113 | 2 |
| 5 | 5/5 | 77.09 | 38.30 | 135.49 | 10445 | 3 |
| 6 | 6/6 | 62.29 | 32.08 | 161.45 | 10057 | 1 |
| 7 | 7/7 | 90.02 | 29.32 | 151.97 | 13681 | 0 |
| 8 | 8/8 | 89.91 | 25.56 | 171.06 | 15380 | 4 |
| 9 | 9/9 | 107.86 | 25.64 | 172.71 | 18629 | 1 |
| 10 | 10/10 | 116.11 | 22.58 | 170.02 | 19740 | 5 |
| 11 | 11/11 | 196.80 | 22.11 | 176.05 | 34647 | 12 |
| 12 | 12/12 | 131.89 | 20.12 | 183.14 | 24154 | 7 |
| 13 | 13/13 | 130.68 | 18.31 | 192.85 | 25202 | 4 |
| 14 | 14/14 | 142.39 | 17.43 | 198.92 | 28325 | 5 |
| 15 | 14/15 | 234.53 | 17.46 | 162.51 | 38114 | 12 |
| 16 | 16/16 | 162.12 | 15.67 | 196.64 | 31879 | 3 |

Measured waves: 30.72 min. Correct 135/136; corrections 69. Token-weighted prefix hit 84.55%.

C15 replica 9 failed the unchanged 16-turn protocol limit after invalid tool calls / unsupported Raise syntax. The original strict runner stopped before C16. Failure remains FAIL, with original evidence retained. Only the unattempted C16 wave was then run on the same native generation/PIDs; no C15 replay or extra inference retry. There was an operator-control pause before C16, so pass 1 was not fully uninterrupted. The follow-up runner records task correctness separately and completes independent later waves; it still aborts on safety/transport faults and does not relax any task evaluator.

### Pass 2

| C | Correct tasks | Wall s | Request decode median tok/s | Aggregate workflow tok/s | Output tokens | Corrections |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 48.87 | 104.20 | 95.58 | 4671 | 4 |
| 2 | 2/2 | 35.14 | 72.20 | 119.66 | 4205 | 1 |
| 3 | 3/3 | 41.87 | 51.85 | 132.93 | 5565 | 0 |
| 4 | 4/4 | 56.52 | 42.89 | 141.99 | 8025 | 1 |
| 5 | 5/5 | 77.81 | 37.28 | 156.68 | 12191 | 3 |
| 6 | 6/6 | 75.01 | 31.86 | 160.41 | 12033 | 2 |
| 7 | 7/7 | 75.77 | 28.24 | 168.89 | 12797 | 1 |
| 8 | 8/8 | 89.75 | 26.62 | 175.26 | 15730 | 4 |
| 9 | 9/9 | 103.44 | 24.26 | 168.18 | 17397 | 2 |
| 10 | 10/10 | 100.59 | 22.13 | 177.15 | 17820 | 1 |
| 11 | 11/11 | 132.58 | 20.48 | 170.81 | 22646 | 2 |
| 12 | 12/12 | 140.74 | 19.66 | 190.63 | 26830 | 6 |
| 13 | 13/13 | 126.95 | 17.90 | 196.59 | 24958 | 1 |
| 14 | 14/14 | 132.10 | 16.87 | 195.60 | 25839 | 1 |
| 15 | 14/15 | 180.39 | 16.23 | 186.67 | 33673 | 12 |
| 16 | 16/16 | 142.23 | 15.82 | 204.52 | 29088 | 2 |

Measured waves: 26.00 min. Correct 135/136; corrections 43. Token-weighted prefix hit 80.34%.

C15 replica 11 failed with missing_required_task_submission. Its edited code passed its tests, but the first finish_task call had incorrect accounting, and the corrected call exceeded the required four-to-six-sentence summary. The model then ended without a valid submission. Arguments were parsed JSON; this was not a malformed-JSON transport failure or the same task/16-turn failure as in pass 1. No replay or evaluator relaxation.

### Historical heavy comparison

Same frozen initial inputs; historical 4.0.4/B was not rerun. Request decode median is a per-request rate, not the sum of concurrently active streams. Wall time includes all task work and therefore changes with reasoning length and tool corrections.

| C | 4.0.4 request median | 4.0.5 pass 1 | 4.0.5 pass 2 | 4.0.4 wall s | 4.0.5 p1 wall s | 4.0.5 p2 wall s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 107.27 | 109.75 | 104.20 | 48.56 | 45.85 | 48.87 |
| 2 | 66.56 | 71.65 | 72.20 | 35.49 | 34.26 | 35.14 |
| 3 | 52.48 | 56.50 | 51.85 | 40.82 | 158.21 | 41.87 |
| 4 | 41.48 | 42.76 | 42.89 | 59.28 | 63.26 | 56.52 |
| 5 | 37.38 | 38.30 | 37.28 | 91.49 | 77.09 | 77.81 |
| 6 | 32.70 | 32.08 | 31.86 | 91.24 | 62.29 | 75.01 |
| 7 | 28.18 | 29.32 | 28.24 | 73.65 | 90.02 | 75.77 |
| 8 | 26.46 | 25.56 | 26.62 | 90.15 | 89.91 | 89.75 |
| 9 | 23.77 | 25.64 | 24.26 | 144.05 | 107.86 | 103.44 |
| 10 | 22.72 | 22.58 | 22.13 | 103.14 | 116.11 | 100.59 |
| 11 | 19.85 | 22.11 | 20.48 | 120.76 | 196.80 | 132.58 |
| 12 | 18.75 | 20.12 | 19.66 | 129.29 | 131.89 | 140.74 |
| 13 | 18.53 | 18.31 | 17.90 | 189.70 | 130.68 | 126.95 |
| 14 | 17.96 | 17.43 | 16.87 | 132.72 | 142.39 | 132.10 |
| 15 | 16.63 | 17.46 | 16.23 | 205.31 | 234.53 | 180.39 |
| 16 | 16.40 | 15.67 | 15.82 | 175.07 | 162.12 | 142.23 |

## Speculation at all five depths

Accepted/proposed counts are kept separately from reached counts. Depth proposal totals may differ from overall drafted totals because the native metrics have different accounting scopes.

| Test | Depth | Proposed | Reached | Accepted | Accepted/proposed |
|---|---:|---:|---:|---:|---:|
| fast 1 | 1 | 12913 | 12913 | 10901 | 84.42% |
| fast 1 | 2 | 7102 | 6538 | 5670 | 79.84% |
| fast 1 | 3 | 3348 | 2991 | 2692 | 80.41% |
| fast 1 | 4 | 1785 | 1608 | 1490 | 83.47% |
| fast 1 | 5 | 1027 | 895 | 838 | 81.60% |
| fast 2 | 1 | 13051 | 13051 | 11037 | 84.57% |
| fast 2 | 2 | 6829 | 6310 | 5511 | 80.70% |
| fast 2 | 3 | 3192 | 2876 | 2563 | 80.29% |
| fast 2 | 4 | 1647 | 1464 | 1350 | 81.97% |
| fast 2 | 5 | 972 | 831 | 778 | 80.04% |
| fast 3 | 1 | 12866 | 12866 | 10930 | 84.95% |
| fast 3 | 2 | 6801 | 6323 | 5579 | 82.03% |
| fast 3 | 3 | 3121 | 2818 | 2529 | 81.03% |
| fast 3 | 4 | 1643 | 1463 | 1370 | 83.38% |
| fast 3 | 5 | 955 | 833 | 773 | 80.94% |
| heavy 1 | 1 | 77585 | 77585 | 65669 | 84.64% |
| heavy 1 | 2 | 48806 | 44914 | 39309 | 80.54% |
| heavy 1 | 3 | 28016 | 25266 | 23066 | 82.33% |
| heavy 1 | 4 | 17574 | 16207 | 15203 | 86.51% |
| heavy 1 | 5 | 12135 | 11216 | 10822 | 89.18% |
| heavy 2 | 1 | 68979 | 68979 | 59376 | 86.08% |
| heavy 2 | 2 | 43465 | 40456 | 35920 | 82.64% |
| heavy 2 | 3 | 24851 | 22745 | 21063 | 84.76% |
| heavy 2 | 4 | 16242 | 15096 | 14245 | 87.70% |
| heavy 2 | 5 | 11276 | 10507 | 10184 | 90.32% |

## JSON and image tests

All final live suites passed: 11 JSON contracts/status checks, six single-image reuse/replacement cases, four two-image checkpoint cases, and functional vision/internal-JSON/cancellation checks. Public ingress stayed closed; only the isolated loopback gateway was used.

| Suite | Result | Scope |
|---|---|---|
| Client JSON | 11/11 | Chat object/schema/local refs/guided, Responses, SSE, invalid schema 400, tool intermediate, Chat/Responses image JSON |
| Single-image prefix cache | 6/6 | Chat and Responses: cold, same-image hit, changed-image invalidation; native counter accounting verified |
| Two-image prefix cache | 4/4 | Primer, new second image, unchanged history, changed second image; only the valid prefix before the changed image is retained |
| Functional compatibility | PASS | Thinking selectors; dedicated vision job, Chat SSE agent history and Responses; checkpoint/visual-note JSON; owned cancellation then a new successful answer |

JSON compatibility validates and repairs within bounded limits; it is not native constrained decoding and does not guarantee arbitrary outputs. Vision checks use owned synthetic fixtures. The frozen JSON fixture identifies itself as 1.3.0; the tested native and gateway candidate is 4.0.5.

## Memory and runtime stability

five-second samples from model-ready through final checks; initial rejected startup excluded

| Rank | Minimum MemAvailable GiB | Maximum model swap bytes | Maximum model cgroup swap bytes | Host swap in/out pages |
|---:|---:|---:|---:|---|
| 0 | 9.66 | 0 | 0 | 43 / 1 |
| 1 | 11.51 | 0 | 0 | 237 / 0 |
| 2 | 11.39 | 0 | 0 | 0 / 6 |
| 3 | 9.73 | 0 | 0 | 80 / 1 |

Host swap occupancy includes other processes and pre-existing pages; it is distinct from native model/cgroup swap. Samples are five seconds apart and cannot exclude every sub-second transient. No Linux swap policy or clocks were changed.

Native log suspect lines: none in the collected post-ready journals.

Final head graph-controller counters:

`smart_graph_stats version=4.0.3 digest=f1d68a109ac467fb generation=0 resident=233 ghost=0 hits=193303 misses=588 captures=588 eager=0 evictions=0 ghost_returns=0 quick_returns=0 capture_host_ns=2883581768 control_ns=1234637366 polls=763 destroy_ns=0 trim_ns=0 available_min=10393878528 reserve=1073741824 recovery_checks=0 recovery_growth_noisy=0`

The diagnostic component label version=4.0.3 is inherited from the unchanged graph controller. The loaded release is 4.0.5, verified by the profile hash, source revision, generation and all four process identities. Host control/capture counters are not GPU elapsed-time measurements.

## Startup qualification correction

The first test profile mistakenly added --benchmark-control to the production LAN listener. The native guard correctly refused startup because this optional interface requires loopback. No benchmark/client inference occurred. Removed the unnecessary flag: the unchanged heavy client owns HTTP sockets and cancellation by disconnect. A fresh common generation was then loaded; all measured passes follow that start. Rejected profile and startup logs remain preserved.

The isolated test gateway initially refused import because its helper environment enabled client JSON without declaring the DeepSeek backend adapter. Added the same backend/adapter selectors already present in the qualified profile, then started it successfully before any final-suite inference. This corrected the test environment, without changing the packaged gateway modules or native candidate.

## Interpretation

The short-screen total for pass 2 is effectively equal to historical 4.0.4/B; cold prefill is also comparable. Heavy decode differences change direction across concurrency cells, with no uniform decline. Heavy wave times and correction counts vary substantially: pass 1 generated 303,380 output tokens versus 273,468 in pass 2. Therefore the shorter second heavy run is not an isolated measurement of controller learning or engine acceleration.

The targeted image-cache checks verify both correct answers and native cache counters. They demonstrate same-image reuse and invalidation at changed images in these fixtures. The campaign is not a full-context 9M-pool pressure test or a 24-hour stability run. Existing controller algorithms and capacity were preserved.

## Public evidence and production selection

The owner subsequently selected this exact candidate and reopened the production
API. Publication includes source/binary hashes, all three short passes, both
heavy passes, cold-prefill samples, five-depth speculation counters and sanitized
JSON/image/memory evidence. Raw prompts, generated private content, operator
addresses, credentials and job databases are excluded.

The combined flyer uses whole TP4 short pass2 and heavy pass2. Its TP3 panel
uses native1.2.9 short pass2 and explicitly labeled historical1.2.5/K2a heavy
full-concurrency samples. Those historical heavy values are not a new1.3.0 run.
Current TP3 qualification remains77/78 and is preserved separately.
