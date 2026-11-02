# TensorFold TP4 4.0.3 — qualification and live operation

4.0.3 is a stability/memory/vision update to 4.0.2. The following short
screen was completed at deployment; this publication runs no new GPU test.
The older [4.0.2 heavy and cold-prefill results](BENCHMARK-4.0.2.md)
remain measurements of 4.0.2, not newly measured 4.0.3 scores.

## Short decode screen

Same authored code/prose fixtures, two passes, pass 2 shown. C1/C3/C6/C12/C16,
thinking OFF, greedy, temperature 0, seed 42, 512 output tokens per request.
The baseline is the existing historical 4.0.2 run, not a new paired run.

| Concurrency | Task | 4.0.2 median/request tok/s | 4.0.3 median/request tok/s | 4.0.2 wall aggregate tok/s | 4.0.3 wall aggregate tok/s | Aggregate change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C1 | code | 112.9 | 115.7 | 112.7 | 115.2 | +2.27% |
| C1 | prose | 88.1 | 86.3 | 87.9 | 86.2 | -1.91% |
| C3 | code | 61.7 | 63.1 | 182.4 | 181.0 | -0.81% |
| C3 | prose | 50.0 | 51.1 | 147.3 | 151.4 | +2.79% |
| C6 | code | 39.6 | 37.7 | 222.7 | 215.4 | -3.31% |
| C6 | prose | 33.1 | 33.4 | 191.3 | 193.3 | +1.05% |
| C12 | code | 23.7 | 24.6 | 263.8 | 266.1 | +0.85% |
| C12 | prose | 21.4 | 21.2 | 241.1 | 242.7 | +0.65% |
| C16 | code | 19.6 | 19.5 | 288.6 | 289.0 | +0.14% |
| C16 | prose | 17.3 | 17.5 | 263.9 | 267.7 | +1.45% |

Per-request median is the median of native request decode rates. Wall aggregate
is total completion tokens divided by the whole cell duration, including
dispatch/small prefills/tails. It is **not** a median of full-concurrency time
slices or the sum of native rates. Aggregate changes range from -3.31% to
+2.79%; request medians from -4.66% to +3.85%. This screen does not isolate
a causal controller speedup or establish day-long performance.

## Observed production operation after deployment

2 November 2026, 13:31:54–22:52:51 CET (about 9h21m). Mixed autonomous and
interactive agent work; this is an observational log audit, not a matched
benchmark. All four ranks retained the deployed binary hashes and PIDs.

| Metric | Observed value |
| --- | ---: |
| Ended native requests | 5,266 |
| Input / newly computed tokens | 533,618,285 / 22,501,955 |
| Token-weighted prefix cache hit | 95.78% |
| Speculative acceptance | 79.77% |
| Graph operation reuse | 99.944% |
| Resident graph groups | 338 |
| Graph statistics writes / failures | 529 / 0 |
| Production image jobs completed | 33/33 |
| Image total latency, median / maximum | 239.8 / 662.5 seconds |
| Model restarts / CUDA or OOM errors found | 0 / 0 |
| Minimum available RAM sampled by controller | 3.46 GiB |

Vision completion was checked for nonempty text or valid tool calls, not
just HTTP status. No nonterminal image job remained at the final check.
The image latency includes queueing and concurrency. This does not score
semantic image-analysis quality or guarantee every future request succeeds.

CUDA inventory stabilized: no new capture for the final ~107 minutes.
The logged cumulative CPU control time was 3.90 seconds, not an A/B GPU
overhead measurement. Pressure eviction did not trigger; real pressure
recovery therefore was not exercised by this observation window.

All four model processes had zero swap at inspection. Existing historical
host diagnostics on three nodes recorded only 64/64/128 KiB of new swap-out
over roughly 9.5 hours (64 KiB OS pages); the fourth node lacks that history.
Old system swap usage must not be mistaken for new model paging.

Finite image retention removed the bodies of 456 eligible historical jobs.
Archived image BLOB bytes were zero. The SQLite file retained reusable free
pages; no blocking VACUUM was run.

## Remaining compatibility limitation

304 native requests returned HTTP 400; 303 explicitly report unsupported
`response_format`/`guided_*` structured output. The saved earlier 4.0.2
window already contained 458 errors with that reason. These errors reach
callers; no same-caller successful fallback was established. They are not
a new graph-controller regression. Fourteen client disconnects were also
recorded; their causes were not established. No server HTTP 5xx appeared
in the captured native chat logs.

## Evidence and offline verification

[Numeric exports](../results/20261102-4.0.3/) include both short passes,
comparison, sanitized aggregate live observations, exact production binary
and source provenance, and hashes. Operator addresses, credentials, raw
journals, prompts, generated outputs, image bodies and job IDs are omitted.
Run `python3 tools/audit_403.py` to recompute the exported metrics and verify
the source/payload/gateway binding offline. [Installation](DS41F-4.0.3.md).
