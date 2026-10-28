# 1,000 molecules and all 231 pockets: ten-pass inference benchmark

Measured 28 October 2026 on RTX 3050 8 GB, driver 617.42. All 231 standard pockets from 102 DUD-E and 15 LIT-PCBA targets are included. The 1,000 molecules are evenly spaced positions from AMPC in lexical LMDB order; no label-based selection. Exact indices and prepared-tensor hashes are in the manifest.

## Timing method

Preprocessing is completed before benchmarking (conformer 0, seed 1, max 511 pocket atoms). Inputs reside in host RAM. Molecules use batch size 8, pockets batch size 1. Each encoder is grouped by exact input shape, identically for all backends. A bounded graph cache keeps one shape at a time, covering every pocket without GPU-memory pressure.

For each group: three untimed warmups, then **10 timed passes**. Each record is inferred ten times in measured sections. Aggregate sample j is the sum of pass j across all shape groups; these are grouped-pass totals, not ten uninterrupted end-to-end sweeps. TensorRT and graphs alternate order each pass. PyTorch runs separately; no simultaneous GPU benchmark jobs.

**Included:** host-to-device transfers, encoder execution, device-to-host transfers, synchronization and runtime dispatch/allocation. **Excluded:** disk I/O, preprocessing, model loading, warmups, graph capture, numerical checks, scoring/ranking and saving. Capture counters must remain unchanged inside measured passes. Every timed pass is checked afterward against its warmup outputs. This is warm host-to-host inference, not a GPU-kernel-only benchmark or cold-start measurement.

PyTorch is original-model FP16, torch 1.11.0+cu113, Ubuntu 26.04 WSL with process-local `GLIBC_TUNABLES=glibc.rtld.execstack=2` for legacy-library compatibility. TensorRT 10.13.3.9 is FP32 with TF32 disabled, native Windows; graph replay uses exactly those engines. OS and precision differences remain, so PyTorch comparisons describe these deployments. Ten repeats quantify short-term variation, not all sources of bias.

## All three backends, separated by encoder

| Workload | Backend | Mean | Median | Std. dev. | Min–max |
|---|---|---:|---:|---:|---:|
| 1000 molecule records | pytorch | 1.912337 s | 1.897244 s | 0.063321 s | 1.849631–2.051714 s |
| 1000 molecule records | tensorrt | 1.051991 s | 1.050225 s | 0.008086 s | 1.043073–1.071525 s |
| 1000 molecule records | cuda-graphs | 0.981801 s | 0.976299 s | 0.012662 s | 0.971483–1.009563 s |
| 231 pocket records | pytorch | 5.047920 s | 5.038574 s | 0.042260 s | 4.979569–5.131227 s |
| 231 pocket records | tensorrt | 5.432372 s | 5.432539 s | 0.018606 s | 5.406961–5.459557 s |
| 231 pocket records | cuda-graphs | 5.295009 s | 5.296650 s | 0.034341 s | 5.246449–5.342538 s |

## Three-way time reductions

| Comparison | Molecules | Pockets |
|---|---:|---:|
| tensorrt relative to pytorch | +44.99% | -7.62% |
| cuda-graphs relative to pytorch | +48.66% | -4.89% |
| cuda-graphs relative to tensorrt | +6.67% | +2.53% |

Positive percentages mean less elapsed time; negative means slower. Graph replay is enabled for both encoders here. Shape grouping and warmed graphs are intentional; do not extrapolate these times to a cold, mixed-shape screen.

## Prediction differences

All 231,000 molecule/pocket similarities are calculated after timing. Each target score takes the maximum over that target’s pockets. Because these 1,000 molecules come from AMPC, scores against other receptors are computational agreement checks, not labelled screening-accuracy tests. Top-1% means the top 10 of 1,000 molecules.

| Compared with TensorRT | Molecule embedding max error | Pocket embedding max error | Score max error | Score mean error | Worst target rank correlation | Minimum top-10 overlap |
|---|---:|---:|---:|---:|---:|---:|
| pytorch | 0.00162519142 | 0.0018877387 | 0.0039833039 | 0.000404364459 | 0.999954076 | 9/10 |
| cuda-graphs | 0 | 0 | 0 | 0 | 1.000000000 | 10/10 |

Raw ten-pass samples, per-shape samples, embedding mean errors/cosines, and per-target score/rank comparisons are in `results.json`. Input manifest and model hashes accompany the report. No timings were discarded. Graph equality is assessed from measured outputs. PyTorch FP16 is not assumed bitwise equal to FP32.
