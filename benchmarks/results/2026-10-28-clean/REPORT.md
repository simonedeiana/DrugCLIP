# Clean molecule and receptor inference benchmark

Measured 28 October 2026 on NVIDIA RTX 3050 8 GB, driver 617.42.

## Dataset and measurement

4,434 molecule records and 9 receptor/pocket records: all 2,898 AMPC molecules and its pocket, plus 512 evenly spaced lexical-LMDB records each from GBA, IDH1 and VDR, and up to three evenly spaced pockets per target (3, 3 and 2). Sampling positions and prepared-input SHA256 hashes are in `manifest.json`. This deliberately small subset is not a full-dataset accuracy evaluation.

Inputs were prepared once (conformer 0, seed 1, max 511 pocket atoms), then loaded into host RAM before timing. Molecule batch size 8; pocket batch size 1. Each encoder/shape was warmed before measurement; CUDA Graph capture was excluded, with an assertion that no capture occurred during timing.

**Timed:** input transfers, encoder inference, output transfer to host and synchronization, including normal runtime allocation/dispatch overhead. **Not timed:** LMDB access, preprocessing, loading inputs/models, warmup/capture, numerical validation, scoring, ranking and writing results. These are host-to-host inference measurements, not CUDA-event kernel-only timings.

An initial 23-pocket attempt exceeded the 16-shape graph cache and was rejected by the no-capture assertion; it is excluded from results. The nine-pocket subset fits the cache. Six measurements per encoder. Each molecule measurement traverses the prepared set once; each pocket measurement averages ten traversals of all 9 pockets. TensorRT and CUDA Graph passes alternate order. PyTorch runs separately so backends never compete for the GPU. Standard deviations describe these six measurements, not a confidence interval over machines.

PyTorch uses the original model in FP16 with torch 1.11.0+cu113 under Ubuntu 26.04 WSL. The earlier Ubuntu 22.04 distro was no longer installed. `GLIBC_TUNABLES=glibc.rtld.execstack=2` was set for this process to load the legacy binaries; no system setting or model was changed. TensorRT 10.13.3.9 FP32 (TF32 disabled) and its graph path run natively on Windows with the same engines. Thus preprocessing/validation overhead is removed, but precision, OS and runtime differences remain.

## Separate encoder timings

| Encoder | Backend | Mean per subset pass | Std. dev. | Records/s |
|---|---|---:|---:|---:|
| molecule | pytorch | 9.038988 s | 0.595665 s | 490.5 |
| molecule | tensorrt | 5.009289 s | 0.021941 s | 885.2 |
| molecule | cuda-graphs | 4.617688 s | 0.011939 s | 960.2 |
| pocket | pytorch | 0.187528 s | 0.001651 s | 48.0 |
| pocket | tensorrt | 0.190411 s | 0.000745 s | 47.3 |
| pocket | cuda-graphs | 0.183343 s | 0.001536 s | 49.1 |

Molecule: TensorRT versus PyTorch time change -44.58%; CUDA Graphs versus ordinary TensorRT -7.82%; CUDA Graphs versus PyTorch -48.91%. Negative means less time.

Pocket: TensorRT versus PyTorch time change +1.54%; CUDA Graphs versus ordinary TensorRT -3.71%; CUDA Graphs versus PyTorch -2.23%. Negative means less time.

## Numerical differences

Reference is ordinary TensorRT FP32. Embedding differences are measured across every selected record after timing.

| Compared backend | Encoder | Max absolute error | Mean absolute error | Minimum cosine similarity |
|---|---|---:|---:|---:|
| pytorch | molecule | 0.00215472281 | 0.000169059598 | 0.999957845 |
| pytorch | pocket | 0.00119388476 | 0.000184168448 | 0.999988427 |
| cuda-graphs | molecule | 0 | 0 | 1.000000000 |
| cuda-graphs | pocket | 0 | 0 | 1.000000000 |

Predictions are maximum molecule/pocket cosine scores across each target's selected pockets. Scoring and ranking are outside the timer.

| Target | Compared backend | Max score difference | Mean score difference | Rank correlation | Top-1% overlap | Top-1% actives (compared/reference) |
|---|---|---:|---:|---:|---:|---:|
| ampc | pytorch | 0.00282115117 | 0.000462874043 | 0.999990612 | 29/29 | 16/16 |
| ampc | cuda-graphs | 0 | 0 | 1.000000000 | 29/29 | 16/16 |
| GBA | pytorch | 0.00205696002 | 0.000421457866 | 0.999960393 | 6/6 | 0/0 |
| GBA | cuda-graphs | 0 | 0 | 1.000000000 | 6/6 | 0/0 |
| IDH1 | pytorch | 0.00351069868 | 0.000918444362 | 0.999945998 | 6/6 | 0/0 |
| IDH1 | cuda-graphs | 0 | 0 | 1.000000000 | 6/6 | 0/0 |
| VDR | pytorch | 0.00202196836 | 0.000400031655 | 0.999981403 | 6/6 | 0/0 |
| VDR | cuda-graphs | 0 | 0 | 1.000000000 | 6/6 | 0/0 |

Complete per-pass timing samples, AUC values where both classes are present, numerical differences and runtime metadata are in the JSON files. AUC on the 512-record subsets is descriptive only. Exact graph equality is reported from observed values, not assumed. The graph experiment here enables replay for **both** encoders; the earlier full-screening run used it for molecules only. This warm benchmark excludes capture costs and does not imply that pocket graphs help one-off screening.
