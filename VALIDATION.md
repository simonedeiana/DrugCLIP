# Validation history

Updated 2026-10-28. Read the [benchmark history](docs/benchmark-history.md)
and [complete per-target results](benchmarks/README.md). The sections below
record earlier AMPC release checks; their scope statements apply to that stage.

## Initial validation on 2026-10-23

Machine: NVIDIA RTX 3050 8 GB; native Windows and Ubuntu 22.04 under WSL.
Source: bowen-gao/DrugCLIP `7a3a3fa33673f8668c811790f2e4681c98af44ef`.
Data: DUD-E AMPC, 2,898 molecules (48 actives), one pocket; batch size 8.

## End-to-end portable pipeline

| Backend | ROC-AUC | Actives in top 29 | Seconds including model startup |
|---|---:|---:|---:|
| Windows ONNX Runtime CPU | 0.771345029 | 16 | 46.16 |
| Windows TensorRT FP32 | 0.771345029 | 16 | 5.76 |
| WSL TensorRT FP32 | 0.771345029 | 16 | 12.86 |

These are single functional runs, **not controlled performance comparisons**.
They include LMDB reads, preprocessing, model startup, inference and scoring.
WSL loads files from the Windows-mounted filesystem; startup/I/O costs differ.

The independently rerun exporter produced models that also passed the full
AMPC ONNX run (AUC 0.771345029, 16/29 actives). Maximum score differences:

- New Windows TensorRT pipeline vs earlier prepared-input Windows run: 0.
- Windows ONNX vs Windows TensorRT: 1.073e-6.
- WSL TensorRT vs Windows TensorRT: 3.875e-7.
- All backends selected the same top-29 molecule set.

Every tensor in all 364 batches (363 molecule, one pocket) exactly matches the
inputs saved from the original PyTorch preprocessing. Reproduce this check with:

```sh
python verify_preprocessing.py --target path/to/ampc --prepared path/to/upstream-batches
```

The original WSL PyTorch FP16 run gave ROC-AUC 0.771286550; the small difference
from FP32 does not change the top-29 active count. This does not establish
equivalence on other targets, multi-conformer selections or cropped pockets.

## Earlier matched prepared-input timing

The same inputs resident in host RAM, same TensorRT version, FP32 with TF32
disabled, three runs, mean of final two:

| Platform | Seconds | Molecules/second |
|---|---:|---:|
| Windows TensorRT | 3.2213 | 900 |
| WSL TensorRT | 2.9389 | 986 |

Includes host/device transfers, both encoders and scoring; excludes startup,
filesystem reads and preprocessing. WSL was about 9% faster in this comparison.

## Scope

Three preprocessing unit tests passed (hydrogen removal/special-token distances,
explicit conformer selection/input rejection, deterministic pocket cropping).
The fresh ONNX exporter passed the ONNX graph checker and full AMPC inference.
TensorRT engine building was exercised on this Windows GPU.
Both engines were then rebuilt from the freshly exported ONNX models and the
full AMPC TensorRT run passed again (AUC 0.771345029, 16/29 actives, 5.62 seconds
including model startup). The three unit tests also passed under WSL.

Training, HomoAug, full DUD-E and PCBA were not run. No training or model quality
improvement is claimed. FP16/INT8 variants are not shipped because earlier
forced-precision experiments failed correctness checks.


## Completed full-dataset validation

All 117 targets / 4,241,148 molecule records passed in WSL PyTorch FP16,
Windows TensorRT FP32 and Windows TensorRT with molecule CUDA Graph replay.
No CPU fallback occurred. Timed sections summed to 10,381.374, 7,031.024 and
6,839.528 seconds respectively. TensorRT timings include sampled ONNX checks;
startup, saving and saved-baseline comparisons are excluded. Each full run
was performed once. Hardware, OS, precision and I/O differences limit causal
speedup claims between PyTorch and TensorRT.

Every graph embedding and score matched ordinary TensorRT exactly. PyTorch
was close but not identical: minimum ranking correlation 0.999970972,
minimum top-1% overlap 97.14%, maximum AUC difference 0.001935747, and changed
top-1% active counts on three targets. Labels and counts matched. The sampled
ONNX check covered 9,483 embeddings; maximum difference was 1.1064e-6.

Six alternating graph microbenchmark runs reduced warm molecule time from
3.150046 to 2.911473 seconds (7.57%). All 363 molecule batches and the pocket
matched bitwise. An 18-batch eviction/recreation test passed. The full graph
run was on Windows, not Linux. Model weights and preprocessing were unchanged.


Separate [molecule and receptor/pocket timings](docs/encoder-performance.md)
are available. Full-run pocket timers include CPU ONNX checks for TensorRT,
so they are not an isolated comparison of GPU pocket encoder speed.


A [clean subset benchmark](benchmarks/results/2026-10-28-clean/REPORT.md)
now reports both encoders without timed preprocessing or validation. Graph
outputs remained exact; FP16 differences and per-target rankings are quantified.
