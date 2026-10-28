# Benchmark results

Start with the [plain-language benchmark history](../docs/benchmark-history.md).

**New:** [clean receptor + molecule benchmark](results/2026-10-28-clean/REPORT.md),
with preprocessing and validation outside the timer, plus
[reproduction scripts](clean/README.md).

All full runs used the official checkpoint, batch size 8, first conformer,
seed 1 and maximum 511 pocket atoms. Data came from the authors'
[THU-ATOM/DrugCLIP_data](https://huggingface.co/datasets/THU-ATOM/DrugCLIP_data).
Archives were SHA256-verified. DUD-E uses `pocket.lmdb`; LIT-PCBA uses
`pockets.lmdb`. The counts are molecule records, not necessarily unique compounds
across targets. Every backend completed 117 targets and 4,241,148 records.

See the [molecule versus receptor breakdown](../docs/encoder-performance.md)
for separate encoder timings and the important pocket-validation overhead.

## Published evidence

Files in [results/2026-10-23](results/2026-10-23):

| File | Contents |
|---|---|
| windows-tensorrt.json | All 117 baseline target metrics, timings and sampled ONNX errors |
| wsl-pytorch.json | All 117 original-model FP16 target metrics and timings |
| windows-cuda-graphs.json | All 117 graph target metrics, timings, cache counters and baseline agreement |
| backend-comparisons.json | Per-target FP32/FP16 embedding, score, ranking and screening metric differences |
| encoder-timings.json | Separate molecule/pocket stage totals for all three deployments |
| agreement-summary.json | Worst-case full TensorRT/PyTorch comparison and ONNX sample coverage |
| full-run-comparison.json | CUDA Graph aggregate timing and exact output agreement |
| prepared-input-results.json | All six alternating timing samples per encoder, plus cold capture time |
| eviction-test.json | Shape cache reuse/eviction validation |
| dataset-provenance.json | Dataset acquisition and archive verification record |
| earlier-ampc-tests.md | Historical 20-check report; its AMPC-only scope predates the full runs |
| full-backend-report.md | Full dataset accuracy summary and methodology |
| cuda-graph-report.md | CUDA Graph detailed report |

The repository contains complete **per-target results**, not multi-gigabyte raw
embeddings, scores or redistributed datasets. Raw arrays were retained locally
and compared in full. No full CPU ONNX dataset run was performed. No fallback
to CPU occurred in either full TensorRT run.

## Interpreting timing and correctness

Full timings sum each target's preprocessing, inference and scoring time. They
exclude model initialization, output writes and cross-backend saved-array
validation. Graph capture and sampled CPU ONNX checks are included in both
TensorRT configurations where applicable. The full runs were single sequential
runs, not repeated randomized experiments. PyTorch ran under WSL and TensorRT
ran on Windows, so this is not an isolated backend-only comparison.

`PASS` means inference and the configured checks succeeded, not that screening
accuracy exceeds a scientific threshold. Means are unweighted across targets.
Graph checks used atol=rtol=1e-6; observed maximum errors were exactly zero.
PyTorch FP16 agreement is reported quantitatively, not called bitwise equal.

## Repeating the graph microbenchmark

Use the inference requirements plus TensorRT, locally built engines, and trusted
prepared NPZ batches containing `tokens`, `distances`, and `edge_types`:

```bash
python benchmarks/benchmark_cuda_graphs.py --models models/tensorrt \
  --prepared /absolute/path/to/prepared-batches --output results/graph-microbenchmark.json
```

Prepared filenames follow `molecule-*.npz` and `pocket-*.npz`; retain the original
batch order and preprocessing. The script checks outputs before timing, then
alternates six ordinary/graph passes. It does not download data or export models.
For LMDB screening and graph/non-graph agreement, follow the deployment guides.
