# Full DrugCLIP inference benchmark

Dataset source: [THU-ATOM/DrugCLIP_data](https://huggingface.co/datasets/THU-ATOM/DrugCLIP_data). Archives verified against the publisher's SHA256 hashes. Per-target results are published alongside this report; raw arrays remain in the benchmark archive.

Official original checkpoint exported for ONNX/TensorRT. Both full runs use portable preprocessing: first conformer, seed 1, max 511 pocket atoms, batch size 8. This compares inference backends on identical inputs; it is not a claim to reproduce published paper numbers or every alternative pocket preparation.

| Backend | Suite | Passed targets | Molecule records | Mean target AUC | Mean BEDROC(80.5) | Mean EF1% |
|---|---|---:|---:|---:|---:|---:|
| windows-tensorrt | DUD-E | 102/102 | 1,433,536 | 0.806881 | 0.500158 | 31.969813 |
| windows-tensorrt | LIT-PCBA | 15/15 | 2,807,612 | 0.571792 | 0.062208 | 5.508924 |
| wsl-pytorch | DUD-E | 102/102 | 1,433,536 | 0.806856 | 0.500186 | 31.978131 |
| wsl-pytorch | LIT-PCBA | 15/15 | 2,807,612 | 0.571750 | 0.062239 | 5.508924 |

## Timing and coverage

- TensorRT uses strict FP32 with TF32 disabled. Its total time includes sampled CPU ONNX agreement checks; do not treat it as a pure GPU kernel benchmark.
- PyTorch uses the original FP16 model in WSL with the same portable input tensors. Its timing includes preprocessing and WSL data reads.
- Each target is scored once per full backend. These are dataset-wide functional/evaluation runs, not repeated throughput measurements.
- Oversized molecule batches that exceed the existing TensorRT profile use explicitly counted ONNX CPU fallback; those runs are hybrid and are not presented as pure TensorRT.
- ONNX CPU validation samples first/last and every 512th molecule batch, plus every pocket. Full CPU inference is not requested.
- Means above are unweighted means across completed targets, not pooled cross-target AUC.

## Status

- windows-tensorrt: 117 targets attempted; 117 passed.
- wsl-pytorch: 117 targets attempted; 117 passed.

Cross-backend score comparisons available for 117 targets.

## Completed cross-backend validation

All saved molecule and pocket embeddings were compared in addition to scores. Labels and record counts match across all targets. Differences below compare FP32 TensorRT with FP16 PyTorch; they do not imply bitwise equivalence.

```json
{
  "targets": 117,
  "labels_and_counts_match": true,
  "max_score_error": 0.007515370845794678,
  "min_spearman": 0.99997097162661,
  "min_top1_overlap_fraction": 0.9714285714285714,
  "max_abs_auc_delta": 0.0019357469974616848,
  "max_abs_bedroc_delta": 0.0018962158721528288,
  "max_abs_ef1_delta": 0.4441568264907332,
  "targets_with_changed_top1_active_count": 3,
  "max_molecule_embedding_error": 0.0047495365142822266,
  "max_pocket_embedding_error": 0.0018877387046813965,
  "sampled_onnx_embeddings": 9483,
  "max_onnx_error": 1.1064112186431885e-06,
  "cpu_fallback_molecules": 0,
  "tensorrt_seconds": 7031.024006199994,
  "pytorch_seconds": 10381.37354949
}
```
