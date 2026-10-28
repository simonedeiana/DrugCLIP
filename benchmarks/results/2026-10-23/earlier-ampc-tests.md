# Available inference checks — 2026-10-23

Scope: locally available DrugCLIP tests and data. AMPC is the only installed target: 2,898 molecules, 48 actives, one pocket. No additional datasets downloaded.

## Execution status

| Check | Result |
|---|---|
| windows-unit | PASS |
| windows-preprocessing | PASS |
| windows-end-to-end-onnx | PASS |
| windows-end-to-end-tensorrt | PASS |
| windows-release-vectors-onnx | PASS |
| windows-release-vectors-trt | PASS |
| wsl-unit | PASS |
| wsl-preprocessing | PASS |
| wsl-end-to-end-onnx | PASS |
| wsl-end-to-end-tensorrt | PASS |
| wsl-release-vectors-onnx | PASS |
| wsl-release-vectors-trt | PASS |
| windows-prepared-benchmark | PASS |
| wsl-prepared-benchmark | PASS |
| wsl-original-pytorch | PASS |
| windows-bundle-onnx-selftest | PASS |
| wsl-bundle-trt-selftest | PASS |
| wsl-bundle-onnx-validation | PASS |
| wsl-bundle-trt-validation | PASS |
| wsl-legacy-retrieval-and-cache | PASS |

## End-to-end inference

| Platform/backend | ROC-AUC | Top-29 actives | Seconds including startup |
|---|---:|---:|---:|
| windows-onnx | 0.771345029 | 16 | 46.531 |
| windows-tensorrt | 0.771345029 | 16 | 5.770 |
| wsl-onnx | 0.771345029 | 16 | 50.220 |
| wsl-tensorrt | 0.771345029 | 16 | 10.729 |

Single functional runs; these timings include preprocessing and model startup. WSL reads the Windows-mounted filesystem. They are not a controlled Windows-vs-Linux performance comparison.

## Repeated prepared-input TensorRT benchmarks

| Platform | Warm mean seconds | Molecules/second |
|---|---:|---:|
| windows | 3.2124 | 902.1 |
| wsl-comparison | 2.9210 | 992.1 |

Same prepared tensors in host RAM; transfers, encoder execution and scoring included. Model startup, filesystem loading and preprocessing excluded. Three passes; warm mean uses the last two.

## Original WSL PyTorch FP16

ROC-AUC 0.771286550; BEDROC(alpha=80.5) 0.495706071; warm mean 7.6279 seconds.
This timing includes upstream data iteration and preprocessing, so it is not directly comparable to the prepared-input TensorRT timing.

Enrichment factors: {"0.005": 52.324999999999996, "0.01": 33.310344827586206, "0.02": 17.69612068965517, "0.05": 9.160344827586206}

## Cross-backend agreement

```json
[
  {
    "backend": "windows-onnx",
    "max_score_error_vs_windows_onnx": 0.0,
    "top29_overlap": 29
  },
  {
    "backend": "windows-tensorrt",
    "max_score_error_vs_windows_onnx": 1.0728836059570312e-06,
    "top29_overlap": 29
  },
  {
    "backend": "wsl-onnx",
    "max_score_error_vs_windows_onnx": 0.0,
    "top29_overlap": 29
  },
  {
    "backend": "wsl-tensorrt",
    "max_score_error_vs_windows_onnx": 9.238719940185547e-07,
    "top29_overlap": 29
  }
]
```

## Not run

- Full DUD-E and PCBA: remaining datasets are absent. The upstream DUD-E target evaluator was run on every local target (AMPC only).
- Training and HomoAug: outside inference scope.
- Rejected experimental quantized engines: not supported inference backends; previous accuracy failures remain unresolved.

Raw command output is retained in the adjacent `.log` files; per-run scores and metrics are in backend subdirectories.
