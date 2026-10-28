# CUDA Graph benchmark

Native Windows, RTX 3050, existing TensorRT FP32 engines. No precision change. Warm timings include host/device copies and synchronization, but exclude preprocessing. Six alternating baseline/graph measurements; all 363 molecule batches and the pocket produced bitwise-identical embeddings.

| Encoder | Ordinary TensorRT mean (s) | CUDA Graph mean (s) | Time reduction | Cold graph pass (s) |
|---|---:|---:|---:|---:|
| molecule | 3.150046 | 2.911473 | 7.57% | 4.346038 |
| pocket | 0.014536 | 0.013942 | 4.09% | 0.213130 |

Cache eviction/recreation test: 18 batches, 10 captures, 8 hits and 8 evictions; every output bitwise identical.

Full-run configuration uses graph replay only for molecules. Pockets use ordinary TensorRT to avoid capture overhead for infrequent shapes. The cache is limited to 16 shapes, each with its own context and device buffers. Input copies use the same nonblocking stream as replay. No extra padding or changed batching.

## Full dataset run

117/117 targets attempted; 117 passed. The run exited successfully (exit code 0).
Every molecule embedding, pocket embedding, label and score is checked against the saved ordinary TensorRT baseline, at atol=rtol=1e-6. Both TensorRT runs include the same sampled ONNX checks. Validation against saved results is outside per-target timed inference; graph capture is included. Historical full-run timings are single runs, not controlled repeated measurements.

```json
{
  "targets": 117,
  "molecules": 4241148,
  "graph_seconds": 6839.52782300004,
  "baseline_seconds": 7031.024006199994,
  "pytorch_seconds": 10381.37354949,
  "max_embedding_error": 0.0,
  "max_score_error": 0.0,
  "all_scores_bitwise_equal": true,
  "cpu_fallback_molecules": 0
}
```

Source references: [NVIDIA TensorRT CUDA Graph guidance](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html#cuda-graphs).
