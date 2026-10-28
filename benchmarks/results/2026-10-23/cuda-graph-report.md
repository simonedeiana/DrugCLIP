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


## Molecule versus receptor/pocket timings

DrugCLIP has separate molecule and pocket encoders. Here “receptor” means the
binding-pocket representation, not an entire protein. The full runs encoded
4,241,148 molecule records and 231 pocket records across 117 targets.

### Full dataset: preprocessing plus encoding

| Component | WSL PyTorch FP16 | Windows TensorRT FP32 | Windows TensorRT + molecule CUDA Graphs |
|---|---:|---:|---:|
| Molecules | 10366.141 s (172.77 min) | 6920.788 s (115.35 min) | 6727.219 s (112.12 min) |
| Receptors/pockets | 8.666 s | 105.371 s | 107.550 s |

For the molecule stage, the measured TensorRT deployment took
33.24% less time than the PyTorch deployment.
Graph replay reduced that stage by another
2.80% (193.57 seconds).
The graph configuration's molecule stage took
35.10% less time than PyTorch overall.

**These are not isolated GPU inference timings.** The original measurement
wraps LMDB reads, preprocessing, transfers, encoding and validation. Both
TensorRT runs additionally perform CPU ONNX agreement checks inside this timer:
sampled molecule batches and **every pocket**. PyTorch does not perform those
ONNX checks. Consequently, the larger TensorRT pocket totals do not establish
that its pocket encoder is slower. The CPU checking time was not recorded
separately, so it cannot be subtracted reliably after the fact. OS, I/O paths
and precision also differ between the PyTorch and TensorRT deployments.

**CUDA Graphs were enabled only for molecules in the full run.** Pocket
execution remained ordinary TensorRT. Its total changed from 105.371 to
107.550 seconds (2.07% higher) between single runs; this is not a pocket graph
optimization result. Summing these stages excludes the separately timed
scoring and other per-target overhead included in the overall totals.

### Warm encoder benchmark: ordinary TensorRT versus CUDA Graphs

These six alternating runs use prepared inputs, exclude preprocessing and
sampled ONNX checks, and include transfers and synchronization. Results are
means per full prepared molecule pass or per pocket inference.

| Encoder | Ordinary TensorRT | CUDA Graph replay | Time reduction |
|---|---:|---:|---:|
| Molecule: 363 batches / 2,898 records | 3.150046 s | 2.911473 s | 7.57% |
| Receptor/pocket: one record | 14.536 ms | 13.942 ms | 4.09% |

Both encoder outputs were bitwise identical. The cold graph pass took
4.346 seconds for molecules and 213.130 ms for the pocket, including capture.
Pocket captures cost much more than the approximately 0.594 ms saved per warm
inference on this example, which is why they were not enabled in full screening.
This pocket result covers one prepared shape, not all 231 pocket records.
No comparable isolated full-suite PyTorch encoder timing was collected.
