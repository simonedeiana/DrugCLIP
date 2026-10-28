# Performance by encoder

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



## Clean inference follow-up: 28 October 2026

A new limited-data benchmark removes preprocessing and validation from the
inference timer and separates molecules from receptors. It uses 4,434 molecule
records and nine pockets, with six measurements per encoder. Transfers and
synchronization remain included; graph capture, scoring and disk I/O are excluded.

| Encoder / workload | PyTorch FP16 (WSL) | TensorRT FP32 (Windows) | TensorRT + CUDA Graphs (Windows) |
|---|---:|---:|---:|
| 4,434 molecules | 9.039 s | 5.009 s | 4.618 s |
| Nine receptor pockets | 187.53 ms | 190.41 ms | 183.34 ms |

Graphs reduced molecule time by 7.82% relative to ordinary TensorRT and pocket
time by 3.71%. Against PyTorch, graph time was 48.91% lower for molecules but only
2.23% lower for pockets. Ordinary TensorRT was 1.54% slower on pockets in this
experiment. These small receptor differences are specific to this subset and
machine; they do not establish a general advantage. OS and precision differ
between PyTorch and TensorRT. The first PyTorch molecule measurement was slower
than subsequent passes; all six measurements are retained and reported with
standard deviations, rather than discarding the slower result.

Both encoders and all predictions matched ordinary TensorRT exactly with graphs.
PyTorch maximum embedding differences were 0.002155 (molecule) and 0.001194
(pocket); maximum screening-score difference was 0.003511. Ranking correlation
was at least 0.999946, and top-1% sets matched on all four targets. The sampled
LIT-PCBA targets are not full-target accuracy evaluations. This follow-up enables
graphs for both encoders; the full dataset run enabled them only for molecules.

See the [clean benchmark report](../benchmarks/results/2026-10-28-clean/REPORT.md)
for every timing sample, per-target prediction differences, exact selection
manifest, platform changes, and [reproduction scripts](../benchmarks/clean/README.md).
