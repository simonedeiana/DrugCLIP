# DrugCLIP benchmark history: what improved and what stayed the same

Published 28 October 2026. Measurements were performed on 23 October 2026.

## What this project achieved

DrugCLIP ranks molecules by their similarity to a protein binding pocket. We kept
the official trained model and made its inference available through ONNX Runtime
and NVIDIA TensorRT, including native Windows. We then added optional CUDA Graphs,
which reduce the overhead of repeatedly launching GPU operations.

The final CUDA Graph run processed **4,241,148 molecule records across 117 targets**.
Every saved embedding and score exactly matched ordinary TensorRT. This is an
execution-speed improvement; it does not train a better model or improve drug
discovery accuracy.

## The main result

| Full dataset run | Measured time | Change from the preceding row |
|---|---:|---:|
| Original model, PyTorch FP16 in WSL | 173.0 minutes | Starting comparison |
| TensorRT FP32, native Windows | 117.2 minutes | 32.3% less time |
| TensorRT FP32 + CUDA Graphs, native Windows | 114.0 minutes | A further 2.7% less time |

The final run took 34.1% less time than the PyTorch run (about 1.52 times the
throughput). CUDA Graphs alone saved 3.2 minutes. These are sums of per-target
timed sections, not total process wall time: they include preprocessing,
inference, and scoring but exclude startup and saving results. TensorRT timings
also include sampled ONNX checks; saved-baseline graph validation is outside
the timed section. Each full configuration ran once. Different operating
systems, I/O paths and precision mean the PyTorch/TensorRT difference is a
comparison of these deployments, not a pure measurement of TensorRT's speedup.

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

## How we got here

1. **Established the original model baseline.** The legacy PyTorch/Uni-Core
   stack ran under Ubuntu 22.04 in WSL. Initial validation used AMPC: 2,898
   molecules, 48 known actives, one pocket. A newer Linux environment hit a
   compatibility problem with the old PyTorch stack, so it was not forced.
2. **Exported the two encoders to ONNX.** One encoder represents molecules; the
   other represents pockets. Preprocessing, normalized embeddings and similarity
   scoring remain necessary around these models. On AMPC, all 364 prepared input
   batches exactly matched the original preprocessing.
3. **Built TensorRT engines on Windows and WSL.** Strict FP32 engines passed
   numerical checks. Forced FP16/INT8 experiments failed correctness checks and
   were excluded. The original PyTorch FP16 path remained the comparison baseline;
   its numerical behavior is not interchangeable with forced TensorRT FP16.
4. **Ran every locally available inference check.** Twenty checks passed across
   Windows and WSL: preprocessing, ONNX and TensorRT inference, reference vectors,
   dynamic shapes/padding, export/build checks and upstream retrieval caching.
   At this stage only AMPC was available; these were not full-dataset results.
5. **Expanded to both full benchmark datasets.** DUD-E supplied 102 targets and
   1,433,536 molecule records; LIT-PCBA supplied 15 targets and 2,807,612 records.
   All 117 targets passed in both full TensorRT and PyTorch runs. ONNX CPU checks
   sampled 9,483 embeddings, including every pocket. No CPU fallback was needed.
6. **Added and validated CUDA Graph replay.** Six alternating warm measurements
   reduced molecule inference time from 3.150 to 2.911 seconds (7.57%). That test
   used all 363 prepared AMPC molecule batches and included transfers but not
   preprocessing. A separate cache-eviction test passed. The subsequent full
   dataset run confirmed exact output agreement and the smaller 2.7% overall gain.

## Do the results still match?

**CUDA Graphs versus ordinary TensorRT: exactly.** All molecule and pocket
embeddings, labels and scores matched across all 117 targets. Maximum numerical
difference was zero. Rankings and screening metrics are therefore unchanged.

**TensorRT FP32 versus PyTorch FP16: very close, but not identical.** All record
counts and labels matched. The worst target's ranking correlation was 0.999971
(1 means identical ordering), and top-1% sets overlapped by at least 97.14%.
The largest AUC difference was 0.00194; the number of known actives in the top 1%
changed on three targets. Maximum score difference was 0.00752. These differences
must not be described as exact numerical equivalence.

**TensorRT versus sampled ONNX FP32:** maximum embedding difference was
0.00000111. Full CPU inference over all 4.24 million records was not run.

An embedding is the numeric representation produced by an encoder. A score
compares molecule and pocket embeddings. AUC measures separation of actives
from inactives; BEDROC and enrichment factors emphasize the highest-ranked
molecules. Higher values generally indicate better screening, but this work
aimed to preserve those values, not improve them.

| Dataset | Mean AUC, PyTorch | Mean AUC, TensorRT / CUDA Graphs |
|---|---:|---:|
| DUD-E | 0.806856 | 0.806881 |
| LIT-PCBA | 0.571750 | 0.571792 |

These are unweighted averages over targets. Passing a runtime validation test
does not mean that every target has strong predictive performance.

## Why the earlier timing numbers differ

Early AMPC functional runs included model loading and preprocessing. A later
20-check rerun measured 46.531 seconds for Windows ONNX CPU and 5.770 seconds for
Windows TensorRT. Those single runs are useful deployment checks, not controlled
performance estimates. Earlier runs were 46.16 and 5.76 seconds respectively.

Prepared-input tests removed disk reads and preprocessing: an early matched
Windows/WSL comparison was 3.2213 / 2.9389 seconds; the later rerun was
3.2124 / 2.9210 seconds. Both used the final two of three passes, with transfers,
both encoders and scoring included. The original PyTorch warm AMPC result was
7.6279 seconds **including preprocessing**, so dividing it by those TensorRT
times would overstate a directly comparable speedup.

The CUDA Graph microbenchmark measures the molecule encoder separately and
alternates six baseline/graph runs. Its 7.57% time reduction should not be
confused with the 2.7% gain for full screening. Graph capture itself also takes
time: its first molecule pass took 4.346 seconds. Short jobs may not benefit.

## Scope and practical use

Hardware: RTX 3050 8 GB. Native Windows TensorRT 10.13.3.9 with CUDA 12.9;
Ubuntu 22.04 WSL PyTorch 1.11 / CUDA 11.3. Full runs used batch size 8,
conformer 0, seed 1 and at most 511 pocket atoms. They selected the standard
pocket files, not every alternative pocket preparation. The shared portable
preprocessing makes backend comparison reproducible; it is not a claim to
reproduce every random conformer/cropping choice or the paper's published metrics.

CUDA Graphs are opt-in with `--cuda-graphs` on the TensorRT CLI. Only molecule
inference uses graphs, because pocket shapes are executed too infrequently to
justify capture cost. The cache keeps up to 16 shapes and requires additional
GPU memory. Models and precision are unchanged. The full graph run was native
Windows; Linux users should validate it on their own host. Training, HomoAug,
full ONNX CPU coverage and accepted reduced-precision TensorRT were not tested
as part of the full suite.

## Where to go next

- [Fresh Linux deployment](linux-fresh-install.md)
- [Migration from an older installation](linux-migration.md)
- [Complete per-target results and methodology](../benchmarks/README.md)
- [Detailed validation history](../VALIDATION.md)
