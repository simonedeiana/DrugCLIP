# ONNX and TensorRT inference

For step-by-step Linux setup, see [fresh deployment](docs/linux-fresh-install.md)
or [migration from an existing installation](docs/linux-migration.md).

This fork replaces the default inference launchers with a portable LMDB-to-ranking
pipeline. It does not require PyTorch or Uni-Core at inference time. Training and
HomoAug remain upstream code. Original launchers are preserved as
`test-pytorch.sh` and `retrieval-pytorch.sh`.

## Setup (Windows or Linux, Python 3.10)

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements-inference.txt
```

For NVIDIA GPU inference, install `requirements-tensorrt.txt` instead. The tested
configuration is TensorRT 10.13.3.9, CUDA 12.9 and an RTX 3050 8 GB. CUDA runtime
libraries and a compatible NVIDIA driver must be available on the library path.
ONNX uses the CPU execution provider with six threads by default.

## Models

Use `molecule.onnx` and `pocket.onnx` in a `models/` directory. Models are kept out
of Git because each exceeds GitHub's 100 MB file limit. See the repository's
Releases for exported models when available, or export the original checkpoint:

```sh
# Run from the repository root in the original PyTorch 1.11/Uni-Core environment.
# Also install onnx==1.17.0 and the LMDB/NumPy/SciPy preprocessing dependencies.
python export_onnx.py --checkpoint checkpoint_best.pt --output models
```

The exporter strictly loads the checkpoint and emits two FP32 ONNX opset 14
graphs with dynamic batch/sequence axes. It includes each encoder, projection
and L2 normalization. Chemistry preprocessing and similarity scoring are outside
the graph. Model dictionaries correspond to the official checkpoint.

Build TensorRT engines **on the destination OS/GPU**:

```sh
python -m drugclip_runtime.build --models models --output models
```

The builder uses strict FP32, with TF32 disabled. FP16/INT8 are intentionally not
enabled: previous forced low-precision trials failed accuracy validation. Default
profiles support molecule batches 1–8 and up to 128 padded tokens, and pocket
batches of 1 with up to 520 tokens. Increase `--max-molecule-tokens` when building
for larger molecules. Unsupported shapes raise an error; inputs are not truncated.

## Screen and retrieve

```sh
python -m drugclip_runtime --backend onnx --models models --molecules data/DUD-E/raw/all/ampc/mols.lmdb --pockets data/DUD-E/raw/all/ampc/pocket.lmdb --output results/ampc-onnx
python -m drugclip_runtime --backend tensorrt --models models --molecules data/DUD-E/raw/all/ampc/mols.lmdb --pockets data/DUD-E/raw/all/ampc/pocket.lmdb --output results/ampc-trt
```

On Linux, `bash test.sh` and `bash retrieval.sh` accept these same arguments.
Both produce normalized embeddings, `scores.npy`, `rankings.csv`, and
`metrics.json`. Labels are optional for retrieval; binary labels enable ROC-AUC
and top-1% active counts. Multiple pocket conformations are combined by maximum
similarity per molecule. Each call recomputes embeddings; legacy embedding-cache
reuse is not implemented in this first portable version.

LMDB records must have `atoms`, `coordinates`, `smi` for molecules, and
`pocket_atoms`, `pocket_coordinates` for pockets. Load only trusted LMDBs because
the upstream format uses pickle (and may embed RDKit objects).

Hydrogens are removed, coordinates centered, tokens surrounded with CLS/SEP,
distance special-token rows filled with zeros, and batches padded to multiples
of eight. LMDB traversal follows Uni-Core's lexical key order.

`--conformer 0` explicitly chooses the first conformer; use another index to
choose another. Pocket cropping above `--max-pocket-atoms 511` uses a stable RNG
controlled by `--seed`. These are deliberately reproducible choices instead of
upstream's process-dependent `hash((seed, None, index))` randomness. Exact
preprocessing equality was verified on AMPC, where those choices do not alter
the inputs; equivalence for other conformer/cropping cases is not claimed.

## Validation

```sh
python -m unittest discover -s tests
```

See `VALIDATION.md` for measured results. Validation covers the available AMPC
dataset, not all DUD-E/PCBA targets, training, or HomoAug. Training datasets and
those additional benchmarks were not downloaded or executed.

## Attribution and license

Based on Gao et al., DrugCLIP (NeurIPS 2023), upstream commit
`7a3a3fa33673f8668c811790f2e4681c98af44ef`.
See [LICENSE](LICENSE): code is Apache-2.0; weights and model outputs are
CC BY-NC 4.0. Exporting the weights does not remove their noncommercial terms.
