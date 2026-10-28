# Fresh Linux deployment

This guide deploys the portable **inference** runtime: trusted LMDB files →
embeddings → ranked molecules. It does not install the original training or
HomoAug stack. For an existing installation, use the
[migration guide](linux-migration.md).

Commands below target **Ubuntu 22.04 x86-64, Bash and Python 3.10**. Other Linux
distributions need equivalent OS packages and a Python 3.10 environment; the
pinned wheels are not a claim of support for every architecture or Python
version. Validation was performed under Ubuntu 22.04 in WSL, not on a newly
provisioned bare-metal Linux host. See [validation scope](../VALIDATION.md).

## 1. Install prerequisites and clone

ONNX Runtime CPU needs no NVIDIA GPU or CUDA. Start with that backend even if
you intend to add TensorRT.

```bash
sudo apt-get update
sudo apt-get install -y git curl unzip ca-certificates python3.10 python3.10-venv

mkdir -p "$HOME/apps"
cd "$HOME/apps"
git clone https://github.com/simonedeiana/DrugCLIP.git DrugCLIP-portable
cd DrugCLIP-portable
# Current main includes CUDA Graphs; record the exact deployed revision.
git switch main
git rev-parse HEAD > deployed-commit.txt
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-inference.txt
python -m pip check
python -m unittest discover -s tests
```

Run subsequent commands from this checkout with `.venv` activated. CUDA Graph support is on `main`, after v0.1.0. Record the deployed commit for
reproducibility. The model archive remains v0.1.0: weights are unchanged.
Checking out the v0.1.0 code tag omits CUDA Graph support.

## 2. Download and verify the models

The release contains both FP32 ONNX encoders, their metadata, license, and
checksums. The archive is approximately 228 MiB; allow several GiB of free disk
space for environments, downloads, models and optional TensorRT engines.

```bash
mkdir -p downloads models/onnx
curl --fail --location --retry 3 \
  https://github.com/simonedeiana/DrugCLIP/releases/download/v0.1.0/drugclip-fork-models.zip \
  --output downloads/drugclip-fork-models.zip
printf '%s\n' 'ff00e8cad96b13cc66f1bb2191d894bb4bcae97972066907d094440e910029c9  downloads/drugclip-fork-models.zip' | sha256sum --check -
```

Only continue if the archive checksum reports **OK**:

```bash
unzip downloads/drugclip-fork-models.zip -d models/onnx
(cd models/onnx && sha256sum --check SHA256SUMS)
```

Both model checksums must also report **OK**. Code is Apache-2.0; weights and
model outputs retain the upstream **CC BY-NC 4.0** terms. See [LICENSE](../LICENSE).

## 3. Supply your data and run CPU inference

Models do not include datasets. Obtain the upstream data from the
[official data folder](https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV?usp=sharing),
or use your existing compatible files. This runtime starts from prepared LMDB;
it does not generate molecular conformers or extract pockets from raw structures.
Load only trusted LMDBs: the format uses Python pickle and can contain RDKit
objects. `rdkit-pypi` is included in the requirements for those objects.

Set these paths to real files on your machine:

```bash
MOLS=/absolute/path/to/ampc/mols.lmdb
POCKETS=/absolute/path/to/ampc/pocket.lmdb
test -f "$MOLS" && test -f "$POCKETS"
```

After confirming both files exist:

```bash
python -m drugclip_runtime \
  --backend onnx --models models/onnx \
  --molecules "$MOLS" --pockets "$POCKETS" \
  --batch-size 8 --conformer 0 --seed 1 --max-pocket-atoms 511 \
  --output results/ampc-onnx
cat results/ampc-onnx/metrics.json
```

Outputs are `rankings.csv`, `scores.npy`, `molecule-embeddings.npy`,
`pocket-embeddings.npy`, and `metrics.json`. Unlabelled molecule data works for
retrieval; binary 0/1 labels enable AUC and top-1% active counts. For the AMPC
files used in validation, expect 2,898 molecules, one pocket, AUC about
**0.771345**, and **16 actives among the top 29**. Those values are not expected
for different datasets. See [the input contract](../PORTABLE.md#screen-and-retrieve).

## 4. Add NVIDIA TensorRT (optional)

The tested stack is TensorRT **10.13.3.9**, CUDA **12.9**, Python **3.10**, and
RTX 3050 8 GB. Install a compatible NVIDIA driver and CUDA 12 runtime/toolkit
using the instructions for your distribution. Do not replace a working driver
just to run the CPU backend. TensorRT 10.13.3 lists a Linux driver floor of
r535 for CUDA 12.x; your selected toolkit may impose a higher requirement.
See the [version-specific NVIDIA release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.3.html)
and [CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
For WSL, the GPU driver belongs on the Windows host; follow NVIDIA's WSL setup.

```bash
nvidia-smi
python -m pip install -r requirements-tensorrt.txt
python -m pip check
```

If CUDA 12.9 is installed at `/usr/local/cuda-12.9`, expose its libraries in
this shell (adjust the path if yours differs):

```bash
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python - <<'PY'
import tensorrt as trt
from cuda.bindings import runtime as cuda
print('TensorRT:', trt.__version__)
error, count = cuda.cudaGetDeviceCount()
assert int(error) == 0 and count > 0, (error, count)
logger = trt.Logger(trt.Logger.WARNING)
assert trt.Builder(logger) is not None
print('CUDA devices:', count)
PY
```

`nvidia-smi` alone does not prove that user-space CUDA libraries are installed;
its displayed CUDA version describes driver capability. Resolve any failed
preflight before building engines. Use the pinned `tensorrt-cu12` requirements,
not an unpinned `pip install tensorrt` that may select a different CUDA major.

Build engines locally for this OS, GPU and TensorRT installation:

```bash
python -m drugclip_runtime.build --models models/onnx --output models/tensorrt
python -m drugclip_runtime \
  --backend tensorrt --models models/tensorrt \
  --molecules "$MOLS" --pockets "$POCKETS" \
  --batch-size 8 --conformer 0 --seed 1 --max-pocket-atoms 511 \
  --output results/ampc-tensorrt
```

Build time varies by hardware. The builder uses FP32 with TF32 disabled; do not
substitute the failed experimental FP16/INT8 engines. Rebuild engines when
moving machines/platforms or changing TensorRT. Do not copy Windows engines
to Linux.

## 5. Check backend agreement

Run both backends with identical data and preprocessing options, then:

```bash
python - <<'PY'
import numpy as np
a = np.load('results/ampc-onnx/scores.npy')
b = np.load('results/ampc-tensorrt/scores.npy')
assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
print('Maximum score difference:', float(np.max(np.abs(a-b))))
k = max(1, int(np.ceil(len(a) * .01)))
print('Top-1% overlap:', len(set(np.argsort(a)[-k:]) & set(np.argsort(b)[-k:])), '/', k)
np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)
PY
```

The tolerance is a smoke-test threshold, not a universal scientific validation
criterion. Check task-specific rankings and metrics on your own targets.
The validated AMPC run had maximum ONNX/TensorRT score error around 1.1e-6.
Use fresh output directories for separate runs because files are overwritten.

## 6. Optional CUDA Graphs

Add `--cuda-graphs` to the TensorRT command, using the same engines and input
options, and write to `results/ampc-cuda-graphs`. No additional dependencies or
engine rebuild are needed solely for this option. Only molecule inference uses
graphs. The cache holds up to 16 shapes and uses extra GPU memory; capturing
new shapes adds startup cost. Short jobs may not benefit.

```bash
python -m drugclip_runtime --backend tensorrt --cuda-graphs --models models/tensorrt --molecules "$MOLS" --pockets "$POCKETS" --batch-size 8 --conformer 0 --seed 1 --max-pocket-atoms 511 --output results/ampc-cuda-graphs
python - <<'PY'
import numpy as np
for name in ['molecule-embeddings.npy', 'pocket-embeddings.npy', 'scores.npy']:
    a = np.load('results/ampc-tensorrt/' + name)
    b = np.load('results/ampc-cuda-graphs/' + name)
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
    print(name, float(np.max(np.abs(a-b))))
PY
```

The full native Windows graph run matched ordinary TensorRT exactly across
117 targets / 4.24 million records, with 2.7% lower measured full-run time.
The full graph suite has not been run on Linux: validate agreement and speed
on your own host. Capture errors are explicit, not silent fallbacks. If capture
fails, memory is insufficient or performance regresses, omit `--cuda-graphs`.
See the [benchmark history](benchmark-history.md) and
[complete results](../benchmarks/README.md).

## Troubleshooting and repeat runs

| Symptom | Action |
|---|---|
| No matching dependency wheel | Confirm x86-64 and Python 3.10; do not silently change pinned versions. |
| `libcudart.so` missing / CUDA preflight fails | Check CUDA 12 installation and `LD_LIBRARY_PATH`; CPU ONNX remains usable. |
| Engine cannot deserialize | Rebuild from ONNX on this host with the pinned TensorRT version. |
| Unsupported molecule shape | Default engine accepts batch sizes 1–8 and up to 128 padded tokens. Rebuild with `--max-molecule-tokens 256` (or an adequate larger value); longer inputs require more memory. |
| GPU out of memory | Try a smaller inference batch. If engine building fails, free GPU memory or use CPU; a smaller inference batch does not reduce the builder's fixed maximum batch profile. |
| Missing labels | Retrieval works without labels, but AUC is omitted. |
| Different result on multi-conformer/cropped data | Review `--conformer`, `--seed`, and `--max-pocket-atoms`; see migration notes. |

For each later shell, `cd` into the checkout, activate `.venv`, set CUDA library
paths if needed, and run the same command. For scheduled jobs, use absolute
paths to the checkout, `.venv/bin/python`, data, models and output; set the
working directory to the checkout. No web service is installed by these steps.
