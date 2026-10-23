# Linux migration from an existing DrugCLIP installation

Use a **separate checkout and environment**, then switch your inference job
after validating it. This preserves the old code, local modifications, trained
checkpoints, data and Python dependencies for rollback and training. These
instructions cover either original PyTorch/Uni-Core DrugCLIP or an older
portable installation.

## 1. Record the current installation

In the old checkout, with its original Python environment active:

```bash
cd /absolute/path/to/existing/DrugCLIP
OLD_REPO="$PWD"
MIGRATION_LOG="$HOME/drugclip-migration-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$MIGRATION_LOG"
git rev-parse HEAD > "$MIGRATION_LOG/old-commit.txt"
git status --short > "$MIGRATION_LOG/old-status.txt"
git diff HEAD --binary > "$MIGRATION_LOG/tracked-changes.patch"
python -m pip freeze > "$MIGRATION_LOG/old-python-packages.txt"
python -c 'import sys; print(sys.executable)' > "$MIGRATION_LOG/old-python.txt"
```

If an old environment lacks pip, use its package manager's environment export
instead. The patch records tracked changes only; it is **not a backup** of
untracked datasets, checkpoints or other files. Keep the old directory intact
and use your normal backup procedure for irreplaceable files. Record your
current launch command, scheduler working directory and any CUDA environment
settings. Do not run `git reset --hard`, clean the old tree, or install new
requirements into the old environment.

## 2. Install the portable version beside it

Open a **new shell** so the old virtualenv/Conda environment and `PYTHONPATH`
do not leak into the new setup. Follow
[fresh deployment steps 1–2](linux-fresh-install.md#1-install-prerequisites-and-clone)
to create a separate `DrugCLIP-portable` checkout, a Python 3.10 `.venv`, and
verified v0.1.0 ONNX models. Choose another new directory name if that path
already exists; do not clone over an existing checkout.

The exported release models correspond to the **official checkpoint**. If your
old installation uses custom/fine-tuned weights, replacing them with the release
models changes the model. Instead export your checkpoint using `export_onnx.py`
from this fork, executed from this fork's repository root with a compatible
legacy PyTorch/Uni-Core environment. Use a separate export environment when
dependencies need to be added. Preserve the original dictionaries and
architecture; this exporter/runtime assumes the official model configuration.
Custom architectures or vocabularies require adaptation and new validation.

## 3. Reuse data, rebuild engines, and rerun

Activate the **new** environment and use absolute paths to your existing LMDBs:

```bash
cd "$HOME/apps/DrugCLIP-portable"
source .venv/bin/activate
MOLS=/absolute/path/to/existing/DrugCLIP/data/DUD-E/raw/all/ampc/mols.lmdb
POCKETS=/absolute/path/to/existing/DrugCLIP/data/DUD-E/raw/all/ampc/pocket.lmdb

python -m drugclip_runtime \
  --backend onnx --models models/onnx \
  --molecules "$MOLS" --pockets "$POCKETS" \
  --batch-size 8 --conformer 0 --seed 1 --max-pocket-atoms 511 \
  --output results/migration-onnx
```

LMDB inputs are opened read-only. Quiesce any process writing those datasets
while screening them. For GPU inference, follow
[fresh deployment step 4](linux-fresh-install.md#4-add-nvidia-tensorrt-optional),
building into a new engine directory, then run:

```bash
python -m drugclip_runtime \
  --backend tensorrt --models models/tensorrt \
  --molecules "$MOLS" --pockets "$POCKETS" \
  --batch-size 8 --conformer 0 --seed 1 --max-pocket-atoms 511 \
  --output results/migration-tensorrt
```

Do not reuse old engine files after changing OS/GPU/TensorRT or model weights.
Recompute molecule and pocket embeddings together with the selected model;
the portable CLI does not consume the old `--emb-dir` cache.

## 4. Update the invocation and downstream consumers

| Original behavior / argument | Portable replacement |
|---|---|
| `python unimol/test.py` / `unimol/retrieval.py` | `python -m drugclip_runtime` |
| `--path checkpoint_best.pt` | `--models models/onnx` or `models/tensorrt` |
| `--mol-path`, `--pocket-path` | `--molecules`, `--pockets` |
| `--results-path` | `--output` |
| `--fp16`, Uni-Core task/model arguments | Omit them; select `--backend onnx` or `tensorrt`. Shipped TensorRT builder is strict FP32. |
| `--test-task DUDE/PCBA` traverses a benchmark | Invoke the portable CLI per target LMDB pair; full-suite orchestration and all legacy metrics are not implemented. |
| `ranked_compounds.txt` / cached embeddings | Consume `rankings.csv`, `scores.npy` and the two `*-embeddings.npy` files; update downstream paths and parsing. |
| `bash test.sh` / `bash retrieval.sh` | In the fork these now require the portable CLI arguments. Original launchers are `test-pytorch.sh` and `retrieval-pytorch.sh`. |

Training remains PyTorch/Uni-Core. Run it from the old checkout/environment or
use the preserved training code with its legacy dependencies; the new inference
virtualenv is not a replacement training environment.

## 5. Acceptance checks before switching jobs

1. Run the unit tests and confirm input counts, finite scores and normalized
   embeddings. Compare ONNX and TensorRT as in
   [fresh deployment step 5](linux-fresh-install.md#5-check-backend-agreement),
   substituting your `results/migration-*` directories.
2. Run the old pipeline on the same checkpoint and a representative subset of
   your data. Compare AUC (when labelled), score distributions and top-ranked
   molecule identities. Align identities/order before comparing score arrays;
   arbitrary files from different pipelines may have different ordering.
3. Account for precision and preprocessing differences. Original FP16 results
   need not be bit-identical to FP32. The portable default explicitly selects
   conformer 0; upstream randomly selected conformers. Pocket cropping uses a
   stable seed instead of upstream's process-dependent hash. Set pocket atom
   limits intentionally: old retrieval examples used 256, while the validated
   AMPC portable run used 511. Cropping/conformer changes can change rankings.
4. Confirm consumers handle the new CSV/NumPy outputs and omitted legacy metrics
   or caching. The supplied AMPC acceptance values are AUC about 0.771345 and
   16 actives in the top 29; other targets need their own acceptance criteria.

Only AMPC inference and the documented unit/export/build checks were validated
for this release. Full DUD-E, PCBA, training and HomoAug were not run.

## 6. Cut over and retain rollback

Change your existing job's command to the **new checkout's absolute Python
path**, set its working directory to the new checkout, and supply explicit
model/data/output paths. Preserve any CUDA library environment settings needed
by the successful manual GPU run. Write to a new output directory during the
transition, and run one scheduled/manual job to confirm the same results.

To roll back, restore the recorded old command, interpreter, working directory,
CUDA settings and output consumer configuration. The old installation and
checkpoint remain available because this procedure never upgraded them in
place. Keep both installations until your own validation is complete.
