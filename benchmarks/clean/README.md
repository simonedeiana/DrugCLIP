# Reproduce the clean inference benchmark

See [the measured results](../results/2026-10-28-clean/REPORT.md) for the timing
boundary, platform/precision limits and prediction differences.

The scripts need Python 3.10 and the inference requirements. TensorRT runs also
need `requirements-tensorrt.txt` and locally built FP32 engines. PyTorch needs
the original compatible PyTorch/Uni-Core environment, original source checkout
and official checkpoint. Do not install those legacy dependencies into your
portable runtime environment.

Prepare the fixed subset once, outside benchmarking:

```bash
python benchmarks/clean/prepare.py --data-root /absolute/path/to/datasets --output /absolute/path/to/clean-subset
```

The data root contains `DUD-E/DUD-E/ampc` and `LIT-PCBA/lit_pcba/{GBA,IDH1,VDR}`.
Only trusted LMDB files should be used. Selection uses evenly spaced positions
in lexical LMDB order, not random samples. The manifest stores exact source
positions, labels, batch filenames and SHA256 hashes. On different NumPy/archive
versions, verify the manifest and prepared tensors before comparing results.

Run TensorRT and graph replay together (alternating order), then PyTorch with
its own interpreter. Use absolute paths because the PyTorch setup changes its
working directory to the original source checkout:

```bash
python benchmarks/clean/benchmark.py --backend tensorrt --prepared /absolute/path/to/clean-subset --output /absolute/path/to/clean-results --runtime-repo /absolute/path/to/DrugCLIP-portable --models /absolute/path/to/models/tensorrt
python benchmarks/clean/benchmark.py --backend pytorch --prepared /absolute/path/to/clean-subset --output /absolute/path/to/clean-results --runtime-repo /absolute/path/to/DrugCLIP-portable --source /absolute/path/to/original/DrugCLIP --checkpoint /absolute/path/to/checkpoint_best.pt
```

Do not run the backends simultaneously. Prepared inputs are loaded into host
RAM before timing. Every batch completes its output transfer before returning.
The script saves per-pass timings and embeddings; repeatability checks happen
after each timer stops. It fails if graph capture occurs during a measured pass.
`--kind pocket` or `--kind molecule` allows a single-encoder run; use a fresh
output directory when changing inputs to avoid mixing results from different sets.

On the measured Ubuntu 26.04 environment, loading torch 1.11 required the
process-local `GLIBC_TUNABLES=glibc.rtld.execstack=2` setting. This is recorded
in metadata, not required for other Linux versions. The earlier Ubuntu 22.04
installation was no longer present. TensorRT ran on native Windows.

Cross-backend score comparisons use the maximum similarity over the selected
pockets for each target. The report includes maximum/mean embedding and score
errors, rank correlation, top-1% overlap and active counts. Sampled LIT-PCBA
metrics are descriptive; they are not a reproduction of full-target accuracy.


Generate the comparison after both jobs finish:

```bash
python benchmarks/clean/compare.py --prepared /absolute/path/to/clean-subset --results /absolute/path/to/clean-results --output /absolute/path/to/clean-report
```

The comparison script's narrative documents this recorded experiment (date,
hardware and platforms). Update that narrative when reproducing on another
machine; its numerical tables are calculated from the supplied results.
