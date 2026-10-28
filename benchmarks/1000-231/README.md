# Reproduce the 1,000-molecule / 231-pocket benchmark

This expands pocket coverage to all 117 standard DUD-E/LIT-PCBA targets. It uses
1,000 evenly spaced AMPC molecule records and all 231 pockets. Every record is
inferred in ten measured passes, following three untimed warmups per shape group.

Use the same environments described in [the clean benchmark guide](../clean/README.md).
Prepare once, then run backends sequentially, with absolute paths:

```bash
python benchmarks/1000-231/prepare.py --data-root /absolute/path/to/datasets --output /absolute/path/to/inputs
python benchmarks/1000-231/benchmark.py --backend tensorrt --prepared /absolute/path/to/inputs --output /absolute/path/to/results --runtime-repo /absolute/path/to/DrugCLIP-portable --models /absolute/path/to/models/tensorrt
python benchmarks/1000-231/benchmark.py --backend pytorch --prepared /absolute/path/to/inputs --output /absolute/path/to/results --runtime-repo /absolute/path/to/DrugCLIP-portable --source /absolute/path/to/original/DrugCLIP --checkpoint /absolute/path/to/checkpoint_best.pt
python benchmarks/1000-231/compare.py --prepared /absolute/path/to/inputs --results /absolute/path/to/results --output /absolute/path/to/report
```

Use the original PyTorch/Uni-Core interpreter for the PyTorch command. Both
TensorRT modes use one ordinary locally built FP32 engine per encoder.
The benchmark groups inputs by shape identically for all backends, keeping one
captured shape resident at a time. This avoids cache churn and memory pressure
without omitting pockets. All data preparation, capture and validation are
outside the timer. Transfers and synchronization remain inside it.

The ten aggregate samples sum corresponding pass indices across shape groups;
they are not uninterrupted whole-set sweeps. Both raw group times and aggregate
times are preserved. Reproduction requires the same source datasets and model;
compare input-manifest hashes. Use new output directories for each experiment.

Scoring compares all 1,000 molecules with all 231 pockets after timing. This
is a numerical agreement test, not evidence of screening quality against all
117 targets: the molecule labels belong only to AMPC. Per-target predictions
use maximum similarity across each target's pockets. The comparison script's
narrative describes the recorded hardware/date; update it for another machine.

See [the report and raw results](../results/2026-10-28-1000-231/REPORT.md).
