"""Compare every portable input tensor to batches saved from upstream PyTorch."""
import argparse
from pathlib import Path
import numpy as np
from drugclip_runtime.data import batches

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--target', type=Path, required=True)
p.add_argument('--prepared', type=Path, required=True)
a = p.parse_args()
count = 0
for kind, filename in [('molecule', 'mols'), ('pocket', 'pocket')]:
    for i, (inputs, _) in enumerate(batches(a.target / (filename + '.lmdb'), kind, 8 if kind == 'molecule' else 1)):
        with np.load(a.prepared / f'{kind}-{i:04d}.npz') as reference:
            for key in inputs:
                np.testing.assert_array_equal(inputs[key], reference[key], err_msg=f'{kind} batch {i} {key}')
        count += 1
print(f'PASS: {count} batches exactly match upstream preprocessing')
