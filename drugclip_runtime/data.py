"""LMDB preprocessing compatible with DrugCLIP's inference tensors.

LMDB values use Python pickle: load only trusted datasets.
Conformer selection is explicit; pocket cropping uses a stable seeded RNG.
"""
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
from scipy.spatial import distance_matrix

METADATA = json.loads(Path(__file__).with_name('model-metadata.json').read_text())


def records(path):
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False, max_readers=256)
    try:
        with env.begin() as txn:
            # The task imports Uni-Core's LMDBDataset (cursor lexical order).
            for index, (_, value) in enumerate(txn.cursor()):
                yield index, pickle.loads(value)
    finally:
        env.close()


def prepare(record, kind, index=0, conformer=0, seed=1, max_pocket_atoms=511):
    info = METADATA['encoders'][kind]
    vocab = {symbol: i for i, symbol in enumerate(info['symbols'])}
    if kind == 'molecule':
        atoms = np.asarray(record['atoms'])
        coords = np.asarray(record['coordinates'], dtype=np.float32)
        if coords.ndim == 3:
            if not 0 <= conformer < len(coords):
                raise ValueError(f'Conformer {conformer} absent in molecule {index}')
            coords = coords[conformer]
    else:
        atoms = np.asarray([a[1] if a[0].isdigit() else a[0] for a in record['pocket_atoms']])
        coords = np.asarray(record['pocket_coordinates'], dtype=np.float32)
    if coords.shape != (len(atoms), 3) or not np.isfinite(coords).all():
        raise ValueError(f'Invalid coordinates in {kind} record {index}')
    keep = atoms != 'H'
    atoms, coords = atoms[keep], coords[keep]
    if not len(atoms):
        raise ValueError(f'No heavy atoms in {kind} record {index}')
    if kind == 'pocket' and len(atoms) > max_pocket_atoms:
        if max_pocket_atoms < 1:
            raise ValueError('max_pocket_atoms must be positive')
        distance = np.linalg.norm(coords - coords.mean(axis=0), axis=1) + 1
        weight = np.reciprocal(distance)
        weight = np.exp(weight - weight.max())
        weight /= weight.sum()
        rng = np.random.RandomState(np.random.SeedSequence([seed, index]).generate_state(1)[0])
        chosen = rng.choice(len(atoms), max_pocket_atoms, replace=False, p=weight)
        atoms, coords = atoms[chosen], coords[chosen]
    coords = (coords - coords.mean(axis=0)).astype(np.float32)
    tokens = np.array([vocab['[CLS]']] + [vocab.get(a, vocab['[UNK]']) for a in atoms]
                      + [vocab['[SEP]']], dtype=np.int64)
    distances = np.pad(distance_matrix(coords, coords).astype(np.float32), 1)
    return dict(tokens=tokens, distances=distances,
                edge_types=tokens[:, None] * len(vocab) + tokens[None, :])


def collate(items):
    length = ((max(len(x['tokens']) for x in items) + 7) // 8) * 8
    result = dict(tokens=np.zeros((len(items), length), np.int64),
                  distances=np.zeros((len(items), length, length), np.float32),
                  edge_types=np.zeros((len(items), length, length), np.int64))
    for i, item in enumerate(items):
        n = len(item['tokens'])
        result['tokens'][i, :n] = item['tokens']
        for key in ('distances', 'edge_types'):
            result[key][i, :n, :n] = item[key]
    return result


def batches(path, kind, batch_size, **options):
    items, descriptions = [], []
    for index, record in records(path):
        items.append(prepare(record, kind, index=index, **options))
        descriptions.append(dict(index=index, name=str(record.get('smi' if kind == 'molecule' else 'pocket', index)),
                                 label=record.get('label')))
        if len(items) == batch_size:
            yield collate(items), descriptions
            items, descriptions = [], []
    if items:
        yield collate(items), descriptions
