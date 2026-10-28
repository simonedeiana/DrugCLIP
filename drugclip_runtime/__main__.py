"""Screen a trusted molecule LMDB against a pocket LMDB."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .backends import OnnxEncoder, TensorRTEncoder, CudaGraphEncoder
from .data import batches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['onnx', 'tensorrt'], default='onnx')
    p.add_argument('--cuda-graphs', action='store_true', help='Cache CUDA graphs for TensorRT molecule inference')
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--molecules', type=Path, required=True)
    p.add_argument('--pockets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--conformer', type=int, default=0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--max-pocket-atoms', type=int, default=511)
    a = p.parse_args()
    if a.cuda_graphs and a.backend != 'tensorrt':
        p.error('--cuda-graphs requires --backend tensorrt')
    if a.batch_size < 1 or a.max_pocket_atoms < 1:
        p.error('batch-size and max-pocket-atoms must be positive')
    a.output.mkdir(parents=True, exist_ok=True)
    cls = OnnxEncoder if a.backend == 'onnx' else TensorRTEncoder
    suffix = '.onnx' if a.backend == 'onnx' else '-fp32.engine'
    embeddings, names, timing = {}, {}, {}
    started = time.perf_counter()
    for kind, path in [('molecule', a.molecules), ('pocket', a.pockets)]:
        encoder_cls = CudaGraphEncoder if a.cuda_graphs and kind == 'molecule' else cls
        encoder = encoder_cls(a.models / (kind + suffix))
        chunks, names[kind] = [], []
        t = time.perf_counter()
        try:
            for inputs, descriptions in batches(path, kind, a.batch_size if kind == 'molecule' else 1,
                    conformer=a.conformer, seed=a.seed, max_pocket_atoms=a.max_pocket_atoms):
                result = encoder(inputs)
                norms = np.linalg.norm(result, axis=1)
                if not np.isfinite(result).all() or not np.allclose(norms, 1, atol=0.01):
                    raise RuntimeError(f'Invalid {kind} embeddings')
                chunks.append(result)
                names[kind].extend(descriptions)
        finally:
            if hasattr(encoder, 'close'):
                encoder.close()
        if not chunks:
            raise ValueError(f'Empty {kind} dataset')
        embeddings[kind] = np.concatenate(chunks)
        timing[kind] = time.perf_counter() - t
        np.save(a.output / (kind + '-embeddings.npy'), embeddings[kind])
    # Maximum similarity across pocket conformations, matching screening.
    scores = np.full(len(names['molecule']), -np.inf, dtype=np.float32)
    for pocket in embeddings['pocket']:
        scores = np.maximum(scores, embeddings['molecule'] @ pocket)
    order = np.argsort(-scores, kind='stable')
    labels = [x['label'] for x in names['molecule']]
    metrics = dict(backend=a.backend, cuda_graphs=a.cuda_graphs, molecules=len(scores), pockets=len(names['pocket']),
                   seconds_including_startup=time.perf_counter() - started,
                   preprocess_and_inference_seconds=timing, conformer=a.conformer, seed=a.seed,
                   max_pocket_atoms=a.max_pocket_atoms)
    if all(x is not None for x in labels):
        y = np.asarray(labels, dtype=float)
        if not np.isin(y, [0, 1]).all():
            raise ValueError('Benchmark labels must be binary 0/1')
        pos, neg = int(y.sum()), int((y == 0).sum())
        if pos and neg:
            metrics['roc_auc'] = float((rankdata(scores)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
            count = int(np.ceil(len(y) * .01))
            metrics['top_1_percent_count'] = count
            metrics['top_1_percent_actives'] = int(y[order[:count]].sum())
    with (a.output / 'rankings.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'index', 'name', 'score', 'label'])
        for rank, i in enumerate(order, 1):
            writer.writerow([rank, int(i), names['molecule'][i]['name'], float(scores[i]), labels[i]])
    np.save(a.output / 'scores.npy', scores)
    (a.output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
