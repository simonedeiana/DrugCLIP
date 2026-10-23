"""Export original checkpoint using the legacy PyTorch/Uni-Core environment."""
import argparse
from pathlib import Path
import numpy as np
import torch
import onnx
from unicore import options, tasks, checkpoint_utils
from drugclip_runtime.data import prepare, collate


class Encoder(torch.nn.Module):
    def __init__(self, backbone, projection):
        super().__init__()
        self.backbone, self.projection = backbone, projection

    def forward(self, tokens, distances, edge_types):
        m = self.backbone
        n = tokens.shape[1]
        bias = m.gbf_proj(m.gbf(distances, edge_types)).permute(0, 3, 1, 2).contiguous().reshape(-1, n, n)
        rep = m.encoder(m.embed_tokens(tokens), padding_mask=tokens.eq(m.padding_idx), attn_mask=bias)[0][:, 0, :]
        emb = self.projection(rep)
        return emb / emb.norm(dim=-1, keepdim=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser, input_args=['./data', '--user-dir', './unimol',
        '--task', 'drugclip', '--loss', 'in_batch_softmax', '--arch', 'drugclip', '--max-pocket-atoms', '511'])
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(checkpoint_utils.load_checkpoint_to_cpu(str(a.checkpoint))['model'], strict=True)
    model.float().eval().cpu()
    for kind, prefix in [('molecule', 'mol'), ('pocket', 'pocket')]:
        # Synthetic shapes suffice: the exported axes are dynamic.
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        record = dict(atoms=['C', 'N'], coordinates=coords, pocket_atoms=['C', 'N'], pocket_coordinates=coords)
        inputs = collate([prepare(record, kind)])
        encoder = Encoder(getattr(model, prefix + '_model'), getattr(model, prefix + '_project')).eval()
        dest = a.output / (kind + '.onnx')
        torch.onnx.export(encoder, tuple(torch.from_numpy(inputs[k]) for k in ('tokens', 'distances', 'edge_types')),
            str(dest), opset_version=14, input_names=['tokens', 'distances', 'edge_types'], output_names=['embedding'],
            dynamic_axes={'tokens': {0: 'batch', 1: 'atoms'}, 'distances': {0: 'batch', 1: 'atoms', 2: 'atoms'},
                          'edge_types': {0: 'batch', 1: 'atoms', 2: 'atoms'}, 'embedding': {0: 'batch'}},
            do_constant_folding=True)
        onnx.checker.check_model(onnx.load(str(dest)))
        print('Exported', dest, flush=True)


if __name__ == '__main__':
    main()
