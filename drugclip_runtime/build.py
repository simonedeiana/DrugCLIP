"""Build strict FP32 engines locally for the current OS/GPU/TensorRT."""
import argparse
from pathlib import Path
import tensorrt as trt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=['molecule', 'pocket', 'both'], default='both')
    p.add_argument('--max-molecule-tokens', type=int, default=128)
    a = p.parse_args()
    if a.max_molecule_tokens < 40:
        p.error('max-molecule-tokens must be at least 40')
    a.output.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    for kind in (['molecule', 'pocket'] if a.kind == 'both' else [a.kind]):
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(a.models / (kind + '.onnx'))):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level = 2
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        profile = builder.create_optimization_profile()
        ob, on, mb, mn = (8, 40, 8, a.max_molecule_tokens) if kind == 'molecule' else (1, 224, 1, 520)
        for key in ('tokens', 'distances', 'edge_types'):
            shapes = [(b, n) if key == 'tokens' else (b, n, n) for b, n in [(1, 2), (ob, on), (mb, mn)]]
            profile.set_shape(key, *shapes)
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TensorRT build failed')
        dest = a.output / (kind + '-fp32.engine')
        dest.write_bytes(bytes(serialized))
        print('Built', dest, flush=True)


if __name__ == '__main__':
    main()
