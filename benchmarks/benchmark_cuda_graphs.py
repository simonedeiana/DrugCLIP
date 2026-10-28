import sys,time,json,argparse
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W))
from drugclip_runtime.backends import TensorRTEncoder,CudaGraphEncoder
p=argparse.ArgumentParser()
p.add_argument('--models',type=Path,required=True)
p.add_argument('--prepared',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
results=[]
for kind in ['molecule','pocket']:
    files=sorted(a.prepared.glob(kind+'-*.npz'))
    if not files: raise ValueError(f'No {kind} batches found')
    inputs=[]
    for file in files:
        with np.load(file) as f: inputs.append({k:f[k] for k in ['tokens','distances','edge_types']})
    engine=a.models/f'{kind}-fp32.engine'
    base=TensorRTEncoder(engine);graph=CudaGraphEncoder(engine,max_graphs=16)
    expected=[base(x) for x in inputs]
    start=time.perf_counter();actual=[graph(x) for x in inputs];cold=time.perf_counter()-start
    err=max(float(np.max(np.abs(x-y))) for x,y in zip(expected,actual))
    for x,y in zip(expected,actual): np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-6)
    measurements={'baseline':[],'graphs':[]}
    for repeat in range(6):
        for name,enc in ([('baseline',base),('graphs',graph)] if repeat%2==0 else [('graphs',graph),('baseline',base)]):
            start=time.perf_counter()
            # Repeat pocket to amortize timer noise; molecule uses the full 363 batches.
            for _ in range(20 if kind=='pocket' else 1):
                for x in inputs: enc(x)
            measurements[name].append((time.perf_counter()-start)/(20 if kind=='pocket' else 1))
    r=dict(kind=kind,batches=len(inputs),unique_shapes=len({tuple(x['tokens'].shape) for x in inputs}),max_error=err,bitwise_equal=all(np.array_equal(x,y) for x,y in zip(expected,actual)),cold_graph_seconds=cold,seconds=measurements,captures=graph.captures,hits=graph.hits,evictions=graph.evictions)
    results.append(r);print(json.dumps(r),flush=True)
    graph.close();base.close()
a.output.write_text(json.dumps(results,indent=2))
