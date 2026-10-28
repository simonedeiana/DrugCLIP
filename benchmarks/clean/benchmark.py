"""Prepared-input inference benchmark. Validation/scoring and warmup are untimed."""
import argparse,sys,time,json,platform
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['pytorch','tensorrt'],required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--runtime-repo',type=Path,required=True);p.add_argument('--models',type=Path);p.add_argument('--source',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--kind',choices=['all','molecule','pocket'],default='all');a=p.parse_args()
sys.path.insert(0,str(a.runtime_repo));a.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads((a.prepared/'manifest.json').read_text());data={}
for kind in ['molecule','pocket']:
 data[kind]=[]
 for row in manifest['targets']:
  for name in row[kind]['files']:
   with np.load(a.prepared/name) as z:data[kind].append({k:np.ascontiguousarray(z[k]) for k in ['tokens','distances','edge_types']})
if a.backend=='pytorch':
 import os,torch
 sys.path.insert(0,str(a.source));os.chdir(a.source)
 from unicore import checkpoint_utils,options,tasks
 torch.set_num_threads(6);torch.manual_seed(1)
 parser=options.get_validation_parser();options.add_model_args(parser)
 args=options.parse_args_and_arch(parser,input_args=['./data','--user-dir','./unimol','--task','drugclip','--loss','in_batch_softmax','--arch','drugclip','--fp16','--path',str(a.checkpoint),'--max-pocket-atoms','511','--batch-size','8','--num-workers','0','--seed','1'])
 state=checkpoint_utils.load_checkpoint_to_cpu(args.path);task=tasks.setup_task(args);model=task.build_model(args);model.load_state_dict(state['model'],strict=True);del state
 model=model.half().cuda().eval();torch.cuda.synchronize()
 class TorchEncoder:
  def __init__(self,kind):
   prefix='mol' if kind=='molecule' else 'pocket';self.m=getattr(model,prefix+'_model');self.proj=getattr(model,prefix+'_project')
  def __call__(self,x):
   with torch.inference_mode():
    tokens,distances,edges=[torch.from_numpy(x[k]).cuda() for k in ['tokens','distances','edge_types']];m=self.m;n=tokens.shape[1]
    bias=m.gbf_proj(m.gbf(distances,edges)).permute(0,3,1,2).contiguous().reshape(-1,n,n)
    rep=m.encoder(m.embed_tokens(tokens),padding_mask=tokens.eq(m.padding_idx),attn_mask=bias)[0][:,0,:]
    out=self.proj(rep);out=out/out.norm(dim=-1,keepdim=True)
    return out.cpu().float().numpy()
  def close(self):pass
 metadata={'torch':torch.__version__,'cuda':torch.version.cuda,'precision':'FP16','platform':platform.platform(),'glibc_tunable':os.environ.get('GLIBC_TUNABLES'),'gpu':torch.cuda.get_device_name(0)}
else:
 import tensorrt
 from drugclip_runtime.backends import TensorRTEncoder,CudaGraphEncoder
 metadata={'tensorrt':tensorrt.__version__,'precision':'FP32 TF32 disabled','platform':platform.platform()}
result_path=a.output/(a.backend+'-timings.json')
results=json.loads(result_path.read_text()) if a.kind!='all' and result_path.exists() else {'metadata':metadata,'timings':{}}
for kind in (['molecule','pocket'] if a.kind=='all' else [a.kind]):
 encoders={'pytorch':TorchEncoder(kind)} if a.backend=='pytorch' else {'tensorrt':TensorRTEncoder(a.models/f'{kind}-fp32.engine'),'cuda-graphs':CudaGraphEncoder(a.models/f'{kind}-fp32.engine')}
 # Warm every input/shape before starting any timer.
 refs={name:[enc(x) for x in data[kind]] for name,enc in encoders.items()}
 captures={name:getattr(enc,'captures',0) for name,enc in encoders.items()}
 times={name:[] for name in encoders};reps=10 if kind=='pocket' else 1
 for repeat in range(6):
  for name in (list(encoders) if repeat%2==0 else list(encoders)[::-1]):
   enc=encoders[name];start=time.perf_counter()
   for _ in range(reps):outputs=[enc(x) for x in data[kind]]
   elapsed=(time.perf_counter()-start)/reps
   times[name].append(elapsed)
   # Validation is explicitly after the timer, including repeatability.
   for x,y in zip(outputs,refs[name]):np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-6)
   if getattr(enc,'captures',0)!=captures[name]:raise RuntimeError('Graph capture occurred inside timing; increase cache or reduce shapes')
  print(kind,repeat,times,flush=True)
 for name,enc in encoders.items():
  out=np.concatenate(refs[name]);assert np.isfinite(out).all();np.testing.assert_allclose(np.linalg.norm(out,axis=1),1,atol=.01)
  np.save(a.output/f'{name}-{kind}.npy',out)
  results['timings'][name+'-'+kind]={'seconds':times[name],'records':len(out),'batches':len(data[kind]),'unique_shapes':len({tuple(x['tokens'].shape) for x in data[kind]}),'passes_per_measurement':reps,'graph_captures_before_timing':captures[name]}
  enc.close()
 (a.output/(a.backend+'-timings.json')).write_text(json.dumps(results,indent=2))
