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
results={'metadata':metadata,'timings':{},'method':'10 passes per shape group; aggregate pass j sums group pass j. Three untimed warmups per group. Transfers included, validation excluded.'}
for kind in ['molecule','pocket']:
 groups={}
 for i,x in enumerate(data[kind]):groups.setdefault(tuple(x['tokens'].shape),[]).append(i)
 names=['pytorch'] if a.backend=='pytorch' else ['tensorrt','cuda-graphs']
 baseline=TorchEncoder(kind) if a.backend=='pytorch' else TensorRTEncoder(a.models/f'{kind}-fp32.engine')
 totals={name:np.zeros(10) for name in names};saved={name:[None]*len(data[kind]) for name in names};detail=[]
 for shape,indices in sorted(groups.items()):
  encoders={'pytorch':baseline} if a.backend=='pytorch' else {'tensorrt':baseline,'cuda-graphs':CudaGraphEncoder(a.models/f'{kind}-fp32.engine',max_graphs=1)}
  inputs=[data[kind][i] for i in indices]
  for _ in range(3):
   refs={name:[enc(x) for x in inputs] for name,enc in encoders.items()}
  captures={name:getattr(enc,'captures',0) for name,enc in encoders.items()};times={name:[] for name in names}
  for repeat in range(10):
   for name in (names if repeat%2==0 else names[::-1]):
    enc=encoders[name];start=time.perf_counter();outputs=[enc(x) for x in inputs];elapsed=time.perf_counter()-start
    totals[name][repeat]+=elapsed;times[name].append(elapsed)
    for x,y in zip(outputs,refs[name]):np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-6)
    assert getattr(enc,'captures',0)==captures[name], 'capture inside measured pass'
  for name in names:
   for i,out in zip(indices,refs[name]):saved[name][i]=out
  detail.append({'shape':shape,'batches':len(indices),'seconds':times})
  if 'cuda-graphs' in encoders:encoders['cuda-graphs'].close()
  print(kind,'shape',shape,'batches',len(indices),'complete',flush=True)
 for name in names:
  out=np.concatenate(saved[name]);assert np.isfinite(out).all();np.testing.assert_allclose(np.linalg.norm(out,axis=1),1,atol=.01)
  np.save(a.output/f'{name}-{kind}.npy',out)
  results['timings'][name+'-'+kind]={'seconds':totals[name].tolist(),'records':len(out),'batches':len(data[kind]),'shape_groups':len(groups),'group_details':detail}
 baseline.close()
 (a.output/(a.backend+'-timings.json')).write_text(json.dumps(results,indent=2))
 print('DONE',kind,{name:totals[name].tolist() for name in names},flush=True)
