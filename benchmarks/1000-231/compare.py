from pathlib import Path
import json,numpy as np,argparse
from scipy.stats import spearmanr,rankdata
p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output;O.mkdir(parents=True,exist_ok=True)
m=json.loads((a.prepared/'manifest.json').read_text());timings={};meta={}
for backend in ['pytorch','tensorrt']:
 r=json.loads((a.results/(backend+'-timings.json')).read_text());meta[backend]=r['metadata'];timings.update(r['timings'])
for r in timings.values():
 x=np.array(r['seconds']);assert len(x)==10
 r.update(mean=float(x.mean()),median=float(np.median(x)),std=float(x.std(ddof=1)),min=float(x.min()),max=float(x.max()),cv_percent=float(x.std(ddof=1)/x.mean()*100))
emb={name:{kind:np.load(a.results/f'{name}-{kind}.npy') for kind in ['molecule','pocket']} for name in ['pytorch','tensorrt','cuda-graphs']}
for x in emb.values():assert x['molecule'].shape==(1000,128) and x['pocket'].shape==(231,128)
scores={name:x['molecule']@x['pocket'].T for name,x in emb.items()};comparison={}
for name in ['pytorch','cuda-graphs']:
 c={'embeddings':{}}
 for kind in ['molecule','pocket']:
  a,b=emb[name][kind].astype(np.float64),emb['tensorrt'][kind].astype(np.float64);diff=np.abs(a-b)
  c['embeddings'][kind]={'max_error':float(diff.max()),'mean_error':float(diff.mean()),'min_cosine':float(np.min(np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)))),'exact':bool(np.array_equal(a,b))}
 diff=np.abs(scores[name]-scores['tensorrt']);c['scores']={'max_error':float(diff.max()),'mean_error':float(diff.mean()),'exact':bool(np.array_equal(scores[name],scores['tensorrt']))}
 targets=[];offset=0
 for row in m['targets']:
  count=row['pocket']['records'];a=scores[name][:,offset:offset+count].max(axis=1);b=scores['tensorrt'][:,offset:offset+count].max(axis=1);offset+=count
  ordera=np.argsort(-a,kind='stable');orderb=np.argsort(-b,kind='stable')
  t={'suite':row['suite'],'target':row['target'],'pockets':count,'max_score_error':float(np.abs(a-b).max()),'mean_score_error':float(np.abs(a-b).mean()),'spearman':float(spearmanr(a,b).statistic),'top10_overlap':len(set(ordera[:10])&set(orderb[:10]))}
  if row['target']=='ampc' and row['suite']=='DUD-E':
   y=np.array(row['labels']);pos=int(y.sum());neg=1000-pos
   t.update(auc=float((rankdata(a)[y==1].sum()-pos*(pos+1)/2)/(pos*neg)),reference_auc=float((rankdata(b)[y==1].sum()-pos*(pos+1)/2)/(pos*neg)),top10_actives=int(y[ordera[:10]].sum()),reference_top10_actives=int(y[orderb[:10]].sum()))
  targets.append(t)
 c['targets']=targets;c['minimum_rank_correlation']=min(t['spearman'] for t in targets);c['minimum_top10_overlap']=min(t['top10_overlap'] for t in targets);comparison[name]=c
res={'metadata':meta,'timings':timings,'comparisons':comparison}
(O/'results.json').write_text(json.dumps(res,indent=2))
lines=['# 1,000 molecules and all 231 pockets: ten-pass inference benchmark','',
'Measured 28 October 2026 on RTX 3050 8 GB, driver 617.42. All 231 standard pockets from 102 DUD-E and 15 LIT-PCBA targets are included. The 1,000 molecules are evenly spaced positions from AMPC in lexical LMDB order; no label-based selection. Exact indices and prepared-tensor hashes are in the manifest.','',
'## Timing method','',
'Preprocessing is completed before benchmarking (conformer 0, seed 1, max 511 pocket atoms). Inputs reside in host RAM. Molecules use batch size 8, pockets batch size 1. Each encoder is grouped by exact input shape, identically for all backends. A bounded graph cache keeps one shape at a time, covering every pocket without GPU-memory pressure.','',
'For each group: three untimed warmups, then **10 timed passes**. Each record is inferred ten times in measured sections. Aggregate sample j is the sum of pass j across all shape groups; these are grouped-pass totals, not ten uninterrupted end-to-end sweeps. TensorRT and graphs alternate order each pass. PyTorch runs separately; no simultaneous GPU benchmark jobs.','',
'**Included:** host-to-device transfers, encoder execution, device-to-host transfers, synchronization and runtime dispatch/allocation. **Excluded:** disk I/O, preprocessing, model loading, warmups, graph capture, numerical checks, scoring/ranking and saving. Capture counters must remain unchanged inside measured passes. Every timed pass is checked afterward against its warmup outputs. This is warm host-to-host inference, not a GPU-kernel-only benchmark or cold-start measurement.','',
'PyTorch is original-model FP16, torch 1.11.0+cu113, Ubuntu 26.04 WSL with process-local `GLIBC_TUNABLES=glibc.rtld.execstack=2` for legacy-library compatibility. TensorRT 10.13.3.9 is FP32 with TF32 disabled, native Windows; graph replay uses exactly those engines. OS and precision differences remain, so PyTorch comparisons describe these deployments. Ten repeats quantify short-term variation, not all sources of bias.','',
'## All three backends, separated by encoder','',
'| Workload | Backend | Mean | Median | Std. dev. | Min–max |','|---|---|---:|---:|---:|---:|']
for kind in ['molecule','pocket']:
 for name in ['pytorch','tensorrt','cuda-graphs']:
  r=timings[name+'-'+kind];lines.append(f"| {r['records']} {kind} records | {name} | {r['mean']:.6f} s | {r['median']:.6f} s | {r['std']:.6f} s | {r['min']:.6f}–{r['max']:.6f} s |")
lines+=['','## Three-way time reductions','', '| Comparison | Molecules | Pockets |','|---|---:|---:|']
for base,new in [('pytorch','tensorrt'),('pytorch','cuda-graphs'),('tensorrt','cuda-graphs')]:
 vals=[100*(1-timings[new+'-'+k]['mean']/timings[base+'-'+k]['mean']) for k in ['molecule','pocket']]
 lines.append(f'| {new} relative to {base} | {vals[0]:+.2f}% | {vals[1]:+.2f}% |')
lines+=['','Positive percentages mean less elapsed time; negative means slower. Graph replay is enabled for both encoders here. Shape grouping and warmed graphs are intentional; do not extrapolate these times to a cold, mixed-shape screen.','',
'## Prediction differences','',
'All 231,000 molecule/pocket similarities are calculated after timing. Each target score takes the maximum over that target’s pockets. Because these 1,000 molecules come from AMPC, scores against other receptors are computational agreement checks, not labelled screening-accuracy tests. Top-1% means the top 10 of 1,000 molecules.','',
'| Compared with TensorRT | Molecule embedding max error | Pocket embedding max error | Score max error | Score mean error | Worst target rank correlation | Minimum top-10 overlap |','|---|---:|---:|---:|---:|---:|---:|']
for name,c in comparison.items():lines.append(f"| {name} | {c['embeddings']['molecule']['max_error']:.9g} | {c['embeddings']['pocket']['max_error']:.9g} | {c['scores']['max_error']:.9g} | {c['scores']['mean_error']:.9g} | {c['minimum_rank_correlation']:.9f} | {c['minimum_top10_overlap']}/10 |")
lines+=['','Raw ten-pass samples, per-shape samples, embedding mean errors/cosines, and per-target score/rank comparisons are in `results.json`. Input manifest and model hashes accompany the report. No timings were discarded. Graph equality is assessed from measured outputs. PyTorch FP16 is not assumed bitwise equal to FP32.','']
(O/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))
