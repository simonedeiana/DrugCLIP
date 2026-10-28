from pathlib import Path
import json,numpy as np,argparse
from scipy.stats import spearmanr,rankdata
p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();O=args.output;O.mkdir(parents=True,exist_ok=True)
m=json.loads((args.prepared/'manifest.json').read_text());t={}
for backend in ['pytorch','tensorrt']:
 x=json.loads((args.results/(backend+'-timings.json')).read_text());t.update(x['timings'])
summary={k:{**v,'mean_seconds':float(np.mean(v['seconds'])),'std_seconds':float(np.std(v['seconds'],ddof=1)),'min_seconds':min(v['seconds']),'max_seconds':max(v['seconds'])} for k,v in t.items()}
emb={name:{kind:np.load(args.results/f'{name}-{kind}.npy') for kind in ['molecule','pocket']} for name in ['pytorch','tensorrt','cuda-graphs']}
comparisons=[]
for name in ['pytorch','cuda-graphs']:
 entry={'backend':name,'reference':'tensorrt','encoders':{}}
 for kind in ['molecule','pocket']:
  x,y=emb[name][kind].astype(np.float64),emb['tensorrt'][kind].astype(np.float64);d=np.abs(x-y)
  cosine=np.sum(x*y,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1))
  entry['encoders'][kind]={'max_absolute_error':float(d.max()),'mean_absolute_error':float(d.mean()),'min_cosine_similarity':float(cosine.min()),'bitwise_equal':bool(np.array_equal(x,y))}
 comparisons.append(entry)
rows=[];mi=pi=0
for target in m['targets']:
 n=target['molecule']['records'];q=target['pocket']['records'];y=np.array(target['labels']);scores={name:np.max(emb[name]['molecule'][mi:mi+n]@emb[name]['pocket'][pi:pi+q].T,axis=1) for name in emb}
 k=int(np.ceil(n*.01));r={'target':target['target'],'molecules':n,'pockets':q,'actives':int(y.sum()),'top1_count':k,'comparisons':{}}
 for name in ['pytorch','cuda-graphs']:
  a,b=scores[name],scores['tensorrt'];d=np.abs(a-b)
  ar,br=np.argsort(-a,kind='stable'),np.argsort(-b,kind='stable')
  def auc(x):
   pos=int(y.sum());neg=n-pos
   return float((rankdata(x)[y==1].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
  r['comparisons'][name]={'max_score_error':float(d.max()),'mean_score_error':float(d.mean()),'spearman':float(spearmanr(a,b).statistic),'top1_overlap':len(set(ar[:k])&set(br[:k])),'top1_actives':int(y[ar[:k]].sum()),'reference_top1_actives':int(y[br[:k]].sum()),'auc':auc(a),'reference_auc':auc(b),'bitwise_equal':bool(np.array_equal(a,b))}
 rows.append(r);mi+=n;pi+=q
out={'timings':summary,'embedding_comparisons':comparisons,'prediction_comparisons':rows}
(O/'results.json').write_text(json.dumps(out,indent=2))
lines=['# Clean molecule and receptor inference benchmark','', 'Measured 28 October 2026 on NVIDIA RTX 3050 8 GB, driver 617.42.','',
'## Dataset and measurement','',
'4,434 molecule records and 9 receptor/pocket records: all 2,898 AMPC molecules and its pocket, plus 512 evenly spaced lexical-LMDB records each from GBA, IDH1 and VDR, and up to three evenly spaced pockets per target (3, 3 and 2). Sampling positions and prepared-input SHA256 hashes are in `manifest.json`. This deliberately small subset is not a full-dataset accuracy evaluation.','',
'Inputs were prepared once (conformer 0, seed 1, max 511 pocket atoms), then loaded into host RAM before timing. Molecule batch size 8; pocket batch size 1. Each encoder/shape was warmed before measurement; CUDA Graph capture was excluded, with an assertion that no capture occurred during timing.','',
'**Timed:** input transfers, encoder inference, output transfer to host and synchronization, including normal runtime allocation/dispatch overhead. **Not timed:** LMDB access, preprocessing, loading inputs/models, warmup/capture, numerical validation, scoring, ranking and writing results. These are host-to-host inference measurements, not CUDA-event kernel-only timings.','',
'An initial 23-pocket attempt exceeded the 16-shape graph cache and was rejected by the no-capture assertion; it is excluded from results. The nine-pocket subset fits the cache. Six measurements per encoder. Each molecule measurement traverses the prepared set once; each pocket measurement averages ten traversals of all 9 pockets. TensorRT and CUDA Graph passes alternate order. PyTorch runs separately so backends never compete for the GPU. Standard deviations describe these six measurements, not a confidence interval over machines.','',
'PyTorch uses the original model in FP16 with torch 1.11.0+cu113 under Ubuntu 26.04 WSL. The earlier Ubuntu 22.04 distro was no longer installed. `GLIBC_TUNABLES=glibc.rtld.execstack=2` was set for this process to load the legacy binaries; no system setting or model was changed. TensorRT 10.13.3.9 FP32 (TF32 disabled) and its graph path run natively on Windows with the same engines. Thus preprocessing/validation overhead is removed, but precision, OS and runtime differences remain.','',
'## Separate encoder timings','',
'| Encoder | Backend | Mean per subset pass | Std. dev. | Records/s |','|---|---|---:|---:|---:|']
for kind in ['molecule','pocket']:
 for name in ['pytorch','tensorrt','cuda-graphs']:
  r=summary[name+'-'+kind];lines.append(f"| {kind} | {name} | {r['mean_seconds']:.6f} s | {r['std_seconds']:.6f} s | {r['records']/r['mean_seconds']:.1f} |")
for kind in ['molecule','pocket']:
 a,b,c=[summary[x+'-'+kind]['mean_seconds'] for x in ['pytorch','tensorrt','cuda-graphs']]
 lines+=['',f"{kind.capitalize()}: TensorRT versus PyTorch time change {(b/a-1)*100:+.2f}%; CUDA Graphs versus ordinary TensorRT {(c/b-1)*100:+.2f}%; CUDA Graphs versus PyTorch {(c/a-1)*100:+.2f}%. Negative means less time."]
lines+=['','## Numerical differences','', 'Reference is ordinary TensorRT FP32. Embedding differences are measured across every selected record after timing.','', '| Compared backend | Encoder | Max absolute error | Mean absolute error | Minimum cosine similarity |','|---|---|---:|---:|---:|']
for r in comparisons:
 for kind,v in r['encoders'].items():lines.append(f"| {r['backend']} | {kind} | {v['max_absolute_error']:.9g} | {v['mean_absolute_error']:.9g} | {v['min_cosine_similarity']:.9f} |")
lines+=['','Predictions are maximum molecule/pocket cosine scores across each target\'s selected pockets. Scoring and ranking are outside the timer.','', '| Target | Compared backend | Max score difference | Mean score difference | Rank correlation | Top-1% overlap | Top-1% actives (compared/reference) |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 for name,v in r['comparisons'].items():lines.append(f"| {r['target']} | {name} | {v['max_score_error']:.9g} | {v['mean_score_error']:.9g} | {v['spearman']:.9f} | {v['top1_overlap']}/{r['top1_count']} | {v['top1_actives']}/{v['reference_top1_actives']} |")
lines+=['','Complete per-pass timing samples, AUC values where both classes are present, numerical differences and runtime metadata are in the JSON files. AUC on the 512-record subsets is descriptive only. Exact graph equality is reported from observed values, not assumed. The graph experiment here enables replay for **both** encoders; the earlier full-screening run used it for molecules only. This warm benchmark excludes capture costs and does not imply that pocket graphs help one-off screening.','']
(O/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines))
