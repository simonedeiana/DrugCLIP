import sys,json,pickle,hashlib,argparse
from pathlib import Path
import numpy as np,lmdb
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W))
from drugclip_runtime.data import prepare,collate
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();B=a.data_root;O=a.output;O.mkdir(parents=True,exist_ok=True)
manifest={'targets':[],'preprocessing':{'conformer':0,'seed':1,'max_pocket_atoms':511,'molecule_batch':8,'pocket_batch':1},'molecule_selection':'1000 evenly spaced lexical LMDB positions from DUD-E AMPC; no label-based selection'}
for suite,dirname,filename in [('DUD-E','DUD-E','pocket.lmdb'),('LIT-PCBA','lit_pcba','pockets.lmdb')]:
 for root in sorted((B/suite/dirname).iterdir()):
  if not (root/filename).exists():continue
  row={'suite':suite,'target':root.name,'molecule':{'files':[],'records':0}};dest=O/(suite+'-'+root.name);dest.mkdir(exist_ok=True)
  kinds=[('pocket',filename)]
  if suite=='DUD-E' and root.name=='ampc':kinds.insert(0,('molecule','mols.lmdb'))
  for kind,file in kinds:
   env=lmdb.open(str(root/file),subdir=False,readonly=True,lock=False,readahead=False);items=[];labels=[]
   with env.begin() as txn:
    keys=list(txn.cursor().iternext(values=False));indices=list(range(len(keys))) if kind=='pocket' else np.linspace(0,len(keys)-1,1000,dtype=int).tolist()
    for i in indices:
     rec=pickle.loads(txn.get(keys[i]));items.append(prepare(rec,kind,index=i));labels.append(rec.get('label'))
   env.close();names=[];bs=8 if kind=='molecule' else 1
   for i in range(0,len(items),bs):
    path=dest/f'{kind}-{i//bs:04d}.npz';np.savez(path,**collate(items[i:i+bs]));names.append(path.relative_to(O).as_posix())
   row[kind]={'source_records':len(keys),'selected_indices':indices,'files':names,'records':len(indices)}
   if kind=='molecule':row['labels']=labels
  manifest['targets'].append(row)
assert sum(r['molecule']['records'] for r in manifest['targets'])==1000
assert sum(r['pocket']['records'] for r in manifest['targets'])==231
manifest['input_sha256']={name:hashlib.sha256((O/name).read_bytes()).hexdigest() for r in manifest['targets'] for k in ['molecule','pocket'] for name in r[k]['files']}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Prepared 1000 molecules, 231 pockets, 117 targets')
