import sys,json,pickle,hashlib,argparse
from pathlib import Path
import numpy as np,lmdb
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W))
from drugclip_runtime.data import prepare,collate
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();B=a.data_root;O=a.output;O.mkdir(parents=True,exist_ok=True)
manifest={'targets':[],'preprocessing':{'conformer':0,'seed':1,'max_pocket_atoms':511,'molecule_batch':8,'pocket_batch':1}}
for suite,target,limit in [('DUD-E','ampc',None),('LIT-PCBA','GBA',512),('LIT-PCBA','IDH1',512),('LIT-PCBA','VDR',512)]:
 root=B/suite/('DUD-E' if suite=='DUD-E' else 'lit_pcba')/target
 row={'suite':suite,'target':target};dest=O/target;dest.mkdir(exist_ok=True)
 for kind,filename in [('molecule','mols.lmdb'),('pocket','pocket.lmdb' if suite=='DUD-E' else 'pockets.lmdb')]:
  env=lmdb.open(str(root/filename),subdir=False,readonly=True,lock=False,readahead=False)
  items=[];labels=[]
  with env.begin() as txn:
   keys=list(txn.cursor().iternext(values=False));count=len(keys)
   indices=np.linspace(0,count-1,min(count,3),dtype=int).tolist() if kind=='pocket' else (list(range(count)) if limit is None else np.linspace(0,count-1,min(count,limit),dtype=int).tolist())
   for i in indices:
    rec=pickle.loads(txn.get(keys[i]));items.append(prepare(rec,kind,index=i));labels.append(rec.get('label'))
  env.close();bs=8 if kind=='molecule' else 1
  names=[]
  for i in range(0,len(items),bs):
   name=f'{kind}-{i//bs:04d}.npz';np.savez(dest/name,**collate(items[i:i+bs]));names.append(target+'/'+name)
  row[kind]={'source_records':count,'selected_indices':indices,'files':names,'records':len(indices)}
  if kind=='molecule':row['labels']=labels
 manifest['targets'].append(row)
manifest['input_sha256']={name:hashlib.sha256((O/name).read_bytes()).hexdigest() for r in manifest['targets'] for k in ['molecule','pocket'] for name in r[k]['files']}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
print([(r['target'],r['molecule']['records'],r['pocket']['records']) for r in manifest['targets']])
