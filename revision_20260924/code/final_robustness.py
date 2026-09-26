from pathlib import Path
import json,time
import final_train_evidence as t
import numpy as np
import torch,spglib
W=t.ROOT/'results'; idx=np.arange(250); rng=np.random.default_rng(724)
perms=[rng.permutation(int(t.mask[i].sum())) for i in idx]
results={}
for name in t.MODELS:
 c=torch.load(W/'checkpoints'/f'formation_energy_{name}_42.pt',map_location='cpu',weights_only=True)
 model=t.make(name);model.load_state_dict(c['model']);model.eval()
 if name in ['struct','hierarchy']:x=t.features[name][idx].copy();lab=None;site=model
 else:x=t.elem[idx].copy();lab=np.zeros((250,40),int) if name in ['standard','wide'] else t.labels[name][idx].copy();site=model.base
 xp=x.copy();lp=None if lab is None else lab.copy()
 for j,p in enumerate(perms):
  xp[j,:len(p)]=x[j,p]
  if lp is not None:lp[j,:len(p)]=lab[j,p]
 m=t.mask[idx];pad=np.zeros((250,19),np.float32)
 def run(a,l,b):return site(torch.tensor(a),None if l is None or name=='wide' else torch.tensor(l),torch.tensor(b))
 with torch.no_grad():
  a=run(x,lab,m);b=run(xp,lp,m)
  pp=np.concatenate([rng.normal(size=(250,19,x.shape[-1])).astype(np.float32),x],1)
  ll=None if lab is None else np.concatenate([np.zeros((250,19),int),lab],1)
  d=run(pp,ll,np.concatenate([pad,m],1))
  results[name]={'site_permutation_max_eV':float((a-b).abs().max()*c['sd']),'nuisance_padding_max_eV':float((a-d).abs().max()*c['sd'])}
  for n in [1,64]:
   xx=torch.tensor(x[:n]);mm=torch.tensor(m[:n]);ll=None if lab is None or name=='wide' else torch.tensor(lab[:n])
   for _ in range(20):site(xx,ll,mm)
   times=[]
   for _ in range(100):
    st=time.perf_counter();site(xx,ll,mm);times.append((time.perf_counter()-st)*1000)
   results[name]['cpu_batch_'+str(n)+'_median_ms']=float(np.median(times))
# Raw-coordinate recalculation for Cube4 and Sym4 on the whole cohort.
raw={'cube4_permutation_pass':0,'sym4_permutation_pass':0,'cube4_origin_changed':0,'sym4_origin_changed':0}
changed_hist=t.features['cube4'].copy();sym_hist=t.features['sym4'].copy()
for i,r in enumerate(t.rows):
 z=np.array(r['atomic_numbers']);f=np.array(r['frac_coords']);n=len(z);p=np.random.default_rng(i).permutation(n)
 cube=t.base.cube4_label_from_coords(f);raw['cube4_permutation_pass']+=int(np.array_equal(t.base.cube4_label_from_coords(f[p]),cube[p]))
 def sym(fr,zz):
  ds=spglib.get_symmetry_dataset((np.array(r['lattice']),fr,zz),symprec=1e-3)
  _,iv,ct=np.unique(ds.equivalent_atoms,return_inverse=True,return_counts=True);s=np.rint(len(ds.rotations)/ct[iv]).astype(int)
  return np.where(s==1,0,np.where(s==2,1,np.where(s<=4,2,3)))
 sy=sym(f,z);raw['sym4_permutation_pass']+=int(np.array_equal(sym(f[p],z[p]),sy[p]))
 moved=(f+[.137,.271,.419])%1;cm=t.base.cube4_label_from_coords(moved);sm=sym(moved,z)
 raw['cube4_origin_changed']+=int(not np.array_equal(cm,cube));raw['sym4_origin_changed']+=int(not np.array_equal(sm,sy))
 for target,l in [(changed_hist,cm),(sym_hist,sm)]:
  target[i]=0
  for zz,k in zip(z,l):target[i,k*len(t.zs)+t.zindex[zz]]+=1/n
for name,h in [('cube4',changed_hist),('sym4',sym_hist)]:
 c=torch.load(W/'checkpoints'/f'formation_energy_{name}_42.pt',map_location='cpu',weights_only=True);model=t.make(name);model.load_state_dict(c['model']);model.eval()
 with torch.no_grad():delta=(model(torch.tensor(t.features[name]))-model(torch.tensor(h))).abs()*c['sd']
 raw[name+'_origin_prediction_max_eV']=float(delta.max());raw[name+'_origin_prediction_mean_eV']=float(delta.mean())
out={'device':'CPU','torch_threads':4,'warmup':20,'repetitions':100,'timing_scope':'sitewise cached-feature forward only; batch latency, not per-crystal latency; excludes preprocessing','models':results,'raw_structure':raw}
(t.ROOT/'work_results'/'ROBUSTNESS_COST_RECHECK.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
