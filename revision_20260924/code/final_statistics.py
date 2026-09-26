from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'results'
import numpy as np
OLD=ROOT
rows=[json.loads(s) for s in (OLD/'data'/'formation_energy_20260913.jsonl').read_text().splitlines()]
sym=json.loads((W/'site_symmetry_records.json').read_text())
def system(s):
    for top,name in [(2,'triclinic'),(15,'monoclinic'),(74,'orthorhombic'),(142,'tetragonal'),(167,'trigonal'),(194,'hexagonal'),(230,'cubic')]:
        if s<=top:return name
def stats(v,fam,seed):
    _,inv=np.unique(fam,return_inverse=True);n=inv.max()+1
    sums=np.bincount(inv,weights=v);counts=np.bincount(inv)
    rng=np.random.default_rng(seed);ix=rng.integers(0,n,(5000,n))
    draws=sums[ix].sum(1)/counts[ix].sum(1);lo,hi=np.quantile(draws,[.025,.975])
    observed=abs(sums.sum());extreme=0
    for _ in range(40):
        signs=rng.integers(0,2,(5000,n),dtype=np.int8)*2-1
        extreme+=int((abs(signs@sums)>=observed-1e-15).sum())
    return {'delta':float(v.mean()),'ci':[float(lo),float(hi)],'p_raw':(extreme+1)/200001,'families':int(n),'records':len(v)}
def holm(data):
    keys=sorted(data,key=lambda k:data[k]['p_raw']);largest=0
    for i,k in enumerate(keys):largest=max(largest,(len(keys)-i)*data[k]['p_raw']);data[k]['p_holm']=min(largest,1.)
def fetch(task,name,seeds,tag=''):
    dat=[np.load(W/'predictions'/f'{tag}{task}_{name}_{s}.npz') for s in seeds]
    y=dat[0]['y'];ix=dat[0]['index'];pred=np.stack([d['pred'] for d in dat])
    err=np.abs(pred-y[None,:]);return err.mean(0),err.mean(1),ix
def analyse(task,names,seeds,tag=''):
    out={'models':{},'paired':{},'strata':{}}; errors={}
    for name in names:
        err,maes,ix=fetch(task,name,seeds,tag);errors[name]=err
        r=json.loads((W/f'{tag}{task}_{name}_{seeds[0]}.json').read_text())
        out['models'][name]={'mean':float(maes.mean()),'sd':float(maes.std(ddof=1)),'seeds':seeds,'per_seed':maes.tolist(),'parameters':r['trainable_parameters'],'fixed_parameters':r['fixed_parameters']}
    fam=np.array([rows[i]['family_id'] for i in ix]);out['n_test']=len(ix);out['n_families']=len(set(fam))
    pairs=[('cube4',m) for m in ['standard','wide','geo4','occupancy4','random4','sym4']]+[('sym4','standard'),('struct','standard'),('hierarchy','struct')]
    for k,(a,b) in enumerate(pairs):
        if a in errors and b in errors:out['paired'][a+'_vs_'+b]=stats(errors[a]-errors[b],fam,9000+k)
    holm(out['paired'])
    if task=='formation_energy' and not tag:
        systems=np.array([system(sym[i]['spacegroup']) for i in ix]);ops=np.array([sym[i]['nops'] for i in ix]);q=np.quantile(ops,[1/3,2/3]);bands=np.where(ops<=q[0],'low',np.where(ops<=q[1],'medium','high'))
        out['stratum_thresholds']=q.tolist()
        for kind,labels in [('crystal_system',systems),('symmetry_band',bands)]:
            table={};alltests={}
            for j,value in enumerate(sorted(set(labels))):
                sel=labels==value;table[value]={'n':int(sel.sum()),'means':{m:float(e[sel].mean()) for m,e in errors.items()},'paired':{}}
                for k,m in enumerate(['cube4','sym4','wide']):
                    test=stats((errors[m]-errors['standard'])[sel],fam[sel],10000+j*100+k)
                    table[value]['paired'][m+'_vs_standard']=test;alltests[value+'_'+m]=test
            holm(alltests);out['strata'][kind]=table
        out['reweighting']={'empirical':{m:float(e.mean()) for m,e in errors.items()},
          'equal_crystal_systems':{m:float(np.mean([e[systems==s].mean() for s in set(systems)])) for m,e in errors.items()},
          'equal_symmetry_bands':{m:float(np.mean([e[bands==s].mean() for s in set(bands)])) for m,e in errors.items()}}
    return out
def main():
    seeds=[42,123,456,789,1024];names=['standard','wide','cube4','geo4','occupancy4','random4','sym4','struct','hierarchy']
    out={'formation_energy':analyse('formation_energy',names,seeds),
      'band_gap':analyse('band_gap',['standard','wide','cube4','sym4'],seeds[:3]),
      'holdout':analyse('formation_energy',['standard','wide','cube4','sym4'],seeds[:3],'holdout_')}
    err,maes,ix=fetch('formation_energy','standard',seeds[:3]);fam=np.array([rows[i]['family_id'] for i in ix]);d={'shared_projection_masked_batch256':{'mean':float(maes.mean()),'sd':float(maes.std(ddof=1)),'per_seed':maes.tolist()}}
    for arm in ['sample_projection_masked','sample_projection_unmasked','sample_projection_unmasked_fullbatch']:
        dat=[np.load(W/'predictions'/f'diagnostic_{arm}_{s}.npz') for s in seeds[:3]];es=np.abs(np.stack([x['pred'] for x in dat])-dat[0]['y']);ms=es.mean(1)
        d[arm]={'mean':float(ms.mean()),'sd':float(ms.std(ddof=1)),'per_seed':ms.tolist(),'vs_shared':stats(es.mean(0)-err,fam,12345)}
    holm({k:v['vs_shared'] for k,v in d.items() if 'vs_shared' in v});out['protocol_diagnostics']=d
    old=json.loads((OLD/'results'/'legacy_training_summary.json').read_text())
    out['reproduction_max_abs_mae_delta']={n:max(abs(json.loads((W/f'formation_energy_{n}_{s}.json').read_text())['test_mae']-old[f'{n}@{s}']['test_mae']) for s in seeds) for n in names[:6]}
    out['cohort']={'n':len(rows),'elements':len({z for r in rows for z in r['atomic_numbers']}),'families':len({r['family_id'] for r in rows}),
      'systems':{s:sum(system(r['spacegroup'])==s for r in sym) for s in sorted({system(r['spacegroup']) for r in sym})},
      'nops2':sum(r['nops']==2 for r in sym),'min_sites':min(len(r['atomic_numbers']) for r in rows),'max_sites':max(len(r['atomic_numbers']) for r in rows)}
    (ROOT/'work_results'/'RECOMPUTED_STATISTICS.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({t:{m:round(v['mean'],6) for m,v in out[t]['models'].items()} for t in ['formation_energy','band_gap','holdout']},indent=2))
    print('diagnostics',json.dumps(d,indent=2))
if __name__=='__main__':main()
