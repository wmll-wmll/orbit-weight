from pathlib import Path
import sys, json, hashlib, platform, time
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import torch,spglib
from pymatgen.core import Lattice
OLD=ROOT
WORK=ROOT/'work_results'
WORK.mkdir(exist_ok=True)
def write(name,obj):
    (WORK/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def records():
    return [json.loads(s) for s in (OLD/'data'/'formation_energy_20260913.jsonl').read_text().splitlines()]
def main():
    rows=records(); N=len(rows)
    lab=np.load(OLD/'data'/'labels_20260913.npz')
    xe=np.load(OLD/'data'/'X_elem_20260913.npy')
    mask=np.load(OLD/'data'/'mask_20260913.npy')
    sym4=np.zeros((N,40),dtype=np.int64); exact=np.zeros_like(sym4)
    hierarchy=np.zeros((N,40,22),dtype=np.float32)
    mismatches=[]; nops=[]; symmetry=[]; rows4=[]; feature_delta=0.; old_distance_delta=[]
    rng=torch.Generator().manual_seed(0); proj=(torch.randn(4,64,generator=rng)/2).numpy()
    for i,r in enumerate(rows):
        z=np.array(r['atomic_numbers']); n=len(z)
        x4=np.stack([z/100,(z%8)/8,(z%18)/18,(np.searchsorted([2,10,18,36,54,86,118],z,side='left')+1)/10],1).astype(np.float32)
        feature_delta=max(feature_delta,float(np.max(np.abs(x4@proj-xe[i,:n]))))
        rows4.append(x4)
        cell=(np.array(r['lattice']),np.array(r['frac_coords']),z)
        ds=spglib.get_symmetry_dataset(cell,symprec=1e-3)
        if ds is None: raise RuntimeError(f'No symmetry {i}')
        eq=np.asarray(ds.equivalent_atoms); _, inv, counts=np.unique(eq,return_inverse=True,return_counts=True)
        mult=counts[inv]; ops=len(ds.rotations); stab=ops/mult
        if not np.allclose(stab,np.rint(stab)): raise RuntimeError('Noninteger stabilizer')
        stab=np.rint(stab).astype(int)
        bins=np.where(stab==1,0,np.where(stab==2,1,np.where(stab<=4,2,3)))
        sym4[i,:n]=bins; exact[i,:n]=inv
        # Label identity is crystallographic stabilizer order, not a local orbit ordinal.
        species_ok=all(len(set(z[inv==j]))==1 for j in range(len(counts)))
        if not species_ok: raise RuntimeError('Mixed species orbit')
        if ds.number!=r['spacegroup'] or ops!=r['n_sym_ops']:
            mismatches.append({'index':i,'id':r['material_id'],'stored_sg':r['spacegroup'],'recomputed_sg':int(ds.number),'stored_nops':r['n_sym_ops'],'recomputed_nops':ops})
        nops.append(ops)
        lat=Lattice(r['lattice']); dm=lat.get_all_distances(r['frac_coords'],r['frac_coords']); np.fill_diagonal(dm,np.inf)
        distances=np.sort(dm,axis=1)[:,:8]
        if distances.shape[1]<8: distances=np.pad(distances,((0,0),(0,8-distances.shape[1])),constant_values=8)
        distances[~np.isfinite(distances)]=8
        lf=np.array([*np.array(lat.abc)/10,*np.array(lat.angles)/180])
        hierarchy[i,:n,:4]=x4; hierarchy[i,:n,4:10]=lf
        hierarchy[i,:n,10:18]=distances/8; hierarchy[i,:n,18]=lat.volume/n/100
        hierarchy[i,:n,19]=mult/n; hierarchy[i,:n,20]=np.log2(mult)/6; hierarchy[i,:n,21]=np.log2(stab)/12
        # Recompute after atom permutation and an arbitrary origin translation.
        perm=np.random.default_rng(i).permutation(n)
        moved=(np.asarray(r['frac_coords'])[perm]+[.137,.271,.419])%1
        ds2=spglib.get_symmetry_dataset((cell[0],moved,z[perm]),symprec=1e-3)
        _,iv2,ct2=np.unique(ds2.equivalent_atoms,return_inverse=True,return_counts=True)
        st2=np.rint(len(ds2.rotations)/ct2[iv2]).astype(int)
        bn2=np.where(st2==1,0,np.where(st2==2,1,np.where(st2<=4,2,3)))
        symmetry.append({'index':i,'id':r['material_id'],'spacegroup':int(ds.number),'nops':ops,'multiplicity':mult.tolist(),'stabilizer':stab.tolist(),'symbols':list(ds.site_symmetry_symbols),'permutation_origin_labels_equal':bool(np.array_equal(bn2,bins[perm]))})
        if i%500==0: print('audit',i,flush=True)
    xh_old=np.load(OLD/'data'/'X_hier_20260913.npy')
    old_dist=np.abs(hierarchy[:,:,10:18]-xh_old[:,:,10:18])*mask[:,:,None]
    split=json.loads((OLD/'data'/'split_family_20260913.json').read_text())
    fams={k:{rows[i]['family_id'] for i in split[k+'_idx']} for k in ['train','val','test']}
    stats={'n':N,'python':platform.python_version(),'torch':torch.__version__,'spglib':spglib.__version__,
      'feature_projection_max_abs_delta':feature_delta,'recomputed_spacegroups':len(set(s['spacegroup'] for s in symmetry)),
      'symmetry_mismatches':mismatches,'nops_distribution':{str(v):int(np.sum(np.array(nops)==v)) for v in sorted(set(nops))},
      'sym4_counts':np.bincount(sym4[mask.astype(bool)],minlength=4).tolist(),
      'origin_and_permutation_label_pass':sum(s['permutation_origin_labels_equal'] for s in symmetry),
      'correct_periodic_distance_max_difference_A':float(old_dist.max()*8),
      'structures_with_distance_difference_gt_1e_3_A':int((old_dist.reshape(N,-1).max(1)*8>1e-3).sum()),
      'family_overlap':{a+'_'+b:len(fams[a]&fams[b]) for a,b in [('train','val'),('train','test'),('val','test')]},
      'data_sha256':hashlib.sha256((OLD/'data'/'formation_energy_20260913.jsonl').read_bytes()).hexdigest()}
    np.savez_compressed(WORK/'additional_features.npz',sym4=sym4,exact=exact,struct=hierarchy[:,:,:19],hierarchy=hierarchy)
    write('structure_audit.json',stats);write('site_symmetry_records.json',symmetry)
    print(json.dumps({k:v for k,v in stats.items() if k!='symmetry_mismatches'},indent=2)); print('mismatch count',len(mismatches))
if __name__=='__main__': main()
