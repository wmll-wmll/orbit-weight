"""Frozen-protocol follow-up. No choice of model or hyperparameters uses test errors."""
from pathlib import Path
import sys, json, time, hashlib, importlib.util, platform, argparse
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work_results'
WORK.mkdir(exist_ok=True)
import numpy as np
import torch
from torch import nn
import models as base
OLD=ROOT
for name in ['split_family.json','split_spacegroup.json']:
    if not (WORK/name).exists(): (WORK/name).write_bytes((ROOT/'data'/name).read_bytes())
torch.set_num_threads(4)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
SEEDS=[42,123,456,789,1024]
MODELS=['standard','wide','cube4','geo4','occupancy4','random4','sym4','struct','hierarchy']
PRED=WORK/'predictions'; CKPT=WORK/'checkpoints'; PRED.mkdir(exist_ok=True);CKPT.mkdir(exist_ok=True)
def dump(name,obj): (WORK/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def loadrows(task): return [json.loads(s) for s in (OLD/'data'/f'{task}_20260913.jsonl').read_text().splitlines()]
rows=loadrows('formation_energy'); N=len(rows)
mask=np.load(OLD/'data'/'mask_20260913.npy'); elem=np.load(OLD/'data'/'X_elem_20260913.npy')
new=np.load(ROOT/'data'/'additional_features.npz'); labels=dict(np.load(OLD/'data'/'labels_20260913.npz'));labels['sym4']=new['sym4']
zs=sorted(set(z for r in rows for z in r['atomic_numbers'])); zindex={z:i for i,z in enumerate(zs)}
proto=np.zeros((len(zs),64),np.float32)
for i,r in enumerate(rows):
    for j,z in enumerate(r['atomic_numbers']): proto[zindex[z]]=elem[i,j]
proto=torch.tensor(proto,device=DEVICE)
def histogram(lab,K):
    h=np.zeros((N,K*len(zs)),np.float32)
    for i,r in enumerate(rows):
        for j,z in enumerate(r['atomic_numbers']): h[i,int(lab[i,j])*len(zs)+zindex[z]]+=1/len(r['atomic_numbers'])
    return h
features={m:histogram(labels[m],4) for m in ['cube4','geo4','occupancy4','random4','sym4']}
features['standard']=histogram(np.zeros((N,40),int),1);features['wide']=features['standard']
features['struct']=new['struct'];features['hierarchy']=new['hierarchy']

class HistogramModel(nn.Module):
    def __init__(self,name):
        super().__init__(); self.name=name; self.K=1 if name in ['standard','wide'] else 4
        self.base=base.WideMLP(base.find_wide_H(69185)[0]) if name=='wide' else base.OrbitMLP(self.K)
    def forward(self,h):
        if self.name=='wide':
            phi=proto
            for block in self.base.blocks:phi=block(phi)
        else:
            phi=proto.unsqueeze(0).expand(self.K,-1,-1)
            for block in self.base.blocks:
                phi=block.act(torch.bmm(block.ln(phi),block.ol.weight)+block.ol.bias[:,None,:])
            phi=phi.reshape(-1,64)
        return self.base.head(h@phi).squeeze(-1)

def make(name):
    return base.FeatMLP(19 if name=='struct' else 22) if name in ['struct','hierarchy'] else HistogramModel(name)
def forward(model,name,x,m):
    return model(x,None,m) if name in ['struct','hierarchy'] else model(x)

def train(name,task,seed,split,tag='',diagnostic=False):
    key=f'{tag}{task}_{name}_{seed}'; outfile=WORK/(key+'.json')
    if outfile.exists(): print('skip',key,flush=True);return json.loads(outfile.read_text())
    torch.manual_seed(seed);np.random.seed(seed)
    model=make(name).to(DEVICE)
    x=torch.tensor(features[name],device=DEVICE); m=torch.tensor(mask,device=DEVICE)
    y=torch.tensor([r['target_eV'] for r in loadrows(task)],dtype=torch.float32,device=DEVICE)
    tr=torch.tensor(split['train_idx'],device=DEVICE);va=torch.tensor(split['val_idx'],device=DEVICE);te=torch.tensor(split['test_idx'],device=DEVICE)
    mu=y[tr].mean();sd=y[tr].std().clamp(min=1e-8)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200)
    best=(float('inf'),None,-1);trace=[];start=time.time()
    for ep in range(200):
        model.train();perm=torch.randperm(len(tr))
        losses=[]
        for j in range(0,len(tr),256):
            idx=tr[perm[j:j+256].to(DEVICE)]; pred=forward(model,name,x[idx],m[idx]);loss=nn.functional.mse_loss(pred,(y[idx]-mu)/sd)
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        sch.step();model.eval()
        with torch.no_grad():vmae=float((forward(model,name,x[va],m[va])*sd+mu-y[va]).abs().mean())
        trace.append({'epoch':ep+1,'train_mse_normalized':float(np.mean(losses)),'val_mae':vmae})
        if vmae<best[0]:best=(vmae,{k:v.detach().clone() for k,v in model.state_dict().items()},ep+1)
    model.load_state_dict(best[1]);model.eval()
    with torch.no_grad():pred=(forward(model,name,x[te],m[te])*sd+mu).cpu().numpy()
    np.savez_compressed(PRED/(key+'.npz'),pred=pred,y=y[te].cpu().numpy(),index=te.cpu().numpy())
    torch.save({'model':model.state_dict(),'mu':float(mu),'sd':float(sd),'name':name},CKPT/(key+'.pt'))
    result={'name':name,'task':task,'seed':seed,'batch_size':256,'epochs':200,'selected_epoch':best[2],
      'optimizer':'AdamW','lr':.0005,'weight_decay':.05,'selection':'minimum validation MAE','test_evaluations':1,
      'test_mae':float(np.abs(pred-y[te].cpu().numpy()).mean()),'val_mae':best[0],
      'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
      'fixed_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad),
      'mean_train':float(mu),'sd_train':float(sd),'seconds':time.time()-start,'trace':trace,
      'device':DEVICE,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if DEVICE=='cuda' else None}
    dump(key+'.json',result);print(key,result['test_mae'],round(result['seconds'],1),flush=True)
    return result

def design():
    split=json.loads((OLD/'data'/'split_family_20260913.json').read_text())
    sym=json.loads((ROOT/'data'/'site_symmetry_records.json').read_text());sg=np.array([r['spacegroup'] for r in sym])
    unique,counts=np.unique(sg,return_counts=True); ordered=sorted(zip(counts,unique)); chosen=[];n=0
    for c,g in ordered:
        chosen.append(int(g));n+=int(c)
        if n>=100:break
    test=np.flatnonzero(np.isin(sg,chosen)).tolist();testfamilies={rows[i]['family_id'] for i in test}
    rest=[i for i in range(N) if i not in set(test) and rows[i]['family_id'] not in testfamilies]
    families=sorted({rows[i]['family_id'] for i in rest});order=np.random.default_rng(0).permutation(len(families))
    vaf={families[i] for i in order[:max(1,int(len(families)*.1))]}
    transfer={'train_idx':[i for i in rest if rows[i]['family_id'] not in vaf],'val_idx':[i for i in rest if rows[i]['family_id'] in vaf],'test_idx':test,'holdout_spacegroups':chosen}
    dump('split_family.json',split);dump('split_spacegroup.json',transfer)
    protocol={'created_before_training':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'models':MODELS,'formation_seeds':SEEDS,
     'band_gap_models':['standard','wide','cube4','sym4'],'band_gap_seeds':SEEDS[:3],
     'transfer_models':['standard','wide','cube4','sym4'],'transfer_seeds':SEEDS[:3],
     'sym4_rule':'site stabilizer order: 1 | 2 | 3-4 | >=6; species-preserving spglib orbits, symprec=1e-3 A',
     'primary_pairwise_contrasts':[['cube4',m] for m in ['standard','wide','geo4','occupancy4','random4','sym4']]+[['sym4','standard'],['struct','standard'],['hierarchy','struct']],
     'no_hyperparameter_search':True,'post_review_supplemental_study':True,'existing_test_data_previously_examined':True,
     'inference_scope':'conditional on fixed cohort and split; supplemental results not an untouched external replication',
     'batch_size':256,'epochs':200,'device':DEVICE,'torch':torch.__version__,
     'source_data_sha256':hashlib.sha256((OLD/'data'/'formation_energy_20260913.jsonl').read_bytes()).hexdigest()}
    dump('PROTOCOL_BEFORE_TRAINING.json',protocol)
    # Equivalence verification uses no targets and happens before training.
    audit={}
    for name in ['standard','cube4','wide']:
        torch.manual_seed(42);model=make(name).to(DEVICE).eval(); idx=np.arange(37)
        with torch.no_grad():
            a=model(torch.tensor(features[name][idx],device=DEVICE));o=None if name=='wide' else torch.tensor(np.zeros((37,40),int) if name=='standard' else labels[name][idx],device=DEVICE)
            b=model.base(torch.tensor(elem[idx],device=DEVICE),o,torch.tensor(mask[idx],device=DEVICE))
        audit[name]=float((a-b).abs().max())
        assert audit[name]<1e-5
    dump('histogram_equivalence.json',audit)
    print('protocol frozen',audit,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['design','main','gap','transfer']);args=ap.parse_args()
    if args.stage=='design':design()
    else:
        split=json.loads((WORK/('split_spacegroup.json' if args.stage=='transfer' else 'split_family.json')).read_text())
        task='band_gap' if args.stage=='gap' else 'formation_energy'
        names=MODELS if args.stage=='main' else ['standard','wide','cube4','sym4']
        for name in names:
            for seed in (SEEDS if args.stage=='main' else SEEDS[:3]):train(name,task,seed,split,tag='holdout_' if args.stage=='transfer' else '')
