import final_train_evidence as e
from pathlib import Path
import json,time
import numpy as np
import torch
from torch import nn

ARMS=['sample_projection_masked','sample_projection_unmasked','sample_projection_unmasked_fullbatch']
def run():
    e.dump('DIAGNOSTIC_PROTOCOL.json',{'created_before_training':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
      'arms':['shared_projection_masked_batch256']+ARMS,'seeds':e.SEEDS[:3],
      'purpose':'sequential diagnostic of archived implementation choices, not reproduction of old absolute scores',
      'projection_seed':0,'projection_rule':'independent N(0,1/4) matrix for each crystal; one matrix fixed across model seeds',
      'test_selection':'never; validation-selected checkpoint','epochs':200,
      'limitations':'sequence-dependent effects; original random projection/split states unavailable; test cohort already examined'})
    ns=max(len(set(r['atomic_numbers'])) for r in e.rows)+1
    x=np.zeros((e.N,ns,64),np.float32); masked=np.zeros((e.N,ns),np.float32);unmasked=masked.copy()
    gen=torch.Generator().manual_seed(0)
    for i,r in enumerate(e.rows):
        zs,counts=np.unique(r['atomic_numbers'],return_counts=True);n=len(r['atomic_numbers'])
        x4=np.stack([zs/100,(zs%8)/8,(zs%18)/18,(np.searchsorted([2,10,18,36,54,86,118],zs)+1)/10],1).astype(np.float32)
        proj=(torch.randn(4,64,generator=gen)/2).numpy();x[i,:len(zs)]=x4@proj
        masked[i,:len(zs)]=counts/n;unmasked[i,:len(zs)]=counts/40
        unmasked[i,-1]=(40-n)/40
    x=torch.tensor(x,device=e.DEVICE)
    split=json.loads((e.WORK/'split_family.json').read_text());tr=torch.tensor(split['train_idx'],device=e.DEVICE);va=torch.tensor(split['val_idx'],device=e.DEVICE);te=torch.tensor(split['test_idx'],device=e.DEVICE)
    y=torch.tensor([r['target_eV'] for r in e.rows],dtype=torch.float32,device=e.DEVICE);mu=y[tr].mean();sd=y[tr].std()
    def pred(model,idx,weights):
        v=x[idx]
        for b in model.blocks:v=b.act(b.ln(v)@b.ol.weight[0]+b.ol.bias[0])
        return model.head((v*weights[idx,:,None]).sum(1)).squeeze(-1)
    for arm in ARMS:
        weights=torch.tensor(masked if arm.endswith('_masked') else unmasked,device=e.DEVICE)
        batch=len(tr) if arm.endswith('fullbatch') else 256
        for seed in e.SEEDS[:3]:
            key=f'diagnostic_{arm}_{seed}'
            if (e.WORK/(key+'.json')).exists():continue
            torch.manual_seed(seed);model=e.base.OrbitMLP(1).to(e.DEVICE)
            opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.05);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200)
            best=(float('inf'),None,-1);trace=[];start=time.time()
            for ep in range(200):
                model.train();perm=torch.randperm(len(tr))
                for j in range(0,len(tr),batch):
                    ix=tr[perm[j:j+batch].to(e.DEVICE)];loss=nn.functional.mse_loss(pred(model,ix,weights),(y[ix]-mu)/sd)
                    opt.zero_grad();loss.backward();opt.step()
                sch.step();model.eval()
                with torch.no_grad():v=float((pred(model,va,weights)*sd+mu-y[va]).abs().mean())
                trace.append(v)
                if v<best[0]:best=(v,{k:z.detach().clone() for k,z in model.state_dict().items()},ep+1)
            model.load_state_dict(best[1]);model.eval()
            with torch.no_grad():p=(pred(model,te,weights)*sd+mu).cpu().numpy()
            np.savez_compressed(e.PRED/(key+'.npz'),pred=p,y=y[te].cpu().numpy(),index=te.cpu().numpy())
            out={'arm':arm,'seed':seed,'test_mae':float(np.abs(p-y[te].cpu().numpy()).mean()),'best_epoch':best[2],
              'val_mae':best[0],'batch_size':batch,'val_trace':trace,'seconds':time.time()-start,'torch':torch.__version__,'device':e.DEVICE}
            e.dump(key+'.json',out); print(key,out['test_mae'],round(out['seconds'],1),flush=True)
if __name__=='__main__':run()
