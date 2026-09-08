#!/usr/bin/env python3
"""CGCNN under the SAME protocol as the MLP baselines:
80/20 split (random per seed), AdamW (lr 5e-4, wd 5e-2), CosineAnnealingLR, 200 epochs.
Dataset: data/rev_set (the 2,741 crystals matching the MLP set)."""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet


class Normalizer(object):
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

ROOT = r"E:\operator\cgcnn-master\data\rev_set"
OUT = r"E:\operator\resubmission_crystal_v2\experiments\cgcnn_matched.json"
SEEDS = [42, 123, 456]
EPOCHS = 200
BATCH = 256
D = {"atom_fea_len": 64, "n_conv": 3, "h_fea_len": 128, "n_h": 1}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mae(pred, target):
    return torch.abs(pred - target).mean().item()


def main():
    ds = CIFData(ROOT)
    N = len(ds)
    print(f"dataset {N} crystals, device {DEVICE}", flush=True)
    struct = ds[0][0]
    orig_atom_fea_len = struct[0].shape[-1]
    nbr_fea_len = struct[1].shape[-1]
    print(f"atom_fea_len={orig_atom_fea_len} nbr_fea_len={nbr_fea_len}", flush=True)

    results = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        idx = np.random.permutation(N)
        n_tr = int(N * 0.8)
        tr, te = idx[:n_tr], idx[n_tr:]
        tr_loader = DataLoader(ds, batch_size=BATCH, sampler=SubsetRandomSampler(tr),
                               collate_fn=collate_pool, num_workers=0)
        te_loader = DataLoader(ds, batch_size=BATCH, sampler=SubsetRandomSampler(te),
                               collate_fn=collate_pool, num_workers=0)

        # normalizer on train targets (sample)
        sample = tr[:500]
        _, t_sample, _ = collate_pool([ds[i] for i in sample])
        normalizer = Normalizer(t_sample)

        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=D["atom_fea_len"], n_conv=D["n_conv"],
                                    h_fea_len=D["h_fea_len"], n_h=D["n_h"])
        model = model.to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        crit = nn.MSELoss()

        best_mae, best_state = 1e9, None
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            for batch in tr_loader:
                (af, nf, nfi, cat), target, _ = batch
                af, nf, nfi = af.to(DEVICE), nf.to(DEVICE), nfi.to(DEVICE)
                cat = [c.to(DEVICE) for c in cat]
                target = (target - normalizer.mean) / normalizer.std
                target = target.to(DEVICE)
                out = model(af, nf, nfi, cat)
                loss = crit(out, target)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            model.eval()
            errs = []
            with torch.no_grad():
                for batch in te_loader:
                    (af, nf, nfi, cat), target, _ = batch
                    af, nf, nfi = af.to(DEVICE), nf.to(DEVICE), nfi.to(DEVICE)
                    cat = [c.to(DEVICE) for c in cat]
                    out = model(af, nf, nfi, cat)
                    pred = normalizer.denorm(out.cpu())
                    errs.append(torch.abs(pred - target).numpy())
            mae_test = float(np.concatenate(errs).mean())
            if mae_test < best_mae:
                best_mae = mae_test
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 25 == 0:
                print(f"  seed {seed} ep {ep+1} testMAE={mae_test:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        results[str(seed)] = {"test_mae_eV": best_mae, "best_epoch": None}
        print(f"** seed {seed} best test MAE = {best_mae:.4f} eV/atom", flush=True)
        torch.save({"model": best_state, "normalizer": {"mean": normalizer.mean,
                                                       "std": normalizer.std}}, 
                   os.path.join(os.path.dirname(OUT), f"cgcnn_best_{seed}.pt"))

    mu = float(np.mean([results[str(s)]["test_mae_eV"] for s in SEEDS]))
    sd = float(np.std([results[str(s)]["test_mae_eV"] for s in SEEDS], ddof=1))
    out = {"config": {"n_crystals": N, "device": DEVICE, "epochs": EPOCHS,
                      "protocol": "AdamW lr5e-4 wd5e-2 cosine 80/20",
                      "model": D, "seeds": SEEDS},
           "per_seed": results, "mae_mu_eV": mu, "mae_std_eV": sd}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"CGCNN matched: {mu:.4f} +/- {sd:.4f} eV/atom", flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
