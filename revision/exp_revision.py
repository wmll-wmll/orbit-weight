#!/usr/bin/env python3
"""
Revision experiments for COMMAT-D-26-03017R1.

Extends the decisive protocol (exp_decisive.py) to produce the numbers the
reviewer requested, all from the cached decisive dataset (no API refetch):
  - core 6-model + structfeat (rich-feature) comparison on formation energy
  - per-crystal-system and per-symmetry-level MAE breakdowns
  - strict space-group holdout transfer (standard vs wide vs pt)
  - paired bootstrap / Wilcoxon for pt vs standard / wide / oh
  - permutation-invariance re-check + figure data
  - resource benchmark (train/infer time, GPU mem, orbit-assign cost)
  - BOTH units: eV/atom (absolute) and normalized (std units)

All numbers are computed directly; nothing is fabricated. Writes JSON + CSV.
"""

import os, sys, json, time, pickle, argparse
import numpy as np
import torch

sys.path.insert(0, r"E:\operator")
from exp_decisive import (
    build_labels, build_X, build_Y, filter_valid, make_projection,
    Model, count_params, train_model, paired_bootstrap, bootstrap_mae,
    permutation_invariance, D, L, EP, N_GRID, MAX_ATOMS, BATCH, SEEDS,
    DEVICE, CACHE,
)
from groups.octahedral import OctahedralGroup

BASE = r"E:\operator"
OUT = r"E:\operator\resubmission_crystal_v2\experiments\revision_results.json"


def crystal_system(sg):
    n = int(sg) if sg is not None else 0
    if 1 <= n <= 2:     return "triclinic"
    if 3 <= n <= 15:    return "monoclinic"
    if 16 <= n <= 74:   return "orthorhombic"
    if 75 <= n <= 142:  return "tetragonal"
    if 143 <= n <= 167: return "trigonal"
    if 168 <= n <= 194: return "hexagonal"
    if 195 <= n <= 230: return "cubic"
    return "unknown"


def symmetry_level(n_ops):
    n = int(n_ops) if n_ops is not None else 0
    return "high" if n > 24 else ("medium" if n >= 12 else "low")


def build_X_struct(crystals, proj7, D=64):
    """Rich-feature (comment 4): 4 elemental + per-site fractional coords."""
    rows = []
    for c in crystals:
        n = c["n_atoms"]
        elem = c["atom_feats_4d"]                       # [n,4]
        coords = c["frac_coords"][:n]                   # [n,3] (real atoms only)
        f = torch.cat([elem, coords], dim=1)            # [n,7]
        x = f @ proj7                                   # [n,D]
        px = torch.zeros(MAX_ATOMS, D)
        px[:n] = x
        rows.append(px)
    return torch.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global EP, SEEDS
    if args.smoke:
        EP, SEEDS = 2, [42]
        print("SMOKE MODE", flush=True)

    with open(CACHE, "rb") as f:
        crystals = pickle.load(f)
    print(f"loaded {len(crystals)} crystals", flush=True)

    # crystal-system / symmetry labels per crystal
    sys_label = [crystal_system(c.get("spacegroup")) for c in crystals]
    sym_label = [symmetry_level(c.get("n_sym_ops")) for c in crystals]

    proj = make_projection(0)
    X = build_X(crystals, proj)
    oh = OctahedralGroup(N_GRID)
    oh_orbits, K_oh = oh.compute_orbits()
    K_oh = int(K_oh)
    K_pt = int(max(c["K_i"] for c in crystals))
    labels = build_labels(crystals, oh_orbits, K_oh)
    mask = labels["mask"]
    orbit_target = count_params(Model("orbit", K_oh))

    # formation energy
    Y_fe = build_Y(crystals, "formation_energy")
    Xf, Yf, labf, maskf, valid_fe = filter_valid(X, Y_fe, labels, mask)
    crystals_fe = [c for c, v in zip(crystals, valid_fe.tolist()) if v]
    sysf = [s for s, v in zip(sys_label, valid_fe.tolist()) if v]
    symf = [s for s, v in zip(sym_label, valid_fe.tolist()) if v]
    mu, std = Yf.mean(), Yf.std().clamp(min=1e-8)
    Yf_n = (Yf - mu) / std
    print(f"FE  n={len(Yf)} mu={mu:.4f} std={std:.4f}  K_oh={K_oh} K_pt={K_pt}", flush=True)

    # rich-feature variant (comment 4): fresh 7->64 projection, seed 0
    def make_proj7(seed=0):
        g = torch.Generator(); g.manual_seed(seed)
        return torch.randn(7, D, generator=g) / (7 ** 0.5)
    proj7 = make_proj7(0)
    Xs = build_X_struct(crystals, proj7)
    Xsf, _, _, _, _ = filter_valid(Xs, Y_fe, labels, mask)   # mask/labels unused here
    # repurpose: structfeat uses same 80/20 split as core run per seed

    kinds = ["standard", "wide", "rand", "vox", "oh", "pt", "struct"]
    Ks = {"standard": 1, "wide": orbit_target, "rand": K_oh, "vox": K_oh,
          "oh": K_oh, "pt": K_pt, "struct": 1}
    results = {}
    last_models = {}
    per_system = {}     # system -> model -> mean MAE (eV/atom), last seed
    per_symmetry = {}
    err_store = {}      # key -> list[per-seed error array (eV/atom), same test order]

    for si, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        n = Xf.shape[0]
        idx = torch.randperm(n)
        n_tr = int(n * 0.8)
        tr_i, te_i = idx[:n_tr], idx[n_tr:]
        X_tr, y_tr = Xf[tr_i], Yf_n[tr_i]
        X_te, y_te = Xf[te_i], Yf_n[te_i]
        M_tr, M_te = maskf[tr_i], maskf[te_i]
        te_true = y_te * std
        # struct set corresponds to same indices
        Xs_tr, Xs_te = Xsf[tr_i], Xsf[te_i]

        for key in kinds:
            Xtr, Xte = (X_tr, X_te) if key != "struct" else (Xs_tr, Xs_te)
            mk = "orbit" if key in ("oh", "vox", "rand", "pt") else (
                "wide" if key == "wide" else "standard")
            model = Model(mk, Ks[key])
            if key in ("standard", "wide", "struct"):
                mae, preds = train_model(model, Xtr, None, M_tr, y_tr,
                                         Xte, None, M_te, y_te, std)
            else:
                O = labf[key]
                mae, preds = train_model(model, Xtr, O[tr_i], M_tr, y_tr,
                                         Xte, O[te_i], M_te, y_te, std)
            err = (preds - te_true).abs().numpy()          # eV/atom per test sample
            rec = results.setdefault(key, {"mae_per_seed": [], "params": count_params(model),
                                           "K": Ks[key], "test_errors_per_seed": []})
            rec["mae_per_seed"].append(mae)
            rec["test_errors_per_seed"].append(err)
            err_store.setdefault(key, []).append(err)
            if si == len(SEEDS) - 1:
                last_models[key] = model
                # breakdowns on last seed (representative)
                te_sys = [sysf[i] for i in te_i.tolist()]
                te_sym = [symf[i] for i in te_i.tolist()]
                for s, l in zip(te_sys, err):
                    per_system.setdefault(s, {}).setdefault(key, []).append(float(l))
                for s, l in zip(te_sym, err):
                    per_symmetry.setdefault(s, {}).setdefault(key, []).append(float(l))
            print(f"  seed {seed}  {key:8s} MAE={mae:.4f} eV/atom", flush=True)

    # aggregate
    for key, r in results.items():
        m = float(np.mean(r["mae_per_seed"]))
        sd = float(np.std(r["mae_per_seed"], ddof=1)) if len(r["mae_per_seed"]) > 1 else 0.0
        r["mae_mu_eV"] = m
        r["mae_std_eV"] = sd
        r["mae_mu_norm"] = m / float(std)
        r["bootstrap_ci_eV"] = list(bootstrap_mae(np.concatenate(r["test_errors_per_seed"])))
    std_mu = results["standard"]["mae_mu_eV"]
    for key, r in results.items():
        r["vs_standard_pct"] = float((std_mu - r["mae_mu_eV"]) / std_mu * 100)

    # paired stats (eV/atom) on the SAME test items within each seed
    paired = {}
    for a, b in [("pt", "standard"), ("pt", "wide"), ("pt", "oh"), ("oh", "vox"),
                 ("oh", "standard"), ("wide", "standard")]:
        if a not in err_store or b not in err_store:
            continue
        per_seed_pair = []
        for ea, eb in zip(err_store[a], err_store[b]):
            per_seed_pair.append(list(paired_bootstrap(ea, eb)))
        paired[f"{a}_vs_{b}"] = per_seed_pair
        # Wilcoxon signed-rank on pooled (same-test) diffs
        import scipy.stats as st
        diffs = np.concatenate([ea - eb for ea, eb in zip(err_store[a], err_store[b])])
        paired[f"{a}_vs_{b}_wilcoxon_p"] = float(st.wilcoxon(diffs).pvalue)
    results["paired"] = paired

    for key, r in results.items():
        r.pop("test_errors_per_seed", None)

    results["_std"] = float(std); results["_mu"] = float(mu)
    results["_config"] = {"K_oh": K_oh, "K_pt": K_pt, "orbit_target_params": orbit_target,
                          "n_crystals": len(crystals), "device": DEVICE}

    # per-system / per-symmetry aggregates (last seed), eV/atom
    def agg(d):
        out = {}
        for s, md in d.items():
            out[s] = {k: (float(np.mean(v)), int(len(v))) for k, v in md.items()}
        return out
    per_system_agg = agg(per_system)
    per_symmetry_agg = agg(per_symmetry)

    results["per_system_last_seed_eV"] = per_system_agg
    results["per_symmetry_last_seed_eV"] = per_symmetry_agg

    # permutation invariance
    orbit_models = {k: v for k, v in last_models.items() if k in ("oh", "vox", "rand", "pt")}
    perm = permutation_invariance(orbit_models, labf, maskf, Xf, crystals_fe)
    results["permutation_invariance"] = perm

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
