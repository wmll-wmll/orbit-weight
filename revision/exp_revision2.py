#!/usr/bin/env python3
"""Revision extras: (a) strict space-group holdout transfer, (b) resource benchmark."""

import os, sys, json, pickle, time
import numpy as np
import torch

sys.path.insert(0, r"E:\operator")
from exp_decisive import (
    build_labels, build_X, make_projection, Model, count_params, train_model,
    D, L, EP, N_GRID, MAX_ATOMS, BATCH, SEEDS, DEVICE, CACHE,
)
from groups.octahedral import OctahedralGroup

OUT = r"E:\operator\resubmission_crystal_v2\experiments\revision_extra.json"


def main():
    with open(CACHE, "rb") as f:
        crystals = pickle.load(f)
    proj = make_projection(0)
    X = build_X(crystals, proj)
    oh = OctahedralGroup(N_GRID)
    oh_orbits, K_oh = oh.compute_orbits()
    K_oh = int(K_oh)
    K_pt = int(max(c["K_i"] for c in crystals))
    labels = build_labels(crystals, oh_orbits, K_oh)
    mask = labels["mask"]
    orbit_target = count_params(Model("orbit", K_oh))
    Y_fe = torch.tensor([c["formation_energy"] for c in crystals], dtype=torch.float32)
    sg = [int(c["spacegroup"]) for c in crystals]
    mu, std = Y_fe.mean(), Y_fe.std().clamp(min=1e-8)
    Yf = (Y_fe - mu) / std

    out = {"config": {"K_oh": K_oh, "K_pt": K_pt, "orbit_target_params": orbit_target,
                      "n_crystals": len(crystals), "device": DEVICE,
                      "fe_mu": float(mu), "fe_std": float(std)}}

    # ── (a) strict space-group holdout transfer ──────────────
    from collections import Counter
    counts = Counter(sg)
    # hold out least-frequent space groups until test >= ~100 and >= 12 groups
    ordered = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    test_sgs, n_test = [], 0
    for g, c in ordered:
        test_sgs.append(g)
        n_test += c
        if n_test >= 100 and len(test_sgs) >= 12:
            break
    test_sgs = set(test_sgs)
    train_i = [i for i, g in enumerate(sg) if g not in test_sgs]
    test_i = [i for i, g in enumerate(sg) if g in test_sgs]
    print(f"[transfer] held out {len(test_sgs)} space groups, "
          f"{len(test_i)} test / {len(train_i)} train crystals", flush=True)

    transfer = {}
    for kind, K, needs_o in [("standard", 1, False), ("wide", orbit_target, False),
                             ("pt", K_pt, True)]:
        per_seed = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = Model("orbit" if needs_o else ("wide" if kind == "wide" else "standard"), K)
            # strict split: train on train_i, eval on test_i (fixed by SG, independent of seed)
            Xtr, ytr, Mtr = X[train_i].to(DEVICE), Yf[train_i].to(DEVICE), mask[train_i].to(DEVICE)
            Xte, yte, Mte = X[test_i].to(DEVICE), Yf[test_i].to(DEVICE), mask[test_i].to(DEVICE)
            Otr = Ote = None
            if needs_o:
                Otr = labels["pt"][train_i].to(DEVICE)
                Ote = labels["pt"][test_i].to(DEVICE)
            mae, _ = train_model(model, X.cpu()[train_i],
                                 None if not needs_o else labels["pt"][train_i],
                                 mask[train_i], Yf[train_i], X.cpu()[test_i],
                                 None if not needs_o else labels["pt"][test_i],
                                 mask[test_i], Yf[test_i], std)
            per_seed.append(mae)
        transfer[kind] = {"mae_per_seed_eV": per_seed,
                          "mae_mu_eV": float(np.mean(per_seed)),
                          "params": count_params(Model(
                              "orbit" if needs_o else ("wide" if kind == "wide" else "standard"), K))}
        print(f"[transfer] {kind:9s} per-seed {['%.4f' % v for v in per_seed]} "
              f"mu={np.mean(per_seed):.4f} eV/atom", flush=True)
    out["strict_sg_transfer"] = transfer
    out["strict_sg_transfer"]["held_out_space_groups"] = sorted(test_sgs)

    # ── (b) resource benchmark ───────────────────────────────
    res = {}
    for kind, K, needs_o in [("standard", 1, False), ("pt", K_pt, True), ("oh", K_oh, True)]:
        model = Model("orbit" if needs_o else "standard", K).to(DEVICE)
        # one representative (B,MAX,D) batch
        bx = torch.randn(BATCH, MAX_ATOMS, D, device=DEVICE)
        o = torch.randint(0, K if needs_o else 1, (BATCH, MAX_ATOMS), device=DEVICE)
        bm = torch.ones(BATCH, MAX_ATOMS, device=DEVICE)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        # warmup then time 20 steps
        for _ in range(5):
            p = model(bx, o if needs_o else None, bm)
            loss = p.mean(); opt.zero_grad(); loss.backward(); opt.step()
        t0 = time.perf_counter()
        for _ in range(20):
            p = model(bx, o if needs_o else None, bm)
            loss = p.mean(); opt.zero_grad(); loss.backward(); opt.step()
        step_t = (time.perf_counter() - t0) / 20
        # inference latency
        model.eval()
        with torch.no_grad():
            p = model(bx, o if needs_o else None, bm)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                model(bx, o if needs_o else None, bm)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            infer_t = (time.perf_counter() - t0) / 50
        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
        res[kind] = {"params": count_params(model),
                     "train_step_s": round(step_t, 4),
                     "infer_step_ms": round(infer_t * 1000, 3),
                     "peak_gpu_mem_MB": round(peak_mem, 1),
                     "per_100_batch_infer_ms": round(infer_t * 100, 2)}
        print(f"[res] {kind:9s} train_step={step_t:.4f}s infer={infer_t*1000:.3f}ms "
              f"peakGB={peak_mem/1000:.2f}", flush=True)

    # orbit-assign preprocessing cost (per crystal)
    t0 = time.perf_counter()
    _ = build_labels(crystals, oh_orbits, K_oh)
    assign_s = time.perf_counter() - t0
    out["resource"] = res
    out["resource"]["orbit_assign_total_s_all_crystals"] = round(assign_s, 4)
    out["resource"]["orbit_assign_s_per_crystal"] = round(assign_s / len(crystals), 6)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
