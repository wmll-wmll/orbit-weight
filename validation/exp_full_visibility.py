"""
Supplementary Table 1: Full-visibility results (0% masking).

Without position occlusion, all models with sufficient parameter capacity
achieve high accuracy. This confirms that the value of algebraic structure
emerges under challenging conditions (high occlusion).

Usage:
    python validation/exp_full_visibility.py
"""
import torch, torch.nn as nn, numpy as np, random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groups.base import OrbitLinear, compute_orbits
from cube.cube3d import CubePermutations
from groups.octahedral import OctahedralGroup

D = 64; L = 4; EP = 80; N_TRAIN = 400; N_TEST = 200


def run_group(name, N, generators, device="cuda"):
    oids, K = compute_orbits(N, generators)
    torch.manual_seed(42)
    proto = torch.randn(K, D) / (D ** 0.5)

    # All positions visible
    d_tr = torch.zeros(N_TRAIN, N, D); l_tr = torch.zeros(N_TRAIN, N, dtype=torch.long)
    for s in range(N_TRAIN):
        for i in range(N):
            oid = int(oids[i])
            d_tr[s, i] = proto[oid] + 0.3 * torch.randn(D)
        l_tr[s] = oids
    d_te = torch.zeros(N_TEST, N, D); l_te = torch.zeros(N_TEST, N, dtype=torch.long)
    for s in range(N_TEST):
        for i in range(N):
            oid = int(oids[i])
            d_te[s, i] = proto[oid] + 0.3 * torch.randn(D)
        l_te[s] = oids

    d_tr = d_tr.to(device); l_tr = l_tr.to(device)
    d_te = d_te.to(device); l_te = l_te.to(device)

    def make_model(model_type):
        if model_type == "orbit":
            layers = []
            for _ in range(L):
                layers.extend([nn.LayerNorm(D), OrbitLinear(oids, K, D), nn.GELU()])
            return nn.Sequential(*layers)
        elif model_type == "random":
            rid = torch.randperm(N) % max(K, 2)
            nr = int(rid.max().item()) + 1
            layers = []
            for _ in range(L):
                layers.extend([nn.LayerNorm(D), OrbitLinear(rid, nr, D), nn.GELU()])
            return nn.Sequential(*layers)
        else:
            layers = []
            for _ in range(L):
                layers.extend([nn.LayerNorm(D), nn.Linear(D, D), nn.GELU()])
            return nn.Sequential(*layers)

    def train(model_type):
        vals = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
            model = make_model(model_type).to(device)
            head = nn.Linear(D, K).to(device)
            opt = torch.optim.AdamW(
                list(model.parameters()) + list(head.parameters()),
                lr=1e-3, weight_decay=0.01)
            for _ in range(EP):
                model.train(); head.train(); opt.zero_grad()
                logits = head(model(d_tr))
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, K), l_tr.reshape(-1))
                loss.backward(); opt.step()
            model.eval(); head.eval()
            with torch.no_grad():
                pred = head(model(d_te)).argmax(-1)
                acc = (pred == l_te).float().mean().item()
            vals.append(acc)
            del model, head, opt
            if device == "cuda": torch.cuda.empty_cache()
        return np.mean(vals), np.std(vals, ddof=1)

    orb_mean, orb_std = train("orbit")
    rand_mean, rand_std = train("random")
    uni_mean, uni_std = train("uniform")
    params_orbit = K * D * D + K * D

    print(f"  {name}: Orbit={orb_mean*100:.1f}% Random={rand_mean*100:.1f}% "
          f"Uniform={uni_mean*100:.1f}%")
    return {
        "N": N, "K": K,
        "orbit": (orb_mean, orb_std),
        "random": (rand_mean, rand_std),
        "uniform": (uni_mean, uni_std),
    }


def run(device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 65)
    print("SUPPLEMENTARY TABLE 1: FULL VISIBILITY (0% MASKING)")
    print(f"D={D}, layers={L}, epochs={EP}, train={N_TRAIN}, test={N_TEST}")
    print("=" * 65)

    print("\nCube(5) N=125 K=48")
    cube_gens = list(CubePermutations(5).all_generators().values())
    cube_r = run_group("Cube(5)", 125, cube_gens, device)

    print("\nO_h(3) N=27 K=15")
    oh_gens = OctahedralGroup(3).get_generators()
    oh_r = run_group("O_h(3)", 27, oh_gens, device)

    print("\n" + "=" * 65)
    print("SUMMARY (Supplementary Table 1 format)")
    print("=" * 65)
    print(f"{'Group':<10s} {'N':>4s} {'K':>4s} {'OrbitMLP':>12s} "
          f"{'RandomMLP':>12s} {'UniformMLP':>12s}")
    print("-" * 58)
    for name, r in [("Cube(5)", cube_r), ("O_h(3)", oh_r)]:
        o, os = r["orbit"]; rd, rs = r["random"]; u, us = r["uniform"]
        print(f"{name:<10s} {r['N']:>4d} {r['K']:>4d} "
              f"{o*100:>10.1f}% {'':>1s} {rd*100:>10.1f}% {'':>1s} {u*100:>10.1f}%")

    return cube_r, oh_r


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(device)
