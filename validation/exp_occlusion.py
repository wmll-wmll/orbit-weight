"""
Table 1: Position occlusion experiment.

Tests whether orbit weight sharing enables generalization from partially
observed inputs — the signature of gradient sharing through correct
algebraic structure.

Groups:  Cube(5)  N=125  K=48   (face rotation group)
         O_h(3)   N=27   K=15   (octahedral group)

For each mask level (50%, 70%, 90%), we train OrbitMLP, RandomMLP, and
UniformMLP for 150 epochs, 3 seeds each, and report mean ± s.d.

Usage:
    python validation/exp_occlusion.py
"""
import torch, torch.nn as nn, numpy as np, random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groups.base import OrbitLinear, compute_orbits
from cube.cube3d import CubePermutations
from groups.octahedral import OctahedralGroup

D = 64; L = 4; EP = 150; N_TRAIN = 400; N_TEST = 200


def run_group(name, N, generators, mask_levels, device="cuda"):
    oids, K = compute_orbits(N, generators)
    torch.manual_seed(42)
    proto = torch.randn(K, D) / (D ** 0.5)

    # Test data: all positions visible
    d_te = torch.zeros(N_TEST, N, D); l_te = torch.zeros(N_TEST, N, dtype=torch.long)
    for s in range(N_TEST):
        for i in range(N):
            oid = int(oids[i])
            d_te[s, i] = proto[oid] + 0.3 * torch.randn(D)
        l_te[s] = oids
    d_te = d_te.to(device); l_te = l_te.to(device)

    results = {}
    for mask_frac in mask_levels:
        torch.manual_seed(1234)
        visible = torch.rand(N) > mask_frac
        n_vis = int(visible.sum().item())

        # Train data with masked positions
        d_tr = torch.zeros(N_TRAIN, N, D); l_tr = torch.zeros(N_TRAIN, N, dtype=torch.long)
        for s in range(N_TRAIN):
            for i in range(N):
                oid = int(oids[i])
                d_tr[s, i] = proto[oid] + 0.3 * torch.randn(D) if visible[i] else torch.zeros(D)
            l_tr[s] = oids
        d_tr = d_tr.to(device); l_tr = l_tr.to(device)

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
            else:  # uniform
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
        delta = orb_mean - rand_mean

        results[mask_frac] = {
            "visible": n_vis, "N": N,
            "orbit": (orb_mean, orb_std),
            "random": (rand_mean, rand_std),
            "uniform": (uni_mean, uni_std),
            "delta": delta,
        }
        print(f"  {name} mask={int(mask_frac*100):>2d}% ({n_vis}/{N} vis): "
              f"Orbit={orb_mean*100:.1f}±{orb_std*100:.1f}% "
              f"Random={rand_mean*100:.1f}±{rand_std*100:.1f}% "
              f"Uniform={uni_mean*100:.1f}±{uni_std*100:.1f}% "
              f"O-R={delta*100:+.1f}pp")
    return results


def run(device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 65)
    print("TABLE 1: POSITION OCCLUSION EXPERIMENT")
    print(f"D={D}, layers={L}, epochs={EP}, train={N_TRAIN}, test={N_TEST}")
    print("=" * 65)

    mask_levels = [0.5, 0.7, 0.9]

    print("\nCube(5) N=125 K=48")
    cube_gens = list(CubePermutations(5).all_generators().values())
    cube_results = run_group("Cube(5)", 125, cube_gens, mask_levels, device)

    print("\nO_h(3) N=27 K=15")
    oh_gens = OctahedralGroup(3).get_generators()
    oh_results = run_group("O_h(3)", 27, oh_gens, mask_levels, device)

    print("\n" + "=" * 65)
    print("SUMMARY (Table 1 format)")
    print("=" * 65)
    print(f"{'Mask':>5s} {'Visible':>8s} {'OrbitMLP':>12s} {'RandomMLP':>12s} "
          f"{'UniformMLP':>12s} {'O-R(pp)':>8s}")
    print("-" * 60)

    for label, results in [("Cube(5) K=48", cube_results), ("O_h(3) K=15", oh_results)]:
        print(f"  {label}")
        for mf in mask_levels:
            r = results[mf]
            o, os = r["orbit"]; rd, rs = r["random"]; u, us = r["uniform"]
            print(f"  {int(mf*100):>2d}% {r['visible']:>4d}/{r['N']:<4d} "
                  f"{o*100:>10.1f}±{os*100:.1f}% {rd*100:>10.1f}±{rs*100:.1f}% "
                  f"{u*100:>10.1f}±{us*100:.1f}% {r['delta']*100:+8.1f}")

    return cube_results, oh_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(device)
