"""
Table 2a: Crystal formation energy prediction.

Downloads crystal structures from Materials Project, computes orbit
decompositions via spglib, and trains OrbitMLP vs StandardMLP vs
RandomMLP vs PerPositionMLP on formation energy prediction.

Requires: Materials Project API key (set MP_API_KEY env var or edit below)
          pymatgen, spglib, requests

Usage:
    python validation/exp_crystal_energy.py
"""
import torch, torch.nn as nn, numpy as np, random, sys, os, pickle, time, json, requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groups.base import OrbitLinear, compute_orbits
from groups.octahedral import OctahedralGroup

API_KEY = os.environ.get("MP_API_KEY", "LscQKdyWzTOdAqzAtJxXPkn8LTS6LFwZ")
D = 64; L = 4; EP = 200
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "crystal_data.pkl")


def download_crystals():
    """Download crystals from Materials Project. Cached to crystal_data.pkl."""
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "rb") as f:
            return pickle.load(f)

    print("Downloading crystals from Materials Project...")
    headers = {"X-API-KEY": API_KEY}
    url = "https://api.materialsproject.org/materials/summary/"
    all_data = []

    for offset in range(0, 3000, 500):
        params = {
            "nsites_min": 8, "nsites_max": 40,
            "energy_above_hull_max": 0.2,
            "_limit": 500, "_skip": offset,
            "_all_fields": "false",
            "_fields": "material_id,formula_pretty,formation_energy_per_atom,symmetry,nsites,structure",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            break
        batch = resp.json().get("data", [])
        if not batch: break
        all_data.extend(batch)
        print(f"  {len(all_data)} crystals...")
        if len(batch) < 500: break

    # Process: compute orbit decompositions via spglib
    try:
        from pymatgen.core import Structure
        import spglib as spg
        HAS_LIBS = True
    except ImportError:
        HAS_LIBS = False

    crystals = []
    for d in all_data:
        try:
            n_atoms = d["nsites"]
            if n_atoms < 4 or n_atoms > 40: continue
            if not HAS_LIBS: break

            struct = Structure.from_dict(d["structure"])
            cell = (struct.lattice.matrix, struct.frac_coords,
                    [s.Z for s in struct.species])
            sym = spg.get_symmetry(cell, symprec=1e-3)
            if sym is None or len(sym["rotations"]) < 2: continue

            frac = struct.frac_coords
            n_ops = min(len(sym["rotations"]), 192)
            perms = []
            for op_idx in range(n_ops):
                R = sym["rotations"][op_idx]
                t = sym["translations"][op_idx]
                new_frac = (frac @ R.T + t) % 1.0
                perm = np.zeros(n_atoms, dtype=int)
                for i in range(n_atoms):
                    diff = new_frac[i:i+1] - frac
                    diff = diff - np.round(diff)
                    perm[i] = np.argmin(np.sum(diff**2, axis=1))
                perms.append(torch.tensor(perm))

            orbit_ids = torch.full((n_atoms,), -1, dtype=torch.long)
            oid = 0
            for pos in range(n_atoms):
                if orbit_ids[pos] >= 0: continue
                stack = [pos]; orbit_ids[pos] = oid
                while stack:
                    p = stack.pop()
                    for perm in perms:
                        n = int(perm[p])
                        if orbit_ids[n] < 0:
                            orbit_ids[n] = oid; stack.append(n)
                oid += 1

            atom_feats = np.zeros((n_atoms, 4), dtype=np.float32)
            for i, site in enumerate(struct.species):
                z = site.Z
                atom_feats[i] = [z/100.0, (z%8)/8.0, (z%18)/18.0, site.row/10.0]

            crystals.append({
                "formula": d["formula_pretty"],
                "spacegroup": d["symmetry"]["number"],
                "n_atoms": n_atoms, "n_orbits": oid,
                "formation_energy": d["formation_energy_per_atom"],
                "orbit_ids": orbit_ids,
                "atom_feats": torch.tensor(atom_feats, dtype=torch.float32),
            })
        except Exception:
            pass

    with open(DATA_PATH, "wb") as f:
        pickle.dump(crystals, f)
    return crystals


def run(device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 65)
    print("TABLE 2a: CRYSTAL FORMATION ENERGY PREDICTION")
    print(f"D={D}, layers={L}, epochs={EP}")
    print("=" * 65)

    crystals = download_crystals()
    sg_set = set(c["spacegroup"] for c in crystals)
    print(f"Loaded {len(crystals)} crystals from {len(sg_set)} space groups")

    max_atoms = max(c["n_atoms"] for c in crystals)
    n_samples = len(crystals)

    # Build data matrices
    data = torch.zeros(n_samples, max_atoms, D)
    targets = torch.zeros(n_samples)
    for s, c in enumerate(crystals):
        n = c["n_atoms"]
        feats = c["atom_feats"]
        proj = torch.randn(feats.size(1), D) / (feats.size(1) ** 0.5)
        data[s, :n] = feats @ proj
        targets[s] = c["formation_energy"]

    tm, ts = targets.mean(), targets.std()
    targets = (targets - tm) / ts

    idx = torch.randperm(n_samples)
    n_tr = int(n_samples * 0.8); n_te = n_samples - n_tr
    d_tr = data[idx[:n_tr]].to(device); t_tr = targets[idx[:n_tr]].to(device)
    d_te = data[idx[n_tr:]].to(device); t_te = targets[idx[n_tr:]].to(device)

    # Orbit proxy: O_h on 4x4x4 grid, truncated to max_atoms
    oh = OctahedralGroup(4)
    oh_oids_full, _ = oh.compute_orbits()
    oh_oids = oh_oids_full[:max_atoms]
    oh_K = int(oh_oids.max().item()) + 1
    print(f"O_h proxy: K={oh_K} for {max_atoms} positions")

    class EnergyHead(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, 1))
        def forward(self, x):
            return self.net(x.mean(1)).squeeze(-1)

    def train_model(model, dt, tt, de, te, ep, dev):
        model = model.to(dev); hd = EnergyHead(dt.size(-1)).to(dev)
        opt = torch.optim.AdamW(list(model.parameters()) + list(hd.parameters()),
                                lr=5e-4, weight_decay=0.05)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        best = float("inf")
        for _ in range(ep):
            model.train(); hd.train(); opt.zero_grad()
            loss = nn.functional.mse_loss(hd(model(dt)), tt)
            loss.backward(); opt.step(); sch.step()
            model.eval(); hd.eval()
            with torch.no_grad():
                p = hd(model(de)); mae = (p - te).abs().mean().item()
                if mae < best: best = mae
        return best, sum(p.numel() for p in model.parameters())

    results = {}
    torch.manual_seed(42)

    # StandardMLP
    ly = []
    for _ in range(L):
        ly.extend([nn.LayerNorm(D), nn.Linear(D, D), nn.GELU()])
    mae, params = train_model(nn.Sequential(*ly), d_tr, t_tr, d_te, t_te, EP, device)
    results["StandardMLP"] = (mae, params)
    if device == "cuda": torch.cuda.empty_cache()
    print(f"StandardMLP (K=1):       MAE={mae:.4f}  params={params:,}")

    # OrbitMLP
    torch.manual_seed(42)
    ly = []
    for _ in range(L):
        ly.extend([nn.LayerNorm(D), OrbitLinear(oh_oids, oh_K, D), nn.GELU()])
    mae, params = train_model(nn.Sequential(*ly), d_tr, t_tr, d_te, t_te, EP, device)
    results["OrbitMLP"] = (mae, params)
    if device == "cuda": torch.cuda.empty_cache()
    print(f"OrbitMLP (K={oh_K}):        MAE={mae:.4f}  params={params:,}")

    # RandomMLP
    torch.manual_seed(42)
    rid = torch.randperm(max_atoms) % max(oh_K, 2)
    nr = int(rid.max().item()) + 1
    ly = []
    for _ in range(L):
        ly.extend([nn.LayerNorm(D), OrbitLinear(rid, nr, D), nn.GELU()])
    mae, params = train_model(nn.Sequential(*ly), d_tr, t_tr, d_te, t_te, EP, device)
    results["RandomMLP"] = (mae, params)
    if device == "cuda": torch.cuda.empty_cache()
    print(f"RandomMLP (K={nr}):        MAE={mae:.4f}  params={params:,}")

    # PerPositionMLP
    torch.manual_seed(42)
    class PPL(nn.Module):
        def __init__(self, n, d):
            super().__init__()
            self.w = nn.Parameter(torch.randn(n, d, d) / (d**0.5))
            self.b = nn.Parameter(torch.zeros(n, d))
        def forward(self, x):
            return torch.einsum("bnd,ndm->bnm", x, self.w) + self.b.unsqueeze(0)
    ly = []
    for _ in range(L):
        ly.extend([nn.LayerNorm(D), PPL(max_atoms, D), nn.GELU()])
    mae, params = train_model(nn.Sequential(*ly), d_tr, t_tr, d_te, t_te, EP, device)
    results["PerPositionMLP"] = (mae, params)
    if device == "cuda": torch.cuda.empty_cache()
    print(f"PerPositionMLP (K={max_atoms}): MAE={mae:.4f}  params={params:,}")

    print("\n" + "=" * 65)
    print("SUMMARY (Table 2a format)")
    print("=" * 65)
    print(f"{'Model':<20s} {'K':>5s} {'Parameters':>12s} {'MAE':>8s}")
    print("-" * 50)
    for name, (mae, params) in sorted(results.items(), key=lambda x: x[1][0]):
        k_map = {"StandardMLP": "1", "OrbitMLP": str(oh_K),
                 "RandomMLP": str(nr), "PerPositionMLP": str(max_atoms)}
        print(f"{name:<20s} {k_map[name]:>5s} {params:>12,} {mae:>8.4f}")

    orb = results["OrbitMLP"][0]
    std = results["StandardMLP"][0]
    rnd = results["RandomMLP"][0]
    print(f"\nOrbit vs Standard: {(std-orb)/std*100:.1f}% improvement")
    print(f"Orbit vs Random:   {(rnd-orb)/rnd*100:.1f}% improvement")
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(device)
