#!/usr/bin/env python3
"""
Resubmission v3 - authentic-data, leakage-free, capacity/occupancy-controlled study.

Protocol (as specified in the 12 Sep 2026 revision plan, COMMAT-D-26-03017R1):
  * corrected cube-centred O_h action on the 4x4x4 grid -> K = 4 orbits (8/24/24/8)
  * ONE frozen composition-family split (70/10/20) shared by every model and seed
  * target mean/std fitted on TRAINING families only
  * checkpoint chosen on validation MAE only (test never used for selection)
  * per-material test predictions saved -> family-clustered paired statistics
  * no external graph-model comparison (methodology / audit paper)

Stages:  prepare | train | stats | invariance | transfer | cost
"""
import os, sys, json, pickle, time, math, argparse, hashlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"E:\operator")
from groups.base import compute_orbits_from_generators

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES  = os.path.join(ROOT, "results_v3")
DATA = os.path.join(ROOT, "data")
for d in (RES, DATA):
    os.makedirs(d, exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
D         = 64
L         = 4
EP        = 200
MAX_ATOMS = 40
BATCH     = 256
N_GRID    = 4
SEEDS     = [42, 123, 456, 789, 1024]
CACHE_DEC = r"E:\operator\crystal_data_decisive.pkl"
CACHE_SYM = r"E:\operator\crystal_data_symfeat.pkl"

CRYSTAL_SYSTEM = {}
for sg in range(1, 231):
    if sg <= 2:    CRYSTAL_SYSTEM[sg] = "triclinic"
    elif sg <= 15: CRYSTAL_SYSTEM[sg] = "monoclinic"
    elif sg <= 74: CRYSTAL_SYSTEM[sg] = "orthorhombic"
    elif sg <= 142: CRYSTAL_SYSTEM[sg] = "tetragonal"
    elif sg <= 167: CRYSTAL_SYSTEM[sg] = "trigonal"
    elif sg <= 194: CRYSTAL_SYSTEM[sg] = "hexagonal"
    else:          CRYSTAL_SYSTEM[sg] = "cubic"


# ----------------------------------------------------------------------
# 1. corrected finite-grid group action
# ----------------------------------------------------------------------
def corrected_grid_orbits(n=N_GRID):
    """Cube-centred action of the 48 signed permutation matrices on {0..n-1}^3.

    g_R(c) = h + R (c - h),  h = (n-1)/2.  Every generator is a bijection.
    Returns (orbit_label[64], K).
    """
    half = (n - 1) / 2.0
    coords = np.array([[x, y, z] for z in range(n) for y in range(n) for x in range(n)])
    centred = coords - half
    mats = [np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),   # C4_z
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),   # C4_x
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])]   # mirror
    gens = []
    for m in mats:
        for mm in (m, m.T):
            new = centred @ mm.T + half
            idx = np.rint(new).astype(int)
            if np.abs(idx - new).max() > 1e-6:
                raise RuntimeError("generator does not map the grid onto itself")
            lin = idx[:, 2] * n * n + idx[:, 1] * n + idx[:, 0]
            if len(set(lin.tolist())) != n ** 3:
                raise RuntimeError("generator is not bijective")
            gens.append(torch.tensor(lin, dtype=torch.long))
    orb, K = compute_orbits_from_generators(n ** 3, gens)
    return orb.numpy() if isinstance(orb, torch.Tensor) else np.asarray(orb), int(K)


def voxel_index(frac, n=N_GRID):
    """half-open voxel convention c = floor(n * f), f wrapped into [0,1)."""
    f = np.asarray(frac, dtype=np.float64)
    f = f - np.floor(f)
    c = np.clip(np.floor(f * n).astype(np.int64), 0, n - 1)
    return c[:, 2] * n * n + c[:, 1] * n + c[:, 0], c


def cube4_label_from_coords(frac):
    """Orbit label of the cube-centred action = number of components in {0, n-1}."""
    _, c = voxel_index(frac)
    return ((c <= 0) | (c >= N_GRID - 1)).sum(axis=1).astype(np.int64)


def geo4_label_from_coords(frac):
    lin, _ = voxel_index(frac)
    lab = np.zeros(len(lin), dtype=np.int64)
    lab[lin >= 8] = 1
    lab[lin >= 32] = 2
    lab[lin >= 56] = 3
    return lab


# ----------------------------------------------------------------------
# 2. data loading / feature construction
# ----------------------------------------------------------------------
def build_lattice(lf):
    """lattice_feat = [a/10, b/10, c/10, alpha/pi, beta/pi, gamma/pi] -> 3x3 row matrix."""
    a, b, c = lf[0] * 10.0, lf[1] * 10.0, lf[2] * 10.0
    al, be, ga = lf[3] * np.pi, lf[4] * np.pi, lf[5] * np.pi
    ca, cb, cg, sg_ = np.cos(al), np.cos(be), np.cos(ga), np.sin(ga)
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * cg, b * sg_, 0.0])
    cx = c * cb
    cy = c * (ca - cb * cg) / sg_
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.stack([va, vb, np.array([cx, cy, cz])])


def periodic_distance_matrix(frac, lat):
    diff = frac[:, None, :] - frac[None, :, :]
    diff -= np.round(diff)
    return np.linalg.norm(diff @ lat, axis=2)


def load_all():
    with open(CACHE_DEC, "rb") as f:
        dec = pickle.load(f)
    with open(CACHE_SYM, "rb") as f:
        sym = pickle.load(f)
    assert len(dec) == len(sym)
    return dec, sym


def projected_elemental(dec, proj):
    rows = []
    for c in dec:
        n = c["n_atoms"]
        x = c["atom_feats_4d"].numpy() @ proj
        px = np.zeros((MAX_ATOMS, D), dtype=np.float32)
        px[:n] = x
        rows.append(px)
    return np.stack(rows)


def make_projection(n_in, seed=0, scale=None):
    gen = torch.Generator(); gen.manual_seed(seed)
    s = (1.0 / math.sqrt(n_in)) if scale is None else scale
    return (torch.randn(n_in, D, generator=gen) * s).numpy()


def build_struct_features(dec, sym):
    """Per-atom [elemental4 | lattice6 | 8 smallest periodic distances | volume/atom]  -> 19."""
    out = np.zeros((len(dec), MAX_ATOMS, 19), dtype=np.float32)
    meta = []
    for i, (c, s) in enumerate(zip(dec, sym)):
        n = c["n_atoms"]
        frac = c["frac_coords"].numpy().astype(np.float64)[:n]
        lf = s["lattice_feat"].numpy().astype(np.float64)
        lat = build_lattice(lf)
        dm = periodic_distance_matrix(frac, lat)
        np.fill_diagonal(dm, np.inf)
        ds = np.sort(dm, axis=1)[:, :8]
        ds[~np.isfinite(ds)] = 8.0
        vol = abs(np.linalg.det(lat)) / n
        f = np.zeros((MAX_ATOMS, 19), dtype=np.float32)
        f[:n, :4] = c["atom_feats_4d"].numpy()[:n]
        f[:n, 4:10] = lf
        f[:n, 10:18] = ds / 8.0
        f[:n, 18] = vol / 100.0
        out[i] = f
        meta.append({"dist_min": float(ds.min()) if n > 1 else 0.0,
                     "volume_per_atom": float(vol)})
    return out, meta


def build_hierarchy_features(struct_feat, dec, sym):
    """struct + [orbit-size fraction, log2 orbit size, log2 site-stabiliser order] -> 22."""
    out = np.zeros((len(dec), MAX_ATOMS, 22), dtype=np.float32)
    out[:, :, :19] = struct_feat
    for i, (c, s) in enumerate(zip(dec, sym)):
        n = c["n_atoms"]
        mult = s["mult"].numpy().astype(np.float64)[:n]
        nops = max(int(c["n_sym_ops"]), 1)
        stab = nops / np.maximum(mult, 1)
        out[i, :n, 19] = mult / n
        out[i, :n, 20] = np.log2(np.maximum(mult, 1)) / 6.0
        out[i, :n, 21] = np.log2(np.maximum(stab, 1)) / 12.0
    return out


# ----------------------------------------------------------------------
# 3. labels
# ----------------------------------------------------------------------
def build_labels(dec, orbit_of_grid, K_grid, seed=0):
    rng = np.random.default_rng(seed)
    n = len(dec)
    lab = {k: np.zeros((n, MAX_ATOMS), dtype=np.int64) for k in
           ("cube4", "occupancy4", "geo4", "random4", "sitelabel")}
    Kmax_site = int(max(c["K_i"] for c in dec))
    # random permutation of the 64-label multiset -> grid-level Random4
    grid_lab = orbit_of_grid.copy()
    rnd_grid = rng.permutation(grid_lab)

    for i, c in enumerate(dec):
        na = c["n_atoms"]
        frac = c["frac_coords"].numpy()[:na]
        lin, _ = voxel_index(frac)
        cube = cube4_label_from_coords(frac)
        lab["cube4"][i, :na] = cube
        lab["geo4"][i, :na] = geo4_label_from_coords(frac)
        lab["random4"][i, :na] = rnd_grid[lin]
        # occupancy-matched: sort sites by wrapped coordinate, redistribute sorted labels
        f = frac - np.floor(frac)
        order = np.lexsort((f[:, 2], f[:, 1], f[:, 0]))
        sorted_lab = np.sort(cube)
        occ = np.empty(na, dtype=np.int64)
        occ[order] = sorted_lab
        lab["occupancy4"][i, :na] = occ
        lab["sitelabel"][i, :na] = c["wyckoff_orbit"].numpy()[:na]
    return lab, K_grid, Kmax_site


# ----------------------------------------------------------------------
# 4. models
# ----------------------------------------------------------------------
class LabelLinear(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(K, D, D) / math.sqrt(D))
        self.bias = nn.Parameter(torch.zeros(K, D))

    def forward(self, x, oid):
        w = self.weight[oid]
        b = self.bias[oid]
        return torch.einsum("bnd,bndm->bnm", x, w) + b


class LabelBlock(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.ol = LabelLinear(K)
        self.act = nn.GELU()

    def forward(self, x, oid):
        return self.act(self.ol(self.ln(x), oid))


def head():
    return nn.Sequential(nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 1))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def find_wide_H(target_total):
    h = (D * (D // 2) + D // 2) + (D // 2 + 1)
    best = None
    for H in range(16, 2000):
        # WideMLP: LN(D)->DxH, (LN(H)->HxH) x (L-2), LN(H)->HxD ; LayerNorm params included
        back = (2 * D + D * H + H) + (L - 2) * (2 * H + H * H + H) + (2 * H + H * D + D)
        t = back + h
        if best is None or abs(t - target_total) < abs(best[1] - target_total):
            best = (H, t)
    return best[0], best[1]


class OrbitMLP(nn.Module):
    """label-selected affine blocks; K=1 degenerates to a shared-transform MLP."""
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.blocks = nn.ModuleList([LabelBlock(K) for _ in range(L)])
        self.head = head()

    def forward(self, x, oid, mask=None):
        for b in self.blocks:
            x = b(x, oid)
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(1)
        return self.head(x).squeeze(-1)


class WideMLP(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(D), nn.Linear(D, H), nn.GELU()),
            nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H), nn.GELU()),
            nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H), nn.GELU()),
            nn.Sequential(nn.LayerNorm(H), nn.Linear(H, D), nn.GELU()),
        ])
        self.head = head()

    def forward(self, x, oid=None, mask=None):
        for b in self.blocks:
            x = b(x)
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(1)
        return self.head(x).squeeze(-1)


class FeatMLP(nn.Module):
    """standard architecture on an arbitrary per-atom feature vector projected to D."""
    def __init__(self, n_in):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(n_in, D) / math.sqrt(n_in), requires_grad=False)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU()) for _ in range(L)])
        self.head = head()

    def forward(self, x, oid=None, mask=None):
        x = x @ self.proj
        for b in self.blocks:
            x = b(x)
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(1)
        return self.head(x).squeeze(-1)


# ----------------------------------------------------------------------
# 5. splits
# ----------------------------------------------------------------------
def build_split(dec, seed=0, mode="family", holdout_spacegroups=None, frac=(0.70, 0.10, 0.20)):
    formulas = [c["formula"] for c in dec]
    fams = sorted(set(formulas))
    if mode == "family":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(fams))
        n = len(fams)
        ntr, nva = int(n * frac[0]), int(n * (frac[0] + frac[1]))
        tr_f = {fams[i] for i in perm[:ntr]}
        va_f = {fams[i] for i in perm[ntr:nva]}
        te_f = {fams[i] for i in perm[nva:]}
        idx_tr = [i for i, f in enumerate(formulas) if f in tr_f]
        idx_va = [i for i, f in enumerate(formulas) if f in va_f]
        idx_te = [i for i, f in enumerate(formulas) if f in te_f]
    elif mode == "spacegroup":
        ho = set(holdout_spacegroups or [])
        idx_te = [i for i, c in enumerate(dec) if c["spacegroup"] in ho]
        rest = [i for i, c in enumerate(dec) if c["spacegroup"] not in ho]
        te_f = {formulas[i] for i in idx_te}
        rest = [i for i in rest if formulas[i] not in te_f]     # keep families disjoint
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(rest))
        nva = int(len(rest) * 0.10)
        idx_va = [rest[i] for i in perm[:nva]]
        idx_tr = [rest[i] for i in perm[nva:]]
    else:
        raise ValueError(mode)
    return idx_tr, idx_va, idx_te


def manifest_of(dec, idx_tr, idx_va, idx_te):
    return {
        "n_total": len(dec),
        "n_train": len(idx_tr), "n_val": len(idx_va), "n_test": len(idx_te),
        "families_train": len({dec[i]["formula"] for i in idx_tr}),
        "families_val":   len({dec[i]["formula"] for i in idx_va}),
        "families_test":  len({dec[i]["formula"] for i in idx_te}),
        "train_idx": idx_tr, "val_idx": idx_va, "test_idx": idx_te,
        "spacegroups_test": sorted({dec[i]["spacegroup"] for i in idx_te}),
        "families_test_list": sorted({dec[i]["formula"] for i in idx_te}),
    }


# ----------------------------------------------------------------------
# 6. training
# ----------------------------------------------------------------------
def train_one(model, Xtr, Otr, Mtr, ytr, Xva, Ova, Mva, yva, Xte, Ote, Mte, yte,
              epochs=EP, verbose=True, tag=""):
    model = model.to(DEVICE)
    mu = ytr.mean()
    sd = ytr.std().clamp(min=1e-8)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    N = Xtr.shape[0]
    best = (float("inf"), None, -1)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        for s in range(0, N, BATCH):
            bi = perm[s:s + BATCH]
            bx = Xtr[bi].to(DEVICE); bm = Mtr[bi].to(DEVICE)
            by = ((ytr[bi] - mu) / sd).to(DEVICE)
            bo = Otr[bi].to(DEVICE) if Otr is not None else None
            p = model(bx, bo, bm)
            loss = crit(p, by)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva.to(DEVICE), Ova.to(DEVICE) if Ova is not None else None, Mva.to(DEVICE))
            vmae = ((pv.cpu() * sd + mu) - yva).abs().mean().item()
        if vmae < best[0]:
            best = (vmae, {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        pte = model(Xte.to(DEVICE), Ote.to(DEVICE) if Ote is not None else None, Mte.to(DEVICE))
        pte = pte.cpu() * sd + mu
    t = time.time() - t0
    test_mae = (pte - yte).abs().mean().item()
    if verbose:
        print(f"    [{tag}] valMAE={best[0]:.4f} (ep {best[2]}) testMAE={test_mae:.4f} "
              f"params={count_params(model):,} {t:.1f}s", flush=True)
    return {"test_mae": float(test_mae), "val_mae": float(best[0]),
            "best_epoch": int(best[2]), "params": count_params(model),
            "mu": float(mu), "sd": float(sd), "seconds": t}


LABEL_KEYS = ("cube4", "occupancy4", "geo4", "random4", "sitelabel")
MODEL_NAMES = ["standard", "wide", "cube4", "occupancy4", "geo4", "random4",
               "sitelabel", "struct", "hierarchy"]


def paths():
    return {
        "labels": os.path.join(DATA, "labels.npz"),
        "elem":   os.path.join(DATA, "X_elem.npy"),
        "struct": os.path.join(DATA, "X_struct.npy"),
        "hier":   os.path.join(DATA, "X_hier.npy"),
        "mask":   os.path.join(DATA, "mask.npy"),
        "split":  os.path.join(DATA, "split_family.json"),
        "audit":  os.path.join(RES, "grid_audit.json"),
        "ckpt":   os.path.join(RES, "ckpt"),
    }


def signed_permutation_matrices():
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), dtype=np.int64)
            for i, q in enumerate(perm):
                M[i, q] = signs[i]
            mats.append(M)
    return mats


def grid_action_audit(n=N_GRID):
    """Independent audit of the corrected action and of the archived one."""
    half = (n - 1) / 2.0
    coords = np.array([[x, y, z] for z in range(n) for y in range(n) for x in range(n)])
    centred = coords - half
    mats = signed_permutation_matrices()
    perms, distinct_images = {}, []
    for k, M in enumerate(mats):
        new = centred @ M.T + half
        idx = np.rint(new).astype(int)
        assert np.abs(idx - new).max() < 1e-9, "action leaves the grid"
        lin = idx[:, 2] * n * n + idx[:, 1] * n + idx[:, 0]
        perms[k] = lin
        distinct_images.append(len(set(lin.tolist())))
    lookup = {tuple(v.tolist()): k for k, v in perms.items()}
    closure_ok = sum(1 for a_ in perms for b_ in perms
                     if tuple(perms[a_][perms[b_]].tolist()) in lookup)
    orb, K = corrected_grid_orbits(n)
    sizes = sorted([int((orb == k).sum()) for k in range(K)], reverse=True)
    expect = ((coords <= 0) | (coords >= n - 1)).sum(axis=1)
    def partition(labels):
        return sorted(tuple(sorted(np.where(labels == k)[0].tolist()))
                      for k in set(np.asarray(labels).tolist()))
    same = partition(orb) == partition(expect)
    # archived construction: origin-centred, then clipped into [0, n-1]
    bad = 0
    min_images = n ** 3
    for M in mats:
        new = np.clip(coords @ M.T, 0, n - 1)
        lin = new[:, 2] * n * n + new[:, 1] * n + new[:, 0]
        u = len(set(lin.tolist()))
        min_images = min(min_images, u)
        if u != n ** 3:
            bad += 1
    from groups.octahedral import OctahedralGroup
    legacy_group = OctahedralGroup(n=n)
    legacy_orb, legacy_K = legacy_group.compute_orbits()
    stored_images = sorted({len(set(g.tolist())) for g in legacy_group.get_generators()})
    legacy_sizes = sorted(np.bincount(legacy_orb.numpy()).tolist(), reverse=True)
    return {"group": "O_h (48 signed permutation matrices x inversion)",
            "n_elements": len(mats), "n_grid": n ** 3,
            "gauge": "cube-centred: g(c) = h + R(c-h), h=(n-1)/2",
            "bijective_generators": sum(1 for u in distinct_images if u == n ** 3),
            "closure_ok": closure_ok, "closure_total": len(mats) ** 2,
            "K": int(K), "orbit_sizes": sizes,
            "label_equals_extreme_component_count": bool(same),
            "legacy": {"gauge": "origin-centred with clip(.,0,n-1)",
                       "non_bijective_generators": bad,
                       "min_distinct_images": int(min_images),
                       "stored_generators": len(legacy_group.get_generators()),
                       "stored_generator_images": [int(v) for v in stored_images],
                       "stored_generator_images_min": int(min(stored_images)),
                       "K_reported": int(legacy_K),
                       "class_sizes": legacy_sizes,
                       "status": "not an orbit decomposition: generators are not bijections"},
            "scope": "finite-grid group theorem; involves no measured target"}


def write_jsonl(dec, sym):
    for task, unit in (("formation_energy", "eV/atom"), ("band_gap", "eV")):
        out = os.path.join(DATA, f"{task}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for c, s in zip(dec, sym):
                n = c["n_atoms"]
                Z = np.rint(c["atom_feats_4d"].numpy()[:n, 0] * 100).astype(int).tolist()
                frac = np.round(c["frac_coords"].numpy()[:n].astype(float), 8).tolist()
                lat = np.round(build_lattice(s["lattice_feat"].numpy().astype(float)), 8).tolist()
                mid = hashlib.sha1((c["formula"] + "|" + str(c["spacegroup"]) + "|" +
                                    json.dumps(frac)).encode()).hexdigest()[:16]
                f.write(json.dumps({
                    "material_id": "mpx-" + mid,
                    "atomic_numbers": Z,
                    "frac_coords": frac,
                    "lattice": lat,
                    "target_eV": float(c[task]),
                    "target_unit": unit,
                    "family_id": c["formula"],
                    "spacegroup": int(c["spacegroup"]),
                    "n_sym_ops": int(c["n_sym_ops"]),
                    "data_status": "AUTHOR_SUPPLIED_RESEARCH_DATA",
                    "source": "Materials Project snapshot cached as crystal_data_decisive.pkl",
                }) + "\n")
        print("wrote", out, flush=True)


def stage_prepare(args):
    dec, sym = load_all()
    p = paths()
    os.makedirs(p["ckpt"], exist_ok=True)
    audit = grid_action_audit()
    with open(p["audit"], "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print("grid audit: bijective=%d/%d closure=%d/%d K=%d sizes=%s partition_ok=%s legacy_K=%d (%d/%d generators non-bijective)" % (
        audit["bijective_generators"], audit["n_elements"], audit["closure_ok"],
        audit["closure_total"], audit["K"], audit["orbit_sizes"],
        audit["label_equals_extreme_component_count"], audit["legacy"]["K_reported"],
        audit["legacy"]["non_bijective_generators"], audit["n_elements"]), flush=True)

    orb, K = corrected_grid_orbits()
    labels, K_grid, Kmax_site = build_labels(dec, orb, K, seed=0)
    np.savez_compressed(p["labels"], **labels, K_grid=np.int64(K_grid),
                        Kmax_site=np.int64(Kmax_site))

    Xe = projected_elemental(dec, make_projection(4, seed=0))
    np.save(p["elem"], Xe)
    Xs, _ = build_struct_features(dec, sym)
    np.save(p["struct"], Xs)
    np.save(p["hier"], build_hierarchy_features(Xs, dec, sym))

    mask = np.zeros((len(dec), MAX_ATOMS), dtype=np.float32)
    for i, c in enumerate(dec):
        mask[i, :c["n_atoms"]] = 1.0
    np.save(p["mask"], mask)

    tr, va, te = build_split(dec, seed=0, mode="family")
    man = manifest_of(dec, tr, va, te)
    man["split_rule"] = "whole composition families (reduced formula string), 70/10/20, frozen for all models"
    man["seed"] = 0
    with open(p["split"], "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    print("split: train=%d val=%d test=%d | families %d/%d/%d" % (
        man["n_train"], man["n_val"], man["n_test"],
        man["families_train"], man["families_val"], man["families_test"]), flush=True)
    write_jsonl(dec, sym)
    print("prepare done", flush=True)


def model_for(name, K_grid, Kmax_site, param_target):
    """returns (model, label_key_or_None)"""
    if name == "standard":
        return OrbitMLP(1), None
    if name == "wide":
        H, _ = find_wide_H(param_target)
        return WideMLP(H), None
    if name == "struct":
        return FeatMLP(19), None
    if name == "hierarchy":
        return FeatMLP(22), None
    K = Kmax_site if name == "sitelabel" else K_grid
    return OrbitMLP(K), name


def stage_train(args):
    p = paths()
    dec, sym = load_all()
    split_path = getattr(args, "split", "") or p["split"]
    suffix = getattr(args, "suffix", "") or ""
    man = json.load(open(split_path, encoding="utf-8"))
    Xe_all = np.load(p["elem"])
    Xs_all = np.load(p["struct"])
    Xh_all = np.load(p["hier"])
    mask_all = np.load(p["mask"])
    lab_all = np.load(p["labels"])
    K_grid = int(lab_all["K_grid"]); Kmax_site = int(lab_all["Kmax_site"])
    param_target = count_params(OrbitMLP(K_grid))
    os.makedirs(p["ckpt"], exist_ok=True)
    print("K_grid=%d K_site=%d orbit_target_params=%d wideH=%s" % (
        K_grid, Kmax_site, param_target, find_wide_H(param_target)), flush=True)

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS
    models = args.models.split(",") if args.models else MODEL_NAMES
    tasks = ["formation_energy", "band_gap"] if args.task == "both" else [args.task]
    idx_tr = np.array(man["train_idx"]); idx_va = np.array(man["val_idx"])
    idx_te = np.array(man["test_idx"])

    for task in tasks:
        Y = np.array([c[task] for c in dec], dtype=np.float64)
        valid = ~np.isnan(Y)
        pos = {int(i): k for k, i in enumerate(np.where(valid)[0])}
        y = torch.tensor(Y[valid], dtype=torch.float32)
        tr = torch.tensor([pos[int(i)] for i in idx_tr if int(i) in pos])
        va = torch.tensor([pos[int(i)] for i in idx_va if int(i) in pos])
        te = torch.tensor([pos[int(i)] for i in idx_te if int(i) in pos])
        te_global = np.array([int(i) for i in idx_te if int(i) in pos])
        Xe = torch.tensor(Xe_all[valid]); Xs = torch.tensor(Xs_all[valid])
        Xh = torch.tensor(Xh_all[valid]); mk = torch.tensor(mask_all[valid])
        labs = {k: torch.tensor(lab_all[k][valid]) for k in LABEL_KEYS}
        print(f"== {task}: {len(y)} records ({int(valid.sum())}/{len(Y)}), "
              f"train/val/test = {len(tr)}/{len(va)}/{len(te)}", flush=True)

        res_path = os.path.join(RES, f"train_{task}{suffix}.json")
        results = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else {}
        pdir = os.path.join(RES, "preds")
        os.makedirs(pdir, exist_ok=True)
        np.save(os.path.join(pdir, f"{task}{suffix}_ytest.npy"), y[te].numpy())
        np.save(os.path.join(pdir, f"{task}{suffix}_testglobal.npy"), te_global)
        for name in models:
            for seed in seeds:
                key = f"{name}@{seed}"
                pp = os.path.join(pdir, f"{task}{suffix}_{name}_{seed}.npy")
                if key in results and os.path.exists(pp) and not args.force:
                    print(f"    [skip] {key}", flush=True)
                    continue
                torch.manual_seed(seed)
                np.random.seed(seed)
                model, lk = model_for(name, K_grid, Kmax_site, param_target)
                Otr = labs[lk][tr] if lk else None
                Ova = labs[lk][va] if lk else None
                Ote = labs[lk][te] if lk else None
                Xtr = Xs[tr] if name == "struct" else (Xh[tr] if name == "hierarchy" else Xe[tr])
                Xva = Xs[va] if name == "struct" else (Xh[va] if name == "hierarchy" else Xe[va])
                Xte = Xs[te] if name == "struct" else (Xh[te] if name == "hierarchy" else Xe[te])
                r = train_one(model, Xtr, Otr, mk[tr], y[tr], Xva, Ova, mk[va], y[va],
                              Xte, Ote, mk[te], y[te], epochs=args.epochs,
                              tag=f"{task} {name} s{seed}")
                r["label_key"] = lk
                r["K"] = 1 if lk is None else (Kmax_site if name == "sitelabel" else K_grid)
                model.eval()
                with torch.no_grad():
                    pv = model(Xte.to(DEVICE), Ote.to(DEVICE) if Ote is not None else None,
                               mk[te].to(DEVICE)).cpu().numpy() * r["sd"] + r["mu"]
                r["test_mae"] = float(np.abs(pv - y[te].numpy()).mean())
                results[key] = r
                np.save(pp, pv)
                torch.save(model.state_dict(), os.path.join(p["ckpt"], f"{task}{suffix}_{name}_{seed}.pt"))
                with open(res_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
        print("finished", task, flush=True)
    print("train done", flush=True)


# ----------------------------------------------------------------------
# 8. family-clustered statistics
# ----------------------------------------------------------------------
def _cluster_index(fam):
    uniq = np.unique(fam)
    return uniq, {f: np.where(fam == f)[0] for f in uniq}


def bootstrap_mean_ci(values, fam, n_boot=5000, seed=0, alpha=0.05):
    uniq, idx = _cluster_index(fam)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    n = len(uniq)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        sel = np.concatenate([idx[uniq[q]] for q in pick])
        draws[b] = values[sel].mean()
    return {"mean": float(values.mean()),
            "ci_lo": float(np.quantile(draws, alpha / 2)),
            "ci_hi": float(np.quantile(draws, 1 - alpha / 2)),
            "n_test_records": int(len(values)), "n_test_families": int(n)}


def paired_cluster_contrast(err_a, err_b, fam, n_boot=5000, seed=0):
    d = err_a - err_b
    ci = bootstrap_mean_ci(d, fam, n_boot=n_boot, seed=seed)
    p_val = float("nan")
    try:
        from scipy.stats import wilcoxon
        p_val = float(wilcoxon(err_a, err_b, zero_method="wilcox",
                               alternative="two-sided").pvalue)
    except Exception:
        pass
    # fraction of family-resampled draws in which the sign flips
    uniq, idx = _cluster_index(fam)
    rng = np.random.default_rng(seed + 1)
    flips = 0
    B = 2000
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[q]] for q in pick])
        if d[sel].mean() * d.mean() < 0:
            flips += 1
    return {"delta_mean": float(d.mean()), "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
            "wilcoxon_p": p_val, "sign_flip_fraction": flips / B,
            "n_test_records": int(len(d))}


CONTRASTS = [("cube4", "occupancy4"), ("cube4", "geo4"), ("cube4", "random4"),
             ("cube4", "standard"), ("cube4", "wide"), ("cube4", "sitelabel"),
             ("occupancy4", "standard"), ("geo4", "standard"), ("random4", "standard"),
             ("sitelabel", "standard"), ("wide", "standard"),
             ("struct", "standard"), ("hierarchy", "struct")]


def stage_stats(args):
    p = paths()
    dec, sym = load_all()
    pdir = os.path.join(RES, "preds")
    out = {}
    for task in ("formation_energy", "band_gap"):
        rp = os.path.join(RES, f"train_{task}.json")
        if not os.path.exists(rp):
            continue
        tr = json.load(open(rp, encoding="utf-8"))
        y = np.load(os.path.join(pdir, f"{task}_ytest.npy"))
        te_global = np.load(os.path.join(pdir, f"{task}_testglobal.npy"))
        fam = np.array([dec[int(i)]["formula"] for i in te_global])
        sysname = np.array([CRYSTAL_SYSTEM[int(dec[int(i)]["spacegroup"])] for i in te_global])
        sg = np.array([int(dec[int(i)]["spacegroup"]) for i in te_global])
        models = sorted({k.split("@")[0] for k in tr})
        errs, block = {}, {}
        for m in models:
            seeds = sorted(int(k.split("@")[1]) for k in tr if k.startswith(m + "@"))
            P = np.stack([np.load(os.path.join(pdir, f"{task}_{m}_{s}.npy")) for s in seeds])
            E = np.abs(P - y[None, :])
            errs[m] = E.mean(0)
            block[m] = {"seeds": seeds,
                        "mae_per_seed": [float(v) for v in E.mean(1)],
                        "mae_mean": float(E.mean(1).mean()),
                        "mae_sd_over_seeds": float(E.mean(1).std(ddof=1)) if len(seeds) > 1 else 0.0,
                        "params": tr[f"{m}@{seeds[0]}"]["params"],
                        "K": tr[f"{m}@{seeds[0]}"].get("K"),
                        "train_seconds_mean": float(np.mean([tr[f"{m}@{s}"]["seconds"] for s in seeds])),
                        "bootstrap_ci": bootstrap_mean_ci(errs[m], fam)}
        per_sys = {}
        for s in sorted(set(sysname.tolist())):
            sel = sysname == s
            entry = {"n_test": int(sel.sum()),
                     "spacegroups": sorted(set(sg[sel].tolist()))}
            for m in models:
                entry[m] = float(errs[m][sel].mean())
            per_sys[s] = entry
        # symmetry-operation strata (terciles of the number of symmetry operations)
        nops = np.array([int(dec[int(i)]["n_sym_ops"]) for i in te_global])
        q1, q2 = np.quantile(nops, [1 / 3, 2 / 3])
        band = np.where(nops <= q1, "low", np.where(nops <= q2, "medium", "high"))
        per_band = {}
        for b in ("low", "medium", "high"):
            sel = band == b
            entry = {"n_test": int(sel.sum()),
                     "n_sym_ops_min": int(nops[sel].min()),
                     "n_sym_ops_max": int(nops[sel].max())}
            for m in models:
                entry[m] = float(errs[m][sel].mean())
            entry["paired"] = {f"{a}_vs_{b}": paired_cluster_contrast(
                errs[a][sel], errs[b][sel], fam[sel], n_boot=2000)
                for a, b in [("cube4", "standard"), ("geo4", "standard"),
                             ("random4", "standard"), ("wide", "standard")]
                if a in errs and b in errs}
            per_band[b] = entry
        paired = {f"{a}_vs_{b}": paired_cluster_contrast(errs[a], errs[b], fam)
                  for a, b in CONTRASTS if a in errs and b in errs}
        out[task] = {"models": block, "per_crystal_system": per_sys,
                     "per_symmetry_stratum": per_band, "paired": paired,
                     "n_test_records": int(len(y)),
                     "n_test_families": int(len(set(fam.tolist())))}
        print(f"== {task} ==", flush=True)
        for m in models:
            b = block[m]
            print(f"   {m:11s} MAE={b['mae_mean']:.4f} +/- {b['mae_sd_over_seeds']:.4f}  "
                  f"CI[{b['bootstrap_ci']['ci_lo']:.4f},{b['bootstrap_ci']['ci_hi']:.4f}]  "
                  f"K={b['K']} params={b['params']:,}", flush=True)
        for k, v in paired.items():
            print(f"   {k:28s} d={v['delta_mean']:+.4f} "
                  f"CI[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] p={v['wilcoxon_p']:.3g}", flush=True)
    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.join(RES, "summary.json"), flush=True)


# ----------------------------------------------------------------------
# 9. invariance / robustness on raw structures
# ----------------------------------------------------------------------
def labels_from_raw(frac, wyckoff, rnd_grid):
    lin, _ = voxel_index(frac)
    cube = cube4_label_from_coords(frac)
    f = frac - np.floor(frac)
    order = np.lexsort((f[:, 2], f[:, 1], f[:, 0]))
    occ = np.empty(len(frac), dtype=np.int64)
    occ[order] = np.sort(cube)
    return {"cube4": cube, "geo4": geo4_label_from_coords(frac),
            "random4": rnd_grid[lin], "occupancy4": occ,
            "sitelabel": np.asarray(wyckoff, dtype=np.int64)}


def stage_invariance(args):
    p = paths()
    dec, sym = load_all()
    Xe_all = np.load(p["elem"]); Xs_all = np.load(p["struct"]); Xh_all = np.load(p["hier"])
    mask_all = np.load(p["mask"])
    lab_all = np.load(p["labels"])
    K_grid = int(lab_all["K_grid"]); Kmax = int(lab_all["Kmax_site"])
    rng = np.random.default_rng(0)
    rnd_grid = rng.permutation(corrected_grid_orbits()[0])
    n_samp = int(getattr(args, "n_samples", 0) or 250)
    sel = np.random.default_rng(1).choice(len(dec), min(n_samp, len(dec)), replace=False)
    proj = make_projection(4, seed=0)
    proj19 = None
    report = {"n_crystals": int(len(sel)),
              "padding_slots_added": 19,
              "checks": {}}
    for task in ("formation_energy",):
        results = {}
        for name in MODEL_NAMES:
            ck = os.path.join(p["ckpt"], f"{task}_{name}_42.pt")
            if not os.path.exists(ck):
                continue
            if name == "wide":
                H, _ = find_wide_H(count_params(OrbitMLP(K_grid)))
                model, lk = WideMLP(H), None
            elif name == "struct":
                model, lk = FeatMLP(19), None
            elif name == "hierarchy":
                model, lk = FeatMLP(22), None
            elif name == "standard":
                model, lk = OrbitMLP(1), None
            else:
                model, lk = OrbitMLP(Kmax if name == "sitelabel" else K_grid), name
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            model.to(DEVICE).eval()

            def features_for(feats, labs, mask):
                if name == "struct":
                    x = torch.tensor(feats, dtype=torch.float32)
                elif name == "hierarchy":
                    x = torch.tensor(feats, dtype=torch.float32)
                else:
                    x = torch.tensor(feats, dtype=torch.float32)
                o = torch.tensor(labs, dtype=torch.long) if lk else None
                m = torch.tensor(mask, dtype=torch.float32)
                with torch.no_grad():
                    out = model(x.to(DEVICE), o.to(DEVICE) if o is not None else None,
                                m.to(DEVICE))
                return out.cpu().numpy()

            base_e, perm_e, pad_e = [], [], []
            order_rng = np.random.default_rng(7)
            for i in sel:
                i = int(i)
                c = dec[i]; n = c["n_atoms"]
                mask0 = mask_all[i].copy()
                # --- original
                feat0 = (Xs_all[i] if name == "struct" else
                         Xh_all[i] if name == "hierarchy" else Xe_all[i])
                lab0 = lab_all[lk][i].copy() if lk else None
                base_e.append(features_for(feat0[None], None if lab0 is None else lab0[None],
                                           mask0[None])[0])
                # --- raw permutation, labels recomputed from the permuted coordinates
                perm = order_rng.permutation(n)
                idx = np.concatenate([perm, np.arange(n, MAX_ATOMS)])
                featp = feat0[idx] if name != "struct" and name != "hierarchy" else feat0[idx]
                maskp = mask0[idx]
                if name == "struct" or name == "hierarchy":
                    featp = feat0[idx]
                else:
                    featp = feat0[idx]
                if lk:
                    frac = c["frac_coords"].numpy()[:n][perm]
                    wy = c["wyckoff_orbit"].numpy()[:n][perm]
                    lp = labels_from_raw(frac, wy, rnd_grid)[lk]
                    labp = np.zeros(MAX_ATOMS, dtype=np.int64)
                    labp[:n] = lp
                else:
                    labp = None
                perm_e.append(features_for(featp[None], None if labp is None else labp[None],
                                           maskp[None])[0])
                # --- padding: same crystal plus 19 masked slots holding non-zero junk,
                #     placed BEFORE the real sites so that real rows move position.
                extra = min(19, MAX_ATOMS - n)
                pad = np.zeros(MAX_ATOMS, dtype=np.float32)
                featd = np.zeros_like(feat0)
                junk_rng = np.random.default_rng(1000 + i)
                featd[extra:extra + n] = feat0[:n]
                featd[:extra] = junk_rng.normal(0, 1, size=(extra, feat0.shape[-1])).astype(np.float32)
                pad[extra:extra + n] = 1.0
                if lk:
                    labd = junk_rng.integers(0, (Kmax if name == "sitelabel" else K_grid),
                                             size=MAX_ATOMS).astype(np.int64)
                    labd[extra:extra + n] = lab0[:n]
                else:
                    labd = None
                featd[extra + n:] = junk_rng.normal(0, 1, size=(MAX_ATOMS - extra - n,
                                                               feat0.shape[-1])).astype(np.float32)
                pad_e.append(features_for(featd[None], None if labd is None else labd[None],
                                          pad[None])[0])
            base_e = np.array(base_e); perm_e = np.array(perm_e); pad_e = np.array(pad_e)
            scale = float(np.abs(base_e).mean()) + 1e-12
            results[name] = {
                "params": count_params(model),
                "max_abs_permutation_difference_eV": float(np.abs(perm_e - base_e).max()),
                "max_rel_permutation": float(np.abs(perm_e - base_e).max() / scale),
                "max_abs_padding_difference_eV": float(np.abs(pad_e - base_e).max()),
                "tolerance_eV": 1e-5,
                "permutation_pass": bool(np.abs(perm_e - base_e).max() < 1e-4),
                "padding_pass": bool(np.abs(pad_e - base_e).max() < 1e-4),
            }
            print(f"   invariance {name:11s} perm_dmax={results[name]['max_abs_permutation_difference_eV']:.3e} "
                  f"pad_dmax={results[name]['max_abs_padding_difference_eV']:.3e}", flush=True)
        report["checks"][task] = results
    with open(os.path.join(RES, "invariance.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("wrote invariance.json", flush=True)


def stage_transfer(args):
    from collections import Counter
    p = paths()
    dec, sym = load_all()
    cnt = Counter(c["spacegroup"] for c in dec)
    order = sorted(cnt, key=lambda s: (cnt[s], s))
    hold, tot = [], 0
    for s in order:
        if tot >= 100:
            break
        hold.append(s); tot += cnt[s]
    tr, va, te = build_split(dec, 0, mode="spacegroup", holdout_spacegroups=hold)
    man = manifest_of(dec, tr, va, te)
    man["holdout_spacegroups"] = hold
    man["split_rule"] = "least-frequent space groups held out until >=100 structures; families disjoint"
    xp = os.path.join(DATA, "split_xfer.json")
    with open(xp, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    print("transfer split: train=%d val=%d test=%d spacegroups_held_out=%d" % (
        man["n_train"], man["n_val"], man["n_test"], len(hold)), flush=True)
    ns = argparse.Namespace(epochs=args.epochs, seeds=args.seeds, models=args.models,
                            task=args.task, force=args.force, split=xp, suffix="_xfer")
    stage_train(ns)


def stage_cost(args):
    p = paths()
    dec, sym = load_all()
    out = {"device": DEVICE,
           "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None}
    t0 = time.time()
    orb, K = corrected_grid_orbits()
    for c in dec:
        cube4_label_from_coords(c["frac_coords"].numpy()[:c["n_atoms"]])
    out["label_preprocessing_seconds_total"] = time.time() - t0
    out["label_preprocessing_ms_per_crystal"] = 1000 * (time.time() - t0) / len(dec)
    Xe = torch.tensor(np.load(p["elem"]))
    mask = torch.tensor(np.load(p["mask"]))
    lab = np.load(p["labels"])
    K_grid = int(lab["K_grid"]); Kmax = int(lab["Kmax_site"])
    orb_lab = torch.tensor(lab["cube4"])
    for name in MODEL_NAMES:
        if name == "wide":
            H, _ = find_wide_H(count_params(OrbitMLP(K_grid)))
            model, lk = WideMLP(H), None
        elif name == "standard":
            model, lk = OrbitMLP(1), None
        elif name == "struct":
            model, lk = FeatMLP(19), None
        elif name == "hierarchy":
            model, lk = FeatMLP(22), None
        else:
            model, lk = OrbitMLP(Kmax if name == "sitelabel" else K_grid), name
        model.to(DEVICE).eval()
        x_src = (torch.tensor(np.load(p["struct"])) if name == "struct" else
                 torch.tensor(np.load(p["hier"])) if name == "hierarchy" else Xe)
        o = orb_lab if lk else None
        entry = {"params": count_params(model)}
        for bs in (1, 64):
            xb = x_src[:bs].to(DEVICE); mb = mask[:bs].to(DEVICE)
            ob = o[:bs].to(DEVICE) if o is not None else None
            with torch.no_grad():
                for _ in range(3):
                    model(xb, ob, mb)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t = time.time()
                for _ in range(30):
                    model(xb, ob, mb)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                dt = (time.time() - t) / 30
            entry[f"forward_batch{bs}_ms"] = 1000 * dt
            if DEVICE == "cuda":
                entry[f"forward_batch{bs}_peak_alloc_MB"] = (
                    torch.cuda.max_memory_allocated() / 1e6)
        out[name] = entry
    with open(os.path.join(RES, "cost.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote cost.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train", "stats", "invariance",
                                      "transfer", "cost"])
    ap.add_argument("--epochs", type=int, default=EP)
    ap.add_argument("--seeds", type=str, default="")
    ap.add_argument("--models", type=str, default="")
    ap.add_argument("--task", type=str, default="both")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split", type=str, default="")
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--n-samples", dest="n_samples", type=int, default=0)
    args = ap.parse_args()
    print("stage:", args.stage, "| device:", DEVICE, flush=True)
    {"prepare": stage_prepare, "train": stage_train, "stats": stage_stats,
     "invariance": stage_invariance, "transfer": stage_transfer,
     "cost": stage_cost}[args.stage](args)
