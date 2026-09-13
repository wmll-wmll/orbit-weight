#!/usr/bin/env python3
"""Figures for the v3 methodology package. Every figure is written as SVG (source),
600 dpi PNG (for inspection / journal upload) and 600 dpi TIFF."""
import os, json, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results_v3")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 120,
                     "axes.spines.top": False, "axes.spines.right": False})

ORDER = ["standard", "wide", "cube4", "occupancy4", "geo4", "random4",
         "sitelabel", "struct", "hierarchy"]
PRETTY = {
    "standard": "Shared (K=1)",
    "wide": "Wide (capacity-matched)",
    "cube4": "Cube4 (corrected O$_h$ proxy)",
    "occupancy4": "Occupancy4 (occupancy-matched)",
    "geo4": "Geo4 (geometric bins)",
    "random4": "Random4 (random labels)",
    "sitelabel": "SiteLabel (within-crystal orbits)",
    "struct": "Struct (distances + lattice)",
    "hierarchy": "Hierarchy (Struct + sym. descriptors)",
}
COLOR = {"standard": "#4C72B0", "wide": "#DD8452", "cube4": "#55A868",
         "occupancy4": "#C44E52", "geo4": "#8172B3", "random4": "#937860",
         "sitelabel": "#DA8BC3", "struct": "#8C8C8C", "hierarchy": "#CCB974"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight", dpi=600)
    fig.savefig(os.path.join(FIG, f"{name}.tiff"), bbox_inches="tight", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("wrote", name)


def fig1_main(summary):
    d = summary["formation_energy"]["models"]
    names = [m for m in ORDER if m in d]
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    y = np.arange(len(names))[::-1]
    for yi, m in zip(y, names):
        b = d[m]
        lo, hi = b["bootstrap_ci"]["ci_lo"], b["bootstrap_ci"]["ci_hi"]
        ax.barh(yi, b["mae_mean"], color=COLOR[m], alpha=0.9, height=0.62, zorder=2)
        ax.plot([lo, hi], [yi, yi], color="k", lw=1.0, zorder=3)
        ax.plot([lo, lo], [yi - 0.09, yi + 0.09], color="k", lw=1.0, zorder=3)
        ax.plot([hi, hi], [yi - 0.09, yi + 0.09], color="k", lw=1.0, zorder=3)
        ax.text(hi + 0.004, yi, f'{b["mae_mean"]:.4f}',
                va="center", ha="left", fontsize=7.5, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{PRETTY[m]}  ({d[m]["params"]:,} p)' for m in names])
    ax.set_xlabel("Formation-energy MAE (eV/atom), test families held out")
    ax.set_title("Authentic cohort, 2,741 crystals, 5 seeds; bars = family-clustered 95% CI")
    ax.set_xlim(0, max(d[m]["bootstrap_ci"]["ci_hi"] for m in names) * 1.30)
    ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)
    save(fig, "fig1_main_comparison")


def fig2_audit(audit):
    n = 4
    coords = np.array([[x, y, z] for z in range(n) for y in range(n) for x in range(n)])
    expect = ((coords <= 0) | (coords >= n - 1)).sum(axis=1)
    fig = plt.figure(figsize=(7.4, 3.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.60, wspace=0.32)
    cmap = plt.get_cmap("tab10")
    for k, z in enumerate(range(n)):
        ax = fig.add_subplot(gs[0, k])
        m = coords[:, 2] == z
        ax.scatter(coords[m, 0], coords[m, 1], c=[cmap(t) for t in expect[m]],
                   s=64, marker="s", edgecolor="k", linewidth=0.4)
        ax.set_title(f"z = {z}", fontsize=8)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-0.6, n - 0.4)
        ax.set_aspect("equal")
        if k == 0:
            ax.set_ylabel("grid y (voxel index)")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("grid x", fontsize=8)
    ax = fig.add_subplot(gs[1, 0:2])
    lab = np.bincount(expect, minlength=4)
    ax.bar(range(4), lab, color=[cmap(t) for t in range(4)], edgecolor="k", linewidth=0.4)
    for i, v in enumerate(lab):
        ax.text(i, v + 0.6, str(v), ha="center", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([str(i) for i in range(4)], fontsize=8)
    ax.set_xlabel("components equal to 0 or 3", fontsize=8)
    ax.set_ylabel("grid points", fontsize=8)
    ax.set_ylim(0, 29)
    ax.set_title(f'orbit sizes {", ".join(map(str, audit["orbit_sizes"]))}', fontsize=8)
    ax = fig.add_subplot(gs[1, 2:4])
    ax.axis("off")
    ax.text(0.0, 0.98,
            "label  $\\ell(c)=\\#\\{i: c_i\\in\\{0,3\\}\\}$\n\n"
            "worked examples (voxel $c=\\lfloor 4f\\rfloor$)\n"
            "  $f=(0.13,0.62,0.51)\\rightarrow c=(0,2,2)\\rightarrow$ label 1\n"
            "  $f=(0.98,0.02,0.87)\\rightarrow c=(3,0,3)\\rightarrow$ label 3\n"
            "  $f=(0.40,0.40,0.40)\\rightarrow c=(1,1,1)\\rightarrow$ label 0\n\n"
            "Two atoms in the same voxel share the label\n"
            "but keep separate feature vectors.",
            va="top", ha="left", fontsize=7.4, linespacing=1.4)
    fig.suptitle("Corrected cube-centred $O_h$ action on the $4\\times4\\times4$ grid: "
                 "48/48 bijections, 2304/2304 compositions close, $K=4$",
                 fontsize=9, y=0.99)
    save(fig, "fig2_grid_orbits")


def fig3_persystem(summary):
    ps = summary["formation_energy"]["per_crystal_system"]
    systems = ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
               "trigonal", "hexagonal", "cubic"]
    systems = [s for s in systems if s in ps]
    show = ["standard", "wide", "cube4", "occupancy4", "geo4", "random4"]
    x = np.arange(len(systems))
    w = 0.14
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for i, m in enumerate(show):
        ax.bar(x + (i - (len(show) - 1) / 2) * w, [ps[s][m] for s in systems],
               width=w, label=PRETTY[m], color=COLOR[m], edgecolor="k", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={ps[s]['n_test']})" for s in systems], fontsize=8)
    ax.set_ylabel("MAE (eV/atom)")
    ax.set_ylim(0, max(ps[s][m] for s in systems for m in show) * 1.10)
    ax.set_title("Formation-energy error by crystal system (same frozen test families)")
    ax.legend(ncol=3, frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", ls=":", alpha=0.5)
    save(fig, "fig3_per_crystal_system")


def fig4_paired(summary):
    p = summary["formation_energy"]["paired"]
    keys = [k for k in p]
    keys.sort(key=lambda k: abs(p[k]["delta_mean"]))
    fig, ax = plt.subplots(figsize=(6.4, 0.36 * len(keys) + 1.2))
    y = np.arange(len(keys))
    for yi, k in zip(y, keys):
        v = p[k]
        sig = not (v["ci_lo"] <= 0 <= v["ci_hi"])
        c = "#C44E52" if sig else "#7F7F7F"
        ax.plot([v["ci_lo"], v["ci_hi"]], [yi, yi], color=c, lw=1.4)
        ax.plot([v["delta_mean"]], [yi], "o", color=c, ms=4)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("_vs_", " vs ") for k in keys], fontsize=8)
    ax.set_xlabel(r"paired MAE difference $\Delta$ (eV/atom); negative = first model better")
    ax.set_title("Paired contrasts with family-clustered 95% CI (red = CI excludes 0)")
    ax.grid(axis="x", ls=":", alpha=0.5)
    save(fig, "fig4_paired_contrasts")


def fig5_invariance_cost(inv, cost):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    fig.subplots_adjust(wspace=0.42)
    ax = axes[0]
    checks = inv["checks"]["formation_energy"]
    names = [m for m in ORDER if m in checks]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [max(checks[m]["max_abs_permutation_difference_eV"], 1e-12) for m in names],
           width=0.4, label="raw permutation", color="#4C72B0")
    ax.bar(x + 0.2, [max(checks[m]["max_abs_padding_difference_eV"], 1e-12) for m in names],
           width=0.4, label="masked padding", color="#DD8452")
    ax.axhline(1e-4, color="k", ls="--", lw=0.8)
    ax.text(0.4, 1.5e-4, "pass threshold $10^{-4}$ eV", ha="left", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 2e-3)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[m].split(" (")[0] for m in names], rotation=35,
                       ha="right", fontsize=7)
    ax.set_ylabel("max |$\\Delta$ prediction| (eV)")
    ax.set_title("Invariance to raw atom order and\nmasked padding", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.98))
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax = axes[1]
    names2 = [m for m in ORDER if m in cost]
    ax.bar(np.arange(len(names2)), [cost[m]["forward_batch64_ms"] for m in names2],
           color=[COLOR[m] for m in names2], edgecolor="k", linewidth=0.3)
    ax.set_xticks(np.arange(len(names2)))
    ax.set_xticklabels([PRETTY[m].split(" (")[0] for m in names2], rotation=35,
                       ha="right", fontsize=7)
    ax.set_ylabel("cached forward, batch 64 (ms)")
    ax.set_title("Inference cost\n(features cached on device)", fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.5)
    save(fig, "fig5_invariance_cost")


def figS1_workflow():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axis("off")
    steps = [
        ("Authentic cohort",
         "2,741 Materials Project crystals, 158 space groups, 2-40 sites per cell"),
        ("Frozen composition-family split",
         "whole reduced formulas assigned 70/10/20 once: 1,921 train / 270 validation / 550 test"),
        ("Half-open voxel map",
         "c = floor(4f); label = number of voxel components equal to 0 or 3"),
        ("Leakage-free training",
         "train-only target statistics; validation-only checkpoint; test scored once"),
        ("Family-clustered statistics",
         "paired bootstrap over families plus Wilcoxon signed-rank on per-crystal errors"),
    ]
    top, hgt, gap = 0.98, 0.090, 0.042
    for i, (title, detail) in enumerate(steps):
        y = top - i * (hgt + gap) - hgt
        ax.add_patch(FancyBboxPatch((0.07, y), 0.90, hgt, boxstyle="round,pad=0.010",
                                    fc="#EAF1F8", ec="#4C72B0", lw=1.0))
        ax.text(0.10, y + hgt * 0.70, f"{i+1}.  {title}", fontsize=8.5,
                fontweight="bold", va="center", ha="left")
        ax.text(0.10, y + hgt * 0.27, detail, fontsize=7.0, va="center", ha="left")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((0.52, y - 0.004), (0.52, y - gap + 0.004),
                                         arrowstyle="-|>", mutation_scale=8, color="k"))
    y = top - len(steps) * (hgt + gap) - 0.012
    # corrected-action summary box
    bh = 0.100
    ax.add_patch(FancyBboxPatch((0.07, y - bh), 0.90, bh, boxstyle="round,pad=0.010",
                                fc="#F6F1E7", ec="#B08A3E", lw=1.0))
    ax.text(0.10, y - bh * 0.32, "Corrected $O_h$ action (cube-centred)", fontsize=8.5,
            fontweight="bold", va="center", ha="left")
    ax.text(0.10, y - bh * 0.72, "48/48 bijections, 2,304/2,304 compositions close, "
            "$K=4$ orbits of size 8/24/24/8", fontsize=7.0, va="center", ha="left")
    y -= bh + 0.020
    # matched-controls summary box (two detail lines)
    bh2 = 0.150
    ax.add_patch(FancyBboxPatch((0.07, y - bh2), 0.90, bh2, boxstyle="round,pad=0.010",
                                fc="#EDF5EC", ec="#55A868", lw=1.0))
    ax.text(0.10, y - bh2 * 0.22, "Matched controls", fontsize=8.5, fontweight="bold",
            va="center", ha="left")
    ax.text(0.10, y - bh2 * 0.52, "Occupancy4 (occupancy-preserving relabel), "
            "Geo4 (deterministic voxel bins),", fontsize=7.0, va="center", ha="left")
    ax.text(0.10, y - bh2 * 0.80, "Random4 (random labels), Wide (capacity-matched)",
            fontsize=7.0, va="center", ha="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "figS1_workflow")


if __name__ == "__main__":
    summary = json.load(open(os.path.join(RES, "summary.json"), encoding="utf-8"))
    audit = json.load(open(os.path.join(RES, "grid_audit.json"), encoding="utf-8"))
    fig1_main(summary)
    fig2_audit(audit)
    fig3_persystem(summary)
    fig4_paired(summary)
    figS1_workflow()
    ip = os.path.join(RES, "invariance.json")
    cp = os.path.join(RES, "cost.json")
    if os.path.exists(ip) and os.path.exists(cp):
        fig5_invariance_cost(json.load(open(ip, encoding="utf-8")),
                             json.load(open(cp, encoding="utf-8")))
    print("figures done")
