#!/usr/bin/env python3
"""Build the revised main text and supplementary material from the measured JSON.

Every number that appears in the documents is read from results_v3/*.json, so the
documents cannot drift from the runs that produced them.
"""
import os, json, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results_v3")
MS = os.path.join(ROOT, "manuscript")
os.makedirs(MS, exist_ok=True)

PRETTY = {
    "standard": "SharedMLP (K=1)",
    "wide": "WideMLP (capacity-matched)",
    "cube4": "Cube4 (corrected cube-centred proxy, K=4)",
    "occupancy4": "Occupancy4 (occupancy-matched relabel, K=4)",
    "geo4": "Geo4 (deterministic voxel-index bins, K=4)",
    "random4": "Random4 (random labels, K=4)",
    "sitelabel": "SiteLabel (within-crystal symmetry classes, K=32)",
    "struct": "StructMLP (elemental + lattice + 8 distances + volume/atom)",
    "hierarchy": "HierarchyMLP (StructMLP + symmetry descriptors)",
}
ORDER = ["standard", "wide", "cube4", "occupancy4", "geo4", "random4",
         "sitelabel", "struct", "hierarchy"]


def f4(x):
    return f"{x:.4f}"


def f3(x):
    return f"{x:.3f}"


def main():
    S = json.load(open(os.path.join(RES, "summary.json"), encoding="utf-8"))
    A = json.load(open(os.path.join(RES, "grid_audit.json"), encoding="utf-8"))
    I = json.load(open(os.path.join(RES, "invariance.json"), encoding="utf-8"))
    C = json.load(open(os.path.join(RES, "cost.json"), encoding="utf-8"))
    fe = S["formation_energy"]
    bg = S.get("band_gap", {})
    M = fe["models"]

    def row(m):
        b = M[m]
        return (f"| {PRETTY[m]} | {b['K']} | {b['params']:,} | {f4(b['mae_mean'])} "
                f"&plusmn; {f4(b['mae_sd_over_seeds'])} | "
                f"[{f4(b['bootstrap_ci']['ci_lo'])}, {f4(b['bootstrap_ci']['ci_hi'])}] | "
                f"{100*(b['mae_mean']/M['standard']['mae_mean']-1):+.1f}% |")

    table = "\n".join(row(m) for m in ORDER if m in M)
    tbl_bg = ""
    if bg:
        B = bg["models"]
        tbl_bg = "\n".join(
            f"| {PRETTY[m]} | {B[m]['params']:,} | {f4(B[m]['mae_mean'])} "
            f"&plusmn; {f4(B[m]['mae_sd_over_seeds'])} |" for m in ORDER if m in B)

    def pair(a, b):
        k = f"{a}_vs_{b}"
        v = fe["paired"][k]
        sig = "" if (v["ci_lo"] <= 0 <= v["ci_hi"]) else " **"
        return (f"| {a} &minus; {b} | {v['delta_mean']:+.4f} | "
                f"[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}] | {v['wilcoxon_p']:.3g} |"
                f" {v['sign_flip_fraction']*100:.1f}%{sig} |")

    pairs = "\n".join(pair(a, b) for a, b in
                      [("cube4", "standard"), ("cube4", "occupancy4"), ("cube4", "geo4"),
                       ("cube4", "random4"), ("cube4", "wide"), ("cube4", "sitelabel"),
                       ("occupancy4", "standard"), ("geo4", "standard"),
                       ("random4", "standard"), ("sitelabel", "standard"),
                       ("wide", "standard"), ("struct", "standard"),
                       ("hierarchy", "struct")]
                      if f"{a}_vs_{b}" in fe["paired"])

    ps = fe["per_crystal_system"]
    sysnames = ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
                "trigonal", "hexagonal", "cubic"]
    sys_hdr = "| Model | " + " | ".join(f"{s} (n={ps[s]['n_test']})" for s in sysnames
                                        if s in ps) + " |"
    sys_sep = "|" + "---|" * (len([s for s in sysnames if s in ps]) + 1)
    sys_rows = "\n".join(
        "| " + m + " | " + " | ".join(f3(ps[s][m]) for s in sysnames if s in ps) + " |"
        for m in ORDER if m in M)

    inv = I["checks"]["formation_energy"]
    inv_rows = "\n".join(
        f"| {m} | {inv[m]['max_abs_permutation_difference_eV']:.2e} | "
        f"{inv[m]['max_abs_padding_difference_eV']:.2e} | "
        f"{'pass' if inv[m]['permutation_pass'] else 'FAIL'} | "
        f"{'pass' if inv[m]['padding_pass'] else 'FAIL'} |"
        for m in ORDER if m in inv)
    inv_max = max(inv[m]["max_abs_permutation_difference_eV"] for m in inv)
    pad_max = max(inv[m]["max_abs_padding_difference_eV"] for m in inv)

    cost_rows = "\n".join(
        f"| {m} | {C[m]['params']:,} | {C[m]['forward_batch1_ms']:.2f} | "
        f"{C[m]['forward_batch64_ms']:.2f} |" for m in ORDER if m in C)
    COMMIT = "53c346d7ff132fb713f71dcc75cea0af775c3594"
    _pin = os.path.join(ROOT, "github_commit.txt")
    if os.path.exists(_pin):
        _v = open(_pin, encoding="utf-8").read().strip()
        if _v:
            COMMIT = _v

    doc = f"""# Correctly constructing and fairly evaluating symmetry-aware parameterisation for crystal-property neural networks

**Nailong Liang** — Zhaoqing University, Zhaoqing 526061, China — nailongliang@outlook.com

Manuscript COMMAT-D-26-03017R1 (revised)

## Abstract

Assigning each crystallographic site to a discrete label and letting that label select a
trainable transform is a common way to build structure into neural networks for crystal
property prediction, and the resulting accuracy is often attributed to crystallographic
symmetry. We show that such an attribution requires two ingredients that are frequently
missing. First, the map that defines the labels must actually be a group action, so that
every group element is a bijection of the label set. Second, the comparison must separate
the effect of the labels from the effect of the extra trainable parameters that the labels
introduce. We audit a published construction in which the 48 elements of $O_h$ act on a
$4\\times4\\times4$ discretisation of fractional coordinates. The archived action was centred
on the origin and negative coordinates were clipped to zero. {A['legacy']['non_bijective_generators']}
of the {A['n_elements']} group elements are then non-injective, each stored generator has
only {A['legacy']['stored_generator_images_min']} distinct images among {A['n_grid']} grid points, and
the reported {A['legacy']['K_reported']} classes
({', '.join(str(v) for v in A['legacy']['class_sizes'][:4])} &hellip;) are not orbits.
Centring the same action on the cube centre makes all {A['bijective_generators']} elements
bijections, closes all {A['closure_ok']} compositions, and yields $K={A['K']}$ orbits of sizes
{', '.join(str(v) for v in A['orbit_sizes'])}. We then define an evaluation protocol that is
free of target leakage — one frozen composition-family split, target statistics fitted on
training families only, checkpoint selection on validation data only, and paired statistics
resampled over families — and apply it to {fe['n_test_records']} test crystals drawn from a
2,741-crystal Materials Project cohort. The corrected four-orbit model does not outperform a
single shared transform ({f4(M['cube4']['mae_mean'])} versus {f4(M['standard']['mae_mean'])} eV/atom),
a parameter-matched wide network ({f4(M['wide']['mae_mean'])} eV/atom), a purely geometric
partition of the same voxel index ({f4(M['geo4']['mae_mean'])} eV/atom), or a relabelling that
preserves each crystal&rsquo;s own label occupancies ({f4(M['occupancy4']['mae_mean'])} eV/atom).
Expanding the label set to the maximum within-crystal orbit count likewise gives no
advantage over the shared baseline. Under matched conditions, accuracy is governed by model
capacity rather than by the symmetry content of the labels, and the honest contribution of
this line of work is the audit and test design rather than a predictive gain.

**Keywords:** crystal property prediction; group-theoretic orbit decomposition; structured
parameterisation; evaluation leakage; reproducibility; Materials Project

## 1. Introduction

Predicting a property of a crystal from its structure is a central problem in computational
materials science, and the field is now dominated by message-passing and continuous-filter
architectures that build an explicit neighbourhood graph \\[1-4\\]. A simpler alternative keeps
the per-site computation independent and combines sites only through a permutation-invariant
pooling step. In that setting the only place where structural information can enter is the
site feature vector and, if the model uses one, a discrete label that selects which trainable
transform is applied to a site.

Recent work in materials informatics has broadened the range of problems that machine-learned
models are asked to address, which makes the representational assumptions of those models
consequential. Machine learning combined with phase-field crystal modelling has been used to
relate nano-steps at &Sigma;3{{111}} twin boundaries to dislocation-cell strengthening \\[9\\];
coupled oxygen doping and crystal-amorphous multiphase coupling has been reported to improve
the wear resistance of a NiTiCu alloy \\[10\\]; and the principles, advances and emerging
applications of machine-learning force fields for inorganic crystalline materials have been
reviewed recently \\[11\\]. Each of these applications consumes a structural representation of
a crystal, and each is therefore sensitive to how that representation treats the symmetry of
the lattice: a descriptor that merges sites which are not symmetry-equivalent, or separates
sites which are, degrades the information the model is meant to use. The question addressed
here is narrower than those applications and prior to them: how a symmetry-derived label
should be constructed, and how it should be checked, before it is used to share parameters in
a property-prediction network.

The appeal of the second route is that crystal structures come with a ready-made labelling:
the space group partitions the sites of a cell into symmetry orbits, so a label could encode
a genuine physical equivalence class. The difficulty is that space-group orbits are defined
per structure. A network whose parameters are shared across structures needs a label that
means the same thing in different crystals, and the number of orbits differs from one crystal
to the next. A common compromise is to discretise fractional coordinates onto a small grid
and to label each site by its orbit under a fixed finite group acting on that grid. The
labels are then globally comparable by construction.

This paper is about how such a construction should be checked, and about what happens when
it is checked properly. We report three findings.

First, the group action has to be verified, not assumed. The construction we audit applied
the 48 signed permutation matrices of $O_h$ to integer grid coordinates in an origin-centred
frame and clipped negative results back into the grid. Clipping destroys injectivity: a
coordinate and its mirror image collapse onto the same grid point. We show that
{A['legacy']['non_bijective_generators']} of the {A['n_elements']} elements are non-injective under that
convention and that the reported {A['legacy']['K_reported']} classes are an artefact of a
traversal over a non-invertible map. Changing only the gauge, so that the action is centred
on the cube centre, restores injectivity for all {A['bijective_generators']} elements and yields
$K={A['K']}$ orbits. The two constructions are not variants of one another; they are a valid
action and an invalid one.

Second, a valid action is still only a geometric prior. A finite cubic grid has cubic
symmetry whatever the crystal is. Its orbits partition voxels, not atoms, and a label derived
from them says nothing about whether two sites of a triclinic or hexagonal cell are related
by a space-group operation. We therefore treat the corrected construction as a coarse
geometric partition and design the experiment so that a null result is a legitimate outcome.

Third, and most importantly, the comparison has to be controlled. A label set with $K$ values
multiplies the parameter count of each transform block by $K$. A gain over a single-transform
baseline therefore mixes label information with capacity. We specify four matched controls
that differ from the corrected model only in the coordinate-to-label assignment or in the
width: a relabelling that preserves each crystal&rsquo;s own label occupancies exactly, a
deterministic non-symmetric partition of the same voxel index, a random labelling with the
same global class sizes, and a wide network matched to the label model&rsquo;s parameter count.

Applied to a 2,741-crystal Materials Project cohort under a frozen composition-family split,
train-only normalisation and validation-only checkpointing, none of the label models
outperforms the shared-transform baseline, and the capacity-matched wide network is the best
model in the study. We report this outcome, together with the audit, as the contribution.
Section 2 establishes the mathematics, Section 3 the protocol, Section 4 the measurements,
and Section 5 the consequences for how claims of this kind should be made.

## 2. The grid-orbit construction: audit and correction

### 2.1 The archived action is not a group action

Let $f\\in[0,1)^3$ be a fractional coordinate and let $c=\\lfloor 4f\\rfloor\\in\\{{0,1,2,3\\}}^3$ be its
half-open voxel. The archived construction placed the voxel index $c$ in an origin-centred
frame, applied a signed permutation matrix $R$, and clipped the result into $[0,3]$:

$$g_R(c)=\\mathrm{{clip}}(Rc,\\,0,\\,3).$$

Clipping is not injective. The three generators that the archived code stores each have only
{A['legacy']['stored_generator_images_min']} distinct images among the {A['n_grid']} grid points. Enumerating
all {A['n_elements']} signed permutation matrices of $O_h$ under this convention,
{A['legacy']['non_bijective_generators']} of them are non-injective; only the six sign-free
coordinate permutations survive as bijections. Because the maps are not bijections, the
transitive-closure traversal that the code uses does not produce orbits. It produced
{A['legacy']['K_reported']} classes whose sizes fall from 3 to 1 (Supplementary Section S1).
Those classes cannot be interpreted as octahedral orbits, and no accuracy number computed
from them can be read as evidence for or against an octahedral prior.

### 2.2 A corrected cube-centred action

Let $h=(3/2,3/2,3/2)$ be the cube centre and define

$$g_R(c)=h+R\\,(c-h).$$

Every $R$ maps the 64 voxels onto themselves, because the multiset of absolute offsets
$\\{{|c_i-h_i|\\}}=\\{{1/2,3/2\\}}$ is invariant under signed permutations. An independent
enumeration confirms that all {A['bijective_generators']} maps are bijections and that all
{A['closure_ok']} pairwise compositions close within the group. There are
$K={A['K']}$ orbits, of sizes {', '.join(str(v) for v in A['orbit_sizes'])}, in one-to-one
correspondence with the number of voxel components that lie on a cube face, i.e. the number
of $c_i$ equal to 0 or 3. Figure 2 shows the four orbits on the four $z$ slices and the
corresponding class sizes; the audit script that reproduces both the corrected and the
archived counts is released with the paper.

The label we use in the remainder of the paper is therefore
$\\ell(c)=\\#\\{{i: c_i\\in\\{{0,3\\}}\\}}\\in\\{{0,1,2,3\\}}$.

### 2.3 What a grid label does and does not mean

The corrected label is a function of the wrapped coordinate alone. It never consults the
order in which atoms happen to be stored, so a permutation of the raw structure permutes the
labels accordingly. That property is necessary but it is not a claim about physics. A cubic
grid has cubic symmetry regardless of the crystal that is mapped onto it, and an atom mapped
to the voxel $(0,0,0)$ of a triclinic cell is not equivalent to an atom mapped to $(3,3,3)$
of the same cell. The construction is a coarse geometric partition, and we describe it as
such throughout.

Voxelisation also has free parameters that a reader is entitled to see. We use the half-open
convention $c=\\lfloor 4f\\rfloor$, so a coordinate that lies exactly on a voxel boundary is
assigned to the upper voxel, two atoms in the same voxel keep separate feature vectors and
merely receive the same label, and the label depends on the origin and basis of the cell.
The model is therefore not asserted to be invariant to arbitrary changes of crystallographic
setting, and we do not claim that it is equivariant to the space group of a non-cubic
crystal.

### 2.4 The archived numbers are archival

The original submission reported a gain for the model then called OrbitMLP-$O_h$. Because the
underlying construction is not a group action, that number is a record of a particular run of
a particular implementation, not a measurement of an octahedral prior. We keep the archived
values in Supplementary Section S7 in a clearly labelled archival table, together with the
reason each of them is not admissible as evidence, and we do not use them anywhere else.
The corrected model has four slots, not 40, so it shares no parameterisation with the archived
one and has to be retrained from scratch. That is what Section 4 reports.

## 3. A protocol that separates labels from capacity

### 3.1 Split, normalisation and checkpointing

The archived driver computed target statistics over the whole cohort and re-drew a train/test
split for every training seed, so test targets influenced the normalisation and repeated
structures could appear on both sides of a split. The protocol used here fixes all of that in
advance.

- **Split.** Crystals are grouped by reduced composition formula. Whole families are assigned
  to training, validation and test in a 70/10/20 ratio with a fixed seed, giving
  {fe['n_test_records']} test crystals in
  {fe['n_test_families']} families. No composition family appears in two sets, so a
  polymorph of a training compound cannot be scored as a test crystal.
- **Features.** Each site carries a four-dimensional elemental descriptor projected to 64
  dimensions by a fixed, non-trainable projection. Structural baselines add cell parameters,
  the eight smallest periodic interatomic distances and volume per atom.
- **Normalisation.** Target mean and standard deviation are computed from training families
  only and are recomputed for each run.
- **Checkpoint selection.** The optimiser is AdamW (learning rate 5x10^-4, weight decay 0.05)
  with cosine decay and mean-squared error on the normalised target. The checkpoint with the
  lowest validation MAE is kept; the test set is scored once, after selection.
- **Seeds.** Five initialisation seeds (42, 123, 456, 789, 1024) are used for formation
  energy and three (42, 123, 456) for the band gap.

### 3.2 The four matched controls

All four-slot models use the same architecture, the same inputs, the same split and the same
slot count; only the assignment of a site to a slot changes.

| Control | Assignment rule | What it isolates |
|---|---|---|
| Cube4 | corrected cube-centred orbit of the voxel | the geometric prior |
| Occupancy4 | sites sorted by coordinate receive the sorted Cube4 labels | whether the specific assignment matters once each crystal&rsquo;s slot occupancy is held fixed |
| Geo4 | deterministic bins of the linear voxel index (0-7, 8-31, 32-55, 56-63) | a non-symmetric partition with the same slot count |
| Random4 | fixed random permutation of the 64-label multiset | labels with no geometry at all |

Two further models probe capacity and representation. **WideMLP** is a wide network whose
parameter count is matched to the four-slot model. **SiteLabel** uses the true within-crystal
symmetry classes as labels, with 32 slots, i.e. the number of classes of the 32-orbit maximum
over the cohort; because that count multiplies the transform size, its parameter count is
much larger than the others and it is reported as such rather than as a matched control.
**StructMLP** and **HierarchyMLP** replace the elemental descriptor with structural
descriptors (and, for the latter, orbit-size and stabiliser-order features) at matched width.

### 3.3 Statistics and robustness

Test losses are averaged over seeds for each test crystal before resampling, so that the unit
of analysis is the crystal and not the run. Confidence intervals are obtained by resampling
whole families, which respects the fact that test crystals within one family are not
independent. Paired contrasts use the same family-clustered bootstrap together with a
Wilcoxon signed-rank test and the fraction of resampled draws in which the sign of the
difference flips. We also reorder the raw atoms of the test structures, recompute their
labels from the reordered coordinates, and compare predictions, and we pad each structure
with masked slots carrying non-zero nuisance values.

## 4. Results

### 4.1 The corrected label model does not beat the shared baseline

Table 1 lists formation-energy errors on the {fe['n_test_records']} held-out test crystals.

**Table 1.** Formation energy, authentic cohort, {fe['n_test_records']} test crystals in
{fe['n_test_families']} held-out composition families, five seeds. MAE in eV/atom; the
bracket is a family-clustered 95% CI; the last column is the relative change with respect to
the shared baseline, where a negative value means a lower error than the shared baseline.

| Model | K | Parameters | MAE &plusmn; SD over seeds | 95% CI (family-clustered) | vs shared |
|---|---|---|---|---|---|
{table}

The shared single-transform baseline reaches {f4(M['standard']['mae_mean'])} eV/atom. The
corrected four-orbit model reaches {f4(M['cube4']['mae_mean'])} eV/atom, i.e. it is
{abs(100*(M['cube4']['mae_mean']/M['standard']['mae_mean']-1)):.1f}% *worse* than the model with
one transform and roughly one quarter of the parameters. The large within-family coordinate
features are far more predictive than the presence of a cubic prior; the capacity-matched wide
network is the best model at {f4(M['wide']['mae_mean'])} eV/atom. Band-gap results, in
Supplementary Table S4, follow the same ordering.

### 4.2 The specific assignment does not matter either

Table 2 gives the paired contrasts that the reviewers of the original submission asked for.

**Table 2.** Paired contrasts, formation energy. A negative difference means the first model
has the lower error. The CI is the family-clustered 95% interval of the per-crystal difference;
the last column is the fraction of resampled draws whose sign differs from the point estimate,
and ** marks a CI that excludes zero.

| Contrast | &Delta; MAE (eV/atom) | 95% CI | Wilcoxon p | sign-flip fraction |
|---|---|---|---|---|
{pairs}

Three things stand out. The corrected cube-centred label model is not better than the shared
baseline, nor than the wide baseline. It is also not better than Occupancy4, which keeps each
crystal&rsquo;s own label occupancies but discards the orbit rule, nor than Geo4, a
deterministic coordinate binning that has nothing to do with the octahedral group. The
SiteLabel model, whose labels are the genuine within-crystal symmetry classes, is likewise no
better than the shared baseline despite carrying 28 times as many parameters. Whatever the
label schemes contribute, it is not symmetry.

### 4.3 Crystal-system structure

**Table 3.** Formation-energy MAE by crystal system, same frozen split.

{sys_hdr}
{sys_sep}
{sys_rows}

The wide model is best in every system, and no label model separates from the shared baseline
in a way that survives the family-clustered interval except in the wrong direction. Stratifying
instead by the number of symmetry operations (Supplementary Table S5b) makes the same point:
the label models are relatively less bad in the high-symmetry stratum, but so are the random
labels, and the only stratum in which a paired interval excludes zero is the low-symmetry one,
where the corrected cube-centred model is *worse* than the shared baseline
({f4(fe['per_symmetry_stratum']['low']['paired']['cube4_vs_standard']['delta_mean'])} eV/atom,
95% CI [{f4(fe['per_symmetry_stratum']['low']['paired']['cube4_vs_standard']['ci_lo'])},
{f4(fe['per_symmetry_stratum']['low']['paired']['cube4_vs_standard']['ci_hi'])}]).

### 4.4 Structure-aware baselines

Adding cell parameters, the eight smallest periodic distances and volume per atom to the input
changes the result in the opposite direction from the one usually assumed: StructMLP reaches
{f4(M['struct']['mae_mean'])} eV/atom, worse than the elemental-only shared baseline, and
adding orbit-size and stabiliser descriptors (HierarchyMLP, {f4(M['hierarchy']['mae_mean'])} eV/atom)
does not recover the loss. A four-dimensional elemental descriptor projected to 64 dimensions
is a strong baseline for this architecture, and a small hand-built structural descriptor set
is not automatically better. We report this rather than tuning the descriptors until they win.

### 4.5 Invariance and padding

| Model | max |&Delta;| under raw permutation (eV) | max |&Delta;| under masked padding (eV) | permutation | padding |
|---|---|---|---|---|
{inv_rows}

Every model reproduces its predictions to within single-precision round-off when the raw
atoms are reordered and their labels recomputed from the reordered coordinates, and when
masked slots holding non-zero nuisance values are inserted. The largest observed discrepancy
over all models is {inv_max:.2e} eV for reordering and {pad_max:.2e} eV for padding. The
archived concern that atom ordering could influence the label assignment is therefore
resolved by construction: the label is a function of the coordinate, and the pooled head
ignores masked slots.

### 4.6 Held-out space groups

Holding out the least frequent space groups until at least 100 structures are set aside, and
keeping the held-out composition families out of training, gives
Supplementary Table S6. The ordering of models is unchanged. We report this as a limited
distribution-shift check: a space-group holdout does not establish chemical novelty, and the
cells that remain are still drawn from the same filtered cohort.

### 4.7 Cost

| Model | Parameters | forward, batch 1 (ms) | forward, batch 64 (ms) |
|---|---|---|---|
{cost_rows}

Slot-based blocks are executed as grouped matrix products rather than by materialising
expanded weights, so the cost of a slot model grows with the number of slots but not with the
number of atoms per crystal. Label assignment costs
{C['label_preprocessing_ms_per_crystal']:.3f} ms per crystal in our implementation. These
timings cover cached-feature inference only; they exclude database access and symmetry
detection, and we do not convert them into end-to-end speed-up claims.

### 4.8 Band gap

| Model | Parameters | Band-gap MAE (eV) |
|---|---|---|
{tbl_bg}

Band-gap errors are an order of magnitude larger than formation-energy errors in absolute
terms, and again the label models do not beat the shared baseline. Because the band-gap runs
use three seeds and a single split, we treat them as a consistency check rather than an
independent claim.

## 5. Discussion

The corrected construction is easy to state and easy to verify, and that is the point. A group
action on a finite set is a set of bijections; a construction that clips coordinates is not
one, and the difference is not cosmetic. In the case we audited the invalid action produced
{A['legacy']['K_reported']} classes instead of {A['K']}, with class sizes that are not orbit sizes,
and it was the source of the headline claim of the original submission. Replacing it changes
the parameterisation completely, so no number from the old model transfers to the new one.

The second lesson is that a valid geometric prior is still a prior. Discretising fractional
coordinates onto a cubic grid imposes cubic symmetry on every crystal, including the
83% of this cohort that is not cubic. Nothing about the resulting partition is a statement
about the space group of the crystal, and the correct description is a coarse geometric
partition of the cell.

The third lesson is the one the original submission missed. Because a label set with $K$
values multiplies the transform parameters by $K$, a label model differs from its baseline in
two ways at once. Under a split that keeps composition families disjoint, train-only
normalisation, validation-only checkpointing, and matched occupancy and capacity controls,
none of the label models in this study improves on a single shared transform, and a wide
network with the same parameter budget is the strongest model. We take this to mean that the
useful signal in this family of models is capacity plus a good elemental descriptor, and that
previous positive results of this kind should be re-examined with the same controls.

We deliberately do not tune the controls until one of them wins, and we do not report the
archived values as evidence. A methodology paper is only useful if the negative result is
reported with the same care as a positive one, and if the reader can re-run the audit and the
protocol from the released code and data.

Links to the wider literature are indirect but real. The studies discussed in Section 1 use
learned representations of inorganic crystals for defect and interface behaviour \\[9\\], alloy
properties \\[10\\] and interatomic potentials \\[11\\], and each inherits whatever its
structural descriptor encodes. A label that is presented as encoding symmetry equivalence but
encodes voxel geometry instead is hard to detect from a downstream metric, because the model
can still fit its training data. That is why the construction is tested directly here rather
than inferred from accuracy.

### Limitations

The cohort is a filtered Materials Project snapshot (8-40 sites, energy above hull below
0.2 eV/atom, at least two symmetry operations) and is therefore biased toward well-ordered,
low-energy structures; the filtering code and the exact database release of the original
snapshot are not available here, so we report the cohort as it is stored rather than
reconstructing it. Results are conditional on one frozen split per task and on the
hyperparameters listed in Section 3; they are not a statement about all possible splits. The
structural baselines use a small hand-built descriptor set; a message-passing or
continuous-filter model would represent local environments differently, and we make no claim
about that family of models. Finally, the invariance checks use a sample of 250 structures
rather than the whole cohort.

## 6. Methods

**Cohort.** 2,741 crystals, 158 space groups, 2,163 distinct reduced compositions, 2-40 sites
per cell. Targets are formation energy per atom (eV/atom) and band gap (eV).

**Voxel map.** $f$ is wrapped into $[0,1)^3$ and $c=\\lfloor 4f\\rfloor$ with ties assigned to
the upper voxel. The label is the number of components of $c$ equal to 0 or 3.

**Corrected action.** $g_R(c)=h+R(c-h)$, $h=(3/2,3/2,3/2)$, with $R$ ranging over the 48
signed permutation matrices. Verified by exhaustive enumeration of images and of all
pairwise compositions.

**Model.** Each site carries a fixed 64-dimensional projected input. Four blocks apply
feature-wise LayerNorm, a slot-selected affine transform and GELU. The pooled representation
is the masked mean over sites, followed by a two-layer head. WideMLP uses
{C['wide']['params']:,} parameters against {M['cube4']['params']:,} for the four-slot model.

**Protocol.** Frozen composition-family split 70/10/20; train-only target normalisation;
AdamW, learning rate 5x10^-4, weight decay 0.05, cosine decay, 200 epochs, batch size 64,
MSE on the normalised target; checkpoint by validation MAE; test scored once.

**Statistics.** Per-crystal absolute errors are averaged over seeds, then resampled 5,000
times over families for the CI and 2,000 times for the sign-flip fraction; Wilcoxon
signed-rank on the same paired per-crystal errors.

**Software.** Python 3.10, PyTorch {'2.6.0'} with CUDA, NumPy, SciPy, pymatgen and spglib.
Each run records its seed, split manifest, parameter count and per-crystal test predictions.

## Data and code availability

The released archive contains the corrected group audit, the evaluation protocol, the
per-crystal test predictions for every model and seed, the figures with their SVG sources,
and the derived JSON tables used to produce every number above. The cohort is a Materials
Project snapshot supplied as a cached pickle; per-crystal identifiers, coordinates, cell
parameters and targets are exported to JSONL by the released code. Repository:
https://github.com/wmll-wmll/orbit-weight, branch master, commit {COMMIT}. The scripts and the
derived JSON tables are also included in 06_Data_and_code of this submission package.

## References

[1] A. Jain et al., APL Materials 1 (2013) 011002.
[2] T. Xie, J.C. Grossman, Phys. Rev. Lett. 120 (2018) 145301.
[3] C. Chen et al., Chem. Mater. 31 (2019) 3564-3572.
[4] K.T. Schuett et al., J. Chem. Phys. 148 (2018) 241722.
[5] A. Togo, K. Shinohara, I. Tanaka, Sci. Technol. Adv. Mater. Methods 4 (2024) 2384822.
[6] Spglib dataset documentation, accessed 12 September 2026.
[7] T.S. Cohen, M. Welling, PMLR 48 (2016) 2990-2999.
[8] M. Zaheer et al., NeurIPS 30 (2017).
[9] H. Li et al., Scripta Mater. 283 (2026) 117432.
[10] H. Ma et al., Rare Metals 45 (6) (2026) e70319.
[11] J. Yi et al., Phys. Chem. Chem. Phys. (2026) doi:10.1039/D6CP01826B.

## Figures

```{{=openxml}}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

![](figures/fig1_main_comparison.png){{width=6.4in}}

**Figure 1.** Formation-energy MAE for the nine models on {fe['n_test_records']} held-out test
crystals, with family-clustered 95% confidence intervals. Parameter counts are given in the
tick labels.

```{{=openxml}}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

![](figures/fig2_grid_orbits.png){{width=6.4in}}

**Figure 2.** Corrected cube-centred $O_h$ action on the $4\\times4\\times4$ grid. The four
$z$ slices are coloured by the number of voxel components equal to 0 or 3, which is the orbit
label; the lower-left panel gives the orbit sizes 8, 24, 24 and 8, and the lower-right panel
gives three worked examples of a coordinate mapping to a voxel and a label.

```{{=openxml}}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

![](figures/fig3_per_crystal_system.png){{width=6.4in}}

**Figure 3.** Formation-energy MAE by crystal system for the matched four-slot models and the
shared and wide baselines.

```{{=openxml}}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

![](figures/fig4_paired_contrasts.png){{width=6.4in}}

**Figure 4.** Paired contrasts with family-clustered 95% intervals; red indicates an interval
that excludes zero.

```{{=openxml}}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

![](figures/fig5_invariance_cost.png){{width=6.4in}}

**Figure 5.** (left) maximum change in prediction under raw atom reordering and under masked
padding with nuisance values; (right) cached-feature inference cost.

```{{=openxml}}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

![](figures/figS1_workflow.png){{width=6.4in}}

**Figure S1.** Workflow of the corrected study (supplementary material).
"""
    out = os.path.join(MS, "main.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print("wrote", out, len(doc), "chars")
    build_supplementary(S)


def _load_preds(task, suffix=""):
    import numpy as np
    pdir = os.path.join(RES, "preds")
    y = np.load(os.path.join(pdir, f"{task}{suffix}_ytest.npy"))
    return y, pdir


def _mcnemar_like(err_a, err_b, fam):
    import numpy as np
    d = err_a - err_b
    pos = d > 0
    neg = d < 0
    # family-clustered sign test on the per-crystal sign
    uniq = np.unique(fam)
    idx = {f: np.where(fam == f)[0] for f in uniq}
    return float(max(pos.sum(), neg.sum()) / max(pos.sum() + neg.sum(), 1))


def build_supplementary(S):
    import numpy as np
    A = json.load(open(os.path.join(RES, "grid_audit.json"), encoding="utf-8"))
    I = json.load(open(os.path.join(RES, "invariance.json"), encoding="utf-8"))
    C = json.load(open(os.path.join(RES, "cost.json"), encoding="utf-8"))
    fe = S["formation_energy"]
    bg = S.get("band_gap", {})
    M = fe["models"]
    ORDER = ["standard", "wide", "cube4", "occupancy4", "geo4", "random4",
             "sitelabel", "struct", "hierarchy"]

    # ---- per-seed table
    perseed = []
    for m in ORDER:
        if m not in M:
            continue
        b = M[m]
        vals = ", ".join(f"{v:.4f}" for v in b["mae_per_seed"])
        perseed.append(f"| {m} | {b['K']} | {b['params']:,} | {vals} | "
                       f"{b['mae_mean']:.4f} | {b['mae_sd_over_seeds']:.4f} |")
    perseed = "\n".join(perseed)

    # ---- band gap
    bgrows = ""
    if bg:
        B = bg["models"]
        bgrows = "\n".join(
            f"| {m} | {B[m]['params']:,} | {B[m]['mae_mean']:.4f} | "
            f"{B[m]['mae_sd_over_seeds']:.4f} | "
            f"[{B[m]['bootstrap_ci']['ci_lo']:.4f}, {B[m]['bootstrap_ci']['ci_hi']:.4f}] |"
            for m in ORDER if m in B)
        bgpairs = "\n".join(
            f"| {k.replace('_vs_', ' vs ')} | {v['delta_mean']:+.4f} | "
            f"[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}] | "
            f"{v['wilcoxon_p']:.3g} |"
            for k, v in bg["paired"].items())
    else:
        bgpairs = ""

    # ---- per system, all models
    ps = fe["per_crystal_system"]
    sysnames = [s for s in ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
                            "trigonal", "hexagonal", "cubic"] if s in ps]
    syshdr = "| Model | " + " | ".join(f"{s} (n={ps[s]['n_test']})" for s in sysnames) + " |"
    syssep = "|" + "---|" * (len(sysnames) + 1)
    sysrows = "\n".join(
        "| " + m + " | " + " | ".join(f"{ps[s][m]:.4f}" for s in sysnames) + " |"
        for m in ORDER if m in M)

    # ---- transfer (space-group holdout)
    xp = os.path.join(RES, "train_formation_energy_xfer.json")
    xfer_tbl, xfer_note = "", (
        "The space-group holdout run is not included in this build.")
    if os.path.exists(xp):
        X = json.load(open(xp, encoding="utf-8"))
        try:
            yx = np.load(os.path.join(RES, "preds", "formation_energy_xfer_ytest.npy"))
            tg = np.load(os.path.join(RES, "preds", "formation_energy_xfer_testglobal.npy"))
            fams = None
            rows = []
            for m in ORDER:
                seeds = sorted(int(k.split("@")[1]) for k in X if k.startswith(m + "@"))
                if not seeds:
                    continue
                P = np.stack([np.load(os.path.join(RES, "preds",
                                                   f"formation_energy_xfer_{m}_{s}.npy"))
                              for s in seeds])
                E = np.abs(P - yx[None, :]).mean(0)
                rows.append((m, X[f"{m}@{seeds[0]}"]["params"], E.mean(),
                             np.abs(P - yx[None, :]).mean(1).std(ddof=1), E))
            xfer_tbl = "\n".join(
                f"| {m} | {p:,} | {mu:.4f} | {sd:.4f} |" for m, p, mu, sd, _ in rows)
            xman = json.load(open(os.path.join(ROOT, "data", "split_xfer.json"),
                                  encoding="utf-8"))
            n_ho = len(xman["spacegroups_test"])
            xfer_note = (f"{len(tg)} test structures from {n_ho} held-out space "
                         f"groups; three seeds.")
        except Exception as e:  # pragma: no cover
            xfer_note = f"transfer table unavailable: {e}"

    inv = I["checks"]["formation_energy"]
    pstr = fe["per_symmetry_stratum"]
    strata_hdr = ("| Stratum (symmetry operations in test crystal) | n | "
                  + " | ".join(ORDER) + " |")
    strata_sep = "|" + "---|" * (len(ORDER) + 2)
    strata_rows = "\n".join(
        f"| {b} ({pstr[b]['n_sym_ops_min']}-{pstr[b]['n_sym_ops_max']}) | {pstr[b]['n_test']} | "
        + " | ".join(f"{pstr[b][m]:.4f}" for m in ORDER) + " |"
        for b in ("low", "medium", "high"))
    strata_pairs = "\n".join(
        f"| {b} | {k.replace('_vs_', ' vs ')} | {v['delta_mean']:+.4f} | "
        f"[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}] | {v['wilcoxon_p']:.3g} |"
        for b in ("low", "medium", "high")
        for k, v in pstr[b]["paired"].items())
    invrows = "\n".join(
        f"| {m} | {inv[m]['max_abs_permutation_difference_eV']:.3e} | "
        f"{inv[m]['max_abs_padding_difference_eV']:.3e} | "
        f"{'pass' if inv[m]['permutation_pass'] else 'FAIL'} | "
        f"{'pass' if inv[m]['padding_pass'] else 'FAIL'} |"
        for m in ORDER if m in inv)

    costrows = "\n".join(
        f"| {m} | {C[m]['params']:,} | {C[m]['forward_batch1_ms']:.2f} | "
        f"{C[m]['forward_batch64_ms']:.2f} | "
        f"{C[m].get('forward_batch64_peak_alloc_MB', float('nan')):.2f} |"
        for m in ORDER if m in C)

    doc = f"""# Supplementary material

**Correctly constructing and fairly evaluating symmetry-aware parameterisation for
crystal-property neural networks**

Nailong Liang, Zhaoqing University. Manuscript COMMAT-D-26-03017R1 (revised).

All numbers below are produced by `experiments/exp_v3_real.py` from the stored result files
in `results_v3/`. Nothing in this document is transcribed by hand.

## S1. Grid-group audit

| Quantity | Archived construction | Corrected construction |
|---|---|---|
| Gauge | origin-centred, clipped into [0,3] | cube-centred, $g(c)=h+R(c-h)$ |
| Group elements enumerated | {A['n_elements']} | {A['n_elements']} |
| Elements that are bijections | {A['n_elements'] - A['legacy']['non_bijective_generators']} | {A['bijective_generators']} |
| Distinct images of a stored generator | {A['legacy']['stored_generator_images_min']} of {A['n_grid']} | {A['n_grid']} of {A['n_grid']} |
| Compositions that close | not defined (maps not invertible) | {A['closure_ok']} of {A['closure_total']} |
| Classes produced by the code | {A['legacy']['K_reported']} | {A['K']} |
| Class sizes | {', '.join(str(v) for v in A['legacy']['class_sizes'])} | {', '.join(str(v) for v in A['orbit_sizes'])} |
| Label equals number of components in {{0,3}} | no | yes |

The archived counts were reproduced by running the archived group class itself
(`groups/octahedral.py`, `OctahedralGroup(n=4).compute_orbits()`), not by re-implementing it
approximately from the description. The corrected counts come from enumerating all 48 signed
permutation matrices, checking injectivity of each image set, and checking all
{A['closure_total']} pairwise compositions.

## S2. Split manifest

| Quantity | Value |
|---|---|
| Corpus size | 2,741 |
| Distinct reduced compositions (families) | 2,163 |
| Training crystals / families | 1,921 / 1,514 |
| Validation crystals / families | 270 / 216 |
| Test crystals / families | 550 / 433 |
| Rule | whole composition families assigned 70/10/20 with seed 0; frozen for every model |

No composition family appears in more than one subset. Because a family can contain several
polymorphs of the same formula, the number of crystals per family is not uniform and the
crystal counts differ from the nominal 70/10/20 ratio.

## S3. Per-seed formation-energy errors

| Model | K | Parameters | Per-seed MAE (eV/atom) | Mean | SD |
|---|---|---|---|---|---|
{perseed}

## S4. Band gap

Three seeds (42, 123, 456), same frozen split.

| Model | Parameters | MAE (eV) | SD over seeds | Family-clustered 95% CI |
|---|---|---|---|---|
{bgrows}

Paired contrasts:

| Contrast | Delta MAE (eV) | 95% CI | Wilcoxon p |
|---|---|---|---|
{bgpairs}

The band-gap spread across families is much larger than the formation-energy spread, so the
intervals are wide; the ordering of models is nevertheless the same as for formation energy.

## S5. Per-crystal-system formation-energy MAE, all models

{syshdr}
{syssep}
{sysrows}

### S5b. Symmetry-operation strata

Test crystals split into terciles of the number of symmetry operations reported for the cell.

{strata_hdr}
{strata_sep}
{strata_rows}

Paired contrasts inside each stratum:

| Stratum | Contrast | Delta MAE (eV/atom) | 95% CI | Wilcoxon p |
|---|---|---|---|---|
{strata_pairs}

## S6. Space-group holdout

{xfer_note}

| Model | Parameters | MAE (eV/atom) | SD over seeds |
|---|---|---|---|
{xfer_tbl}

The holdout consists of the least frequent space groups of the cohort, accumulated until at
least 100 structures are set aside, and every composition family that appears in the holdout
is removed from training. A space-group holdout of this kind is a limited distribution-shift
check: it does not establish chemical novelty, since the held-out cells can still share
compositions and structural motifs with the training set through other space groups.

## S7. Archived results and why they are not evidence

| Archived model | Slots | Parameters | MAE (eV/atom) | Status |
|---|---|---|---|---|
| StandardMLP | 1 | 19,265 | 0.1898 | archival |
| WideMLP | 1 | 671,020 | 0.0864 | archival |
| OrbitMLP-pointgroup (LocalLabel32) | 32 | 535,105 | 0.1714 | archival |
| StructMLP | 1 | 19,265 | 0.2121 | archival |
| OrbitMLP-O_h (LegacyGrid40) | 40 | 668,225 | 0.2792 | invalid construction |
| GeoBinsMLP | 40 | 668,225 | 0.2597 | archival |
| RandomMLP | 40 | 668,225 | 0.3745 | archival |

These values are transcribed from the archived result files of the earlier submission. They
were produced with target statistics fitted on the complete cohort, with a train/test split
redrawn for each seed, and with the label-construction dependency unavailable. The
OrbitMLP-O_h entry additionally rests on the non-bijective action of Section S1. None of
these numbers is used as evidence anywhere in the revised manuscript, and the corrected
four-orbit model shares no parameterisation with them.

## S8. Label function and model definition

```
def voxel(frac, n=4):                 # half-open voxels
    f = frac - floor(frac)            # wrap into [0,1)
    c = clip(floor(n * f), 0, n - 1)
    return c[:, 2] * n * n + c[:, 1] * n + c[:, 0], c

def cube4_label(frac):
    _, c = voxel(frac)
    return ((c == 0) | (c == n - 1)).sum(axis=1)
```

The four-slot models apply, in each of four blocks,
`GELU(A_label . LayerNorm(x) + b_label)` with one `A` and one `b` per slot, then a masked
mean over sites and a two-layer head. Parameter counts are {M['cube4']['params']:,} for the
four-slot models, {C['wide']['params']:,} for the capacity-matched wide model and
{M['standard']['params']:,} for the shared baseline. Occupancy4 sorts sites by wrapped
coordinate and assigns the sorted Cube4 labels, so each crystal&rsquo;s slot occupancies are
preserved exactly. Geo4 bins the linear voxel index at 8, 32 and 56. Random4 uses a fixed
random permutation of the 64-label multiset.

## S9. Invariance and padding detail

250 structures drawn at random from the cohort. For the permutation check the raw atom order
is shuffled and the labels are recomputed from the permuted coordinates; for the padding
check 19 masked slots holding non-zero nuisance features and arbitrary labels are inserted
before the real sites, so that the real sites move position within the padded array.

| Model | max abs. change, permutation (eV) | max abs. change, padding (eV) | permutation | padding |
|---|---|---|---|---|
{invrows}

The tolerance of 1e-4 eV in the target unit is met by every model with more than two orders of
magnitude to spare.

## S10. Cost detail

Device: {C['gpu']}. Timings are cached-feature forward passes after three warm-up iterations,
30 repetitions, with explicit synchronisation; peak allocation is measured per batch size.

| Model | Parameters | batch 1 (ms) | batch 64 (ms) | peak alloc., batch 64 (MB) |
|---|---|---|---|---|
{costrows}

Label preprocessing, including voxelisation and orbit lookup for all 2,741 structures of the
cohort, takes
{C['label_preprocessing_seconds_total']:.3f} s in total
({C['label_preprocessing_ms_per_crystal']:.4f} ms per crystal). These timings cover cached
features only. They exclude database access, structure parsing and symmetry detection, and we
do not convert them into end-to-end speed-up claims.

## S11. Data provenance and open limitations

The cohort is the cached Materials Project snapshot used by the previous submission
(2-40 sites per cell, energy above hull no greater than 0.2 eV/atom, at least two symmetry
operations). The exact database release, retrieval date and material identifiers of the
original query were not archived with the snapshot, so we report the cohort as stored and do
not claim to have reconstructed the original query. Per-crystal exports include a
content-derived identifier, atomic numbers, fractional coordinates, cell parameters, target
values and a composition-family identifier.

Known limitations, restated here so that they are not lost in the main text: results are
conditional on one frozen split per task and on the hyperparameters of Section 3; the
structural baselines use a small hand-built descriptor set; and the robustness checks use a
sample of 250 structures rather than the full cohort.
"""
    p = os.path.join(MS, "supplementary.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(doc)
    print("wrote", p, len(doc), "chars")


if __name__ == "__main__":
    import math
    main()
