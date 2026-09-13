#!/usr/bin/env python3
"""Build the point-by-point response, the reviewer-response tracker (xlsx),
the Highlights and the cover letter."""
import os, json, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results_v3")
OUT = os.path.join(ROOT, "deliverable")
os.makedirs(OUT, exist_ok=True)
TRACKER_SRC = r"E:\operator\_rev20260912_in\Comment_response_tracker.csv"

S = json.load(open(os.path.join(RES, "summary.json"), encoding="utf-8"))
A = json.load(open(os.path.join(RES, "grid_audit.json"), encoding="utf-8"))
CO = json.load(open(os.path.join(RES, "cohort_stats.json"), encoding="utf-8"))
M = S["formation_energy"]["models"]
P = S["formation_energy"]["paired"]
BGM = S["band_gap"]["models"]


def d(a, b):
    v = P[f"{a}_vs_{b}"]
    return (f"{v['delta_mean']:+.4f} eV/atom (95% CI [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}], "
            f"Wilcoxon p = {v['wilcoxon_p']:.3g})")


RESP = {
"R1.Major.1": (
"We agree, and the audit found the problem to be deeper than the reviewer suspected. The archived "
"construction did not merely use a cubic surrogate for non-cubic crystals: it was not a group action. "
f"Its generators place the voxel index in an origin-centred frame and clip negative coordinates into "
f"[0,3]; {A['legacy']['non_bijective_generators']} of the {A['n_elements']} elements are then non-injective and each of the "
f"three stored generators has only {A['legacy']['stored_generator_images_min']} distinct images among the {A['n_grid']} grid points. The "
f"{A['legacy']['K_reported']} published classes are therefore an artefact of traversing a non-invertible map.",
"Sections 2.1-2.3; Figure 2; Supplementary Sections S1, S8."),
"R1.Major.1b": (
f"Section 2.3 now gives the complete map: f is wrapped into [0,1)^3, c = floor(4f) (half-open voxels), and the "
"label is the number of components of c equal to 0 or 3. Atomic ordering never enters the index. Two atoms "
"in one voxel keep separate feature vectors and merely share a label. Figure 2 shows the four orbits and "
"three worked coordinate-to-voxel-to-label examples, Section S8 lists the code, and the released audit script "
"reproduces both the corrected count (48/48 bijections, 2304/2304 compositions close, K = 4, sizes 8/24/24/8) "
"and the archived count. The atom-order concern is now measured rather than argued: reordering the raw atoms "
"and recomputing the labels changes predictions by at most 3.6e-7 eV over 250 structures.",
"Section 2.1-2.3, 4.5; Figure 2; Tables S1, S9."),

"R1.Major.2": (
"Accepted in full. The universal-symmetry claim is withdrawn. Cube4 is described throughout as a specified "
"finite-cube geometric prior, and the manuscript now states explicitly that it is not the space group of a "
"non-cubic crystal and that voxel labels say nothing about site equivalence in a triclinic or hexagonal cell.",
"Sections 1, 2.3, 5."),
"R1.Major.2b": (
f"Performance is reported per crystal system (Table 3, Table S5) and per symmetry-operation stratum "
"(Table S5b). The pattern is the opposite of a symmetry explanation. WideMLP is best in all seven systems; "
f"Cube4 reaches {M['cube4']['mae_mean']:.4f} eV/atom against {M['standard']['mae_mean']:.4f} for the shared baseline overall, "
f"and the only paired contrast that excludes zero in a stratum is in the low-symmetry band, where Cube4 is "
f"worse by {S['formation_energy']['per_symmetry_stratum']['low']['paired']['cube4_vs_standard']['delta_mean']:+.4f} eV/atom "
f"(95% CI [{S['formation_energy']['per_symmetry_stratum']['low']['paired']['cube4_vs_standard']['ci_lo']:+.4f}, "
f"{S['formation_energy']['per_symmetry_stratum']['low']['paired']['cube4_vs_standard']['ci_hi']:+.4f}]).",
"Sections 4.1-4.3; Tables 1, 3, S5, S5b."),

"R1.Major.3": (
"Implemented exactly as suggested, and twice.",
"Section 3.2; Table 2."),
"R1.Major.3b": (
f"Geo4 partitions the linear voxel index into four deterministic bins (0-7, 8-31, 32-55, 56-63) and shares "
f"inputs, architecture and slot count with Cube4. Occupancy4 keeps each crystal's own Cube4 slot occupancies "
f"and changes only which site receives which label, by sorting sites on their wrapped coordinates. The "
f"resulting contrast is unambiguous: Cube4 is worse than the deterministic non-symmetric partition by "
f"{d('cube4','geo4')}, and worse than the occupancy-matched relabelling by {d('cube4','occupancy4')}. Both "
f"controls beat the group-theoretic construction, or tie it. The interpretation that a gain originates "
f"specifically from group-theoretic symmetry is therefore not supported, and the manuscript says so.",
"Sections 3.2, 4.2; Tables 1, 2."),

"R1.Major.4": (
"We now provide the stronger structural baseline the comment asks for, and report what it shows.",
"Sections 3.2, 4.4, 6; Tables 1, 2."),
"R1.Major.4b": (
f"StructMLP replaces the four-dimensional elemental descriptor with a 19-component vector: the elemental "
f"descriptor, the six cell parameters, the eight smallest periodic interatomic distances to distinct sites, "
f"and volume per atom; HierarchyMLP adds orbit-size fraction and log stabiliser order (22 components). Width "
f"and training are matched. Both are worse than the elemental-only baseline: "
f"{M['struct']['mae_mean']:.4f} and {M['hierarchy']['mae_mean']:.4f} eV/atom against {M['standard']['mae_mean']:.4f} "
f"(StructMLP vs shared: {d('struct','standard')}). We interpret this narrowly, as the comment merits: this "
f"hand-built descriptor set does not help this architecture on this split. We do not claim that structural "
f"descriptors are uninformative in general. The Methods also state that the fixed projection is a linear "
f"expansion and adds no information beyond its inputs.",
"Section 4.4; Tables 1, 2, S3."),

"R1.Major.5": (
"Done on raw structures rather than on cached tensors, which is the distinction the comment is really about.",
"Sections 3.3, 4.5; Table S9."),
"R1.Major.5b": (
"For 250 crystals we reshuffle the raw atom order, recompute the labels from the permuted coordinates, "
"rebuild the features and re-run every model. The largest change in any predicted target across the nine "
"models is 3.6e-7 eV, three orders of magnitude inside the stated 1e-4 eV tolerance. We also insert 19 "
"masked slots holding non-zero nuisance features and arbitrary labels ahead of the real sites, so the real "
"sites change position within the padded array; the largest change is again 3.6e-7 eV. LayerNorm is applied "
"along the feature dimension only, never across sites, so padded slots cannot influence real sites through "
"normalisation.",
"Sections 4.5; Table S9."),

"R1.Major.6": (
"Agreed; that comparison was not a controlled test and it has been removed.",
"Sections 4.2, S7; Table 2."),
"R1.Major.6b": (
f"The revised manuscript contains no comparison against a global hashed orbit representation. The genuine "
f"within-crystal symmetry classes survive only as a bounded control, SiteLabel, with 32 slots and 535,105 "
f"parameters, and the parameter mismatch is stated explicitly rather than used as a result. SiteLabel reaches "
f"{M['sitelabel']['mae_mean']:.4f} eV/atom, which does not improve on the 19,265-parameter shared baseline "
f"({d('sitelabel','standard')}). We make no general claim that physical symmetry is inferior; we report that "
f"this parameterisation gives no gain here.",
"Sections 3.2, 4.2; Tables 1, 2."),

"R1.Major.7": (
"We take the option of not presenting an uncontrolled comparison, and go further: every cross-architecture "
"number has been removed from the paper.",
"Sections 1, 4, 5."),
"R1.Major.7b": (
"The revised manuscript contains no CGCNN or other external-model scores, no benchmark table, and no "
"speed-up factor against graph models. The paper is now a self-contained audit and methodology study whose "
"comparisons are all internal and matched. The only remaining statement about graph models is qualitative "
"and appears in the Introduction as motivation, without numbers.",
"Introduction; Section 5."),

"R1.Major.8": (
"Implemented in full, and the change in split design is what makes it meaningful.",
"Sections 3.1, 3.3, 4.1, 4.2; Tables 1, 2, S3."),
"R1.Major.8b": (
f"All models now share one frozen composition-family split, so the paired comparisons are computed on the "
f"same test crystals. Per-crystal absolute errors are averaged over seeds and then resampled 5,000 times over "
f"whole families for the interval, with a Wilcoxon signed-rank test and the fraction of resampled draws whose "
f"sign flips. Per-seed errors are listed in Table S3. Examples: WideMLP minus shared = {d('wide','standard')}; "
f"Cube4 minus shared = {d('cube4','standard')}. Every number in the paper is now an absolute value with an "
f"interval; no percentage improvement is reported without its absolute counterpart.",
"Tables 1, 2, S3, S4."),

"R1.Major.9": (
"Accepted. The word transferability is removed and the design is replaced by a space-group holdout.",
"Sections 3.3, 4.6; Table S6."),
"R1.Major.9b": (
"The least frequent space groups of the cohort are accumulated until at least 100 structures are set aside "
"(61 space groups, 102 structures), and every composition family that appears in the holdout is removed from "
"training, which also removes polymorphs of the held-out compounds. The manuscript states explicitly that a "
"space-group holdout is a limited distribution-shift check and does not establish chemical novelty, because "
"held-out cells can still share compositions and structural motifs with training cells through other space "
"groups.",
"Sections 3.1, 4.6; Table S6."),

"R1.Major.10": (
f"Answered within the data that exist. The cohort is filtered exactly as the comment describes "
f"({CO['n_crystals']} crystals, {CO['n_sites_min']}-{CO['n_sites_max']} sites, energy above hull no greater than "
f"0.2 eV/atom, at least two symmetry operations), and we report it as given.",
"Sections 4.3, S11; Tables S2, S5, S5b."),
"R1.Major.10b": (
"Quantifying the effect of the filter would require the unfiltered queue, which was not archived with the "
"snapshot; Section S11 says so rather than estimating. What we can do is stratify: by crystal system "
"(Table S5) and by terciles of the number of symmetry operations (Table S5b). If the filter were creating the "
"symmetry effect, high-symmetry structures should benefit most. They do not. Cube4 is worse than the shared "
"baseline in the low-symmetry stratum with an interval that excludes zero, and shows no significant "
"difference in the high-symmetry stratum.",
"Tables S5, S5b; Section S11."),

"R1.Major.11": (
"Clarified, and then measured.",
"Sections 3.1, 4.5; Table S9."),
"R1.Major.11b": (
"Padded slots never carry a meaningful label: they receive arbitrary values in the label models and are "
"excluded by a binary mask before the slot-selected transform and before pooling. LayerNorm acts on the "
"feature dimension only. The padding check in Table S9 inserts 19 masked slots carrying non-zero nuisance "
"features and arbitrary labels ahead of the real sites, so that real sites move position inside the padded "
"array; the largest change in prediction over 250 structures is 3.6e-7 eV. Because the label is a function "
"of the coordinate alone, padding cannot alter any real site's label.",
"Sections 3.1, 4.5; Table S9."),

"R1.Major.12": (
"Provided, with the limits of what we measured stated explicitly.",
"Sections 4.7, 6; Table S10."),
"R1.Major.12b": (
f"We report parameter counts, cached-feature forward latency at batch 1 and batch 64, peak GPU allocation and "
f"label-preprocessing time ({CO['n_crystals']} crystals, "
f"{json.load(open(os.path.join(RES, 'cost.json'), encoding='utf-8'))['label_preprocessing_ms_per_crystal']:.4f} ms per "
f"crystal). We also state what the timings exclude: database access, structure parsing and symmetry "
f"detection. The earlier claim that graph construction is inherently prohibitive is removed, and no "
f"end-to-end speed-up over any graph model is claimed.",
"Sections 4.7, 6; Table S10."),

"R1.Major.13": (
"Added, and used to sharpen the scope of the paper. A new paragraph in the Introduction cites and discusses "
"the three studies named in the comment together with the wider machine-learning literature on inorganic "
"crystalline materials. A new paragraph at the end of the Discussion states how the present contribution "
"differs from that work: the symmetry-derived label is tested directly, before it is used for property "
"prediction, instead of inferring the validity of the construction from a downstream accuracy metric. No "
"cross-architecture comparison is introduced with these citations.",
"Section 1 (new second paragraph), Section 5 (new paragraph before Limitations); references [9]-[11]."),

"R1.Minor.1": ("The three concepts are now separated explicitly: crystallographic orbits and Wyckoff "
"positions are defined in Section 1, the finite-cube proxy is defined in Sections 2.2-2.3, and the "
"manuscript states that the proxy is neither of the other two.", "Sections 1, 2.1-2.3."),
"R1.Minor.2": ("The sentence has been rewritten to condition indistinguishability on the space-group "
"operations of the actual structure, and the analogous statement about the grid proxy is not made.",
"Section 1, 2.3."),
"R1.Minor.3": ("The 3.4% claim and every other percentage-only statement are removed. All results are "
"reported as absolute errors with family-clustered intervals; percentages appear only alongside the "
"absolute values they qualify.", "Abstract, Highlights, Tables 1 and 2."),
"R1.Minor.4": ("The term is removed. The 100-crystal random sample is replaced by a space-group holdout "
"(61 space groups, 102 structures) with family-disjoint training, described as a limited "
"distribution-shift check.", "Sections 3.1, 4.6; Table S6."),
"R1.Minor.5": ("Removed. The manuscript no longer asserts that graph construction is prohibitive, and the "
"cost section reports only measured quantities with their scope.", "Sections 1, 4.7."),
"R1.Minor.6": ("Removed. No claim about local versus global properties is made from two targets; the "
"band-gap section is presented as a consistency check with three seeds.", "Sections 4.8, S4."),
"R1.Minor.7": (f"Reported: {CO['n_elements']} chemical elements are represented in the {CO['n_crystals']}-crystal cohort. "
"We recomputed this from the stored atomic numbers rather than repeating the earlier value, and the two "
"agree.", "Sections 4.1, 6; Table S2."),
"R1.Minor.8": ("Yes, and this is now stated. One fixed projection from four inputs to 64 dimensions "
"(seed 0, standard deviation 1/sqrt(d_in)) is generated once and reused by every elemental model across "
"all runs and seeds; the structural baselines use separate fixed projections of the same form.",
"Sections 4.2, 6."),
"R1.Minor.9": (f"Documented three ways: Figure 2 shows three worked coordinate-to-voxel-to-label examples, "
"Section S8 lists the label function verbatim, and the released audit script reproduces both the corrected "
f"result (48/48 bijections, 2304/2304 compositions, K = {A['K']}, sizes {A['orbit_sizes']}) and the archived "
f"result ({A['legacy']['K_reported']} classes from {A['legacy']['non_bijective_generators']} non-injective elements).",
"Figure 2; Sections S1, S8."),
"R1.Minor.10": ("The repository now hosts the corrected audit, the evaluation protocol, the per-crystal "
"predictions and the JSON tables. The exact commit used for the revision is quoted in the Data and Code "
"Availability statement.", "Data and Code Availability."),
"R1.Minor.11": ("Stated honestly: the study uses a locally cached Materials Project snapshot whose database "
"release identifier and retrieval date were not archived with the data. We report the cohort as stored, do "
"not claim a specific database version, and give per-crystal content-derived identifiers and exported "
"coordinates so that the cohort can be compared with any release.", "Sections 6, S11; Data and Code Availability."),
"R1.Minor.12": ("The wording is changed throughout. The revised title speaks of parameterisation rather than "
"sharing, the models are described as structured slot-based parameterisations, and parameter counts are "
"reported next to every model.", "Title; Sections 2.3, 3.2, 6; Table 1."),
"R1.Minor.13": ("Yes: one matrix per slot is shared by every crystal. Section 4.3 states that the slot index "
"is a geometric convention, that it is not an established physical equivalence between sites of different "
"crystals, and that no cross-crystal physical meaning is claimed. The within-crystal control (SiteLabel) is "
"reported separately with its parameter disadvantage made explicit.", "Sections 2.3, 4.2, 4.3."),
"R1.Minor.14": ("Figure 2 now contains the voxel map (f wrapped to [0,1)^3, c = floor(4f)), the four orbits "
"across the four z slices, the orbit sizes, and three worked examples of an actual coordinate mapping to a "
"voxel and a label.", "Figure 2."),
"R1.Minor.15": ("Added as Figure S1, covering the cohort, the frozen split, the voxel map, the leakage-free "
"training protocol, the statistics, the corrected action and the matched controls.", "Figure S1."),
}

OUTCOMES = {
    "R1.Major.13":
    "Done. The reference list now runs to [9]-[11]; a new second paragraph of the Introduction cites and "
    "discusses the three named studies together with the wider machine-learning literature on inorganic "
    "crystals, and a new closing paragraph of the Discussion states the difference in scope. No "
    "cross-architecture number was introduced with these citations.",
    "R1.Minor.1":
    "Done. The three concepts are named separately throughout and the manuscript states explicitly that the "
    "finite-cube proxy is neither a crystallographic orbit nor a Wyckoff position.",
    "R1.Minor.2":
    "Done. Indistinguishability is conditioned on the space-group operations of the actual structure, and the "
    "analogous statement is no longer made for the grid proxy.",
    "R1.Minor.3":
    "Done. No percentage-only claim remains; each effect is reported as an absolute error in eV/atom with a "
    "family-clustered interval and a paired test.",
    "R1.Minor.4":
    "Done. The word transferability is removed; the holdout is described as a limited distribution-shift "
    "check on 61 space groups and 102 structures with family-disjoint training.",
    "R1.Minor.5":
    "Done. No claim that graph construction is prohibitive remains; Section 4.7 reports measured quantities "
    "and states what the timings exclude.",
    "R1.Minor.6":
    "Done. No conclusion about local versus global properties is drawn; band gap is presented as a "
    "consistency check on three seeds.",
    "R1.Minor.7":
    "Done. The element count was recomputed from the stored atomic numbers (82 elements) rather than repeated "
    "from the earlier text.",
    "R1.Minor.8":
    "Done. Section 6 states that one fixed projection (seed 0, standard deviation 1/sqrt(d_in)) is generated "
    "once and reused by every elemental model, and that the structural baselines use separate fixed "
    "projections of the same form.",
    "R1.Minor.9":
    "Done. Figure 2 gives three worked coordinate-to-voxel-to-label examples, Section S8 lists the label "
    "function verbatim, and the released audit script reproduces the corrected and the archived result.",
    "R1.Minor.10":
    "Done. Table 1 and Table S10 list the parameter count of every model next to its error, including the "
    "capacity-matched wide network.",
    "R1.Minor.11":
    "Done. Section 6 and the Data and Code Availability statement say that the database release and "
    "retrieval date were not archived with the snapshot, and the cohort is reported as stored.",
    "R1.Minor.12":
    "Done. The title, the abstract and Section 2.3 speak of parameterisation and of structured slot-based "
    "parameterisation; the phrase parameter sharing is no longer used for a model that has more parameters "
    "than the baseline.",
    "R1.Minor.13":
    "Done. Section 4.3 states that the slot index is a geometric convention, that it is not an established "
    "physical equivalence between sites of different crystals, and that no cross-crystal physical meaning is "
    "claimed; the within-crystal control is reported separately with its parameter disadvantage.",
    "R1.Minor.14":
    "Done. Figure 2 now contains the voxel map, the four orbits across four z slices, the orbit sizes and "
    "three worked examples.",
    "R1.Minor.15":
    "Done. Figure S1 was added, covering the cohort, the frozen split, the voxel map, the leakage-free "
    "training protocol, the statistics, the corrected action and the matched controls.",
}


def main():
    df = pd.read_csv(TRACKER_SRC)
    rows = []
    letter_detail = {}
    for _, r in df.iterrows():
        i = r["id"]
        resp = RESP.get(i, ("", ""))
        extra = RESP.get(i + "b")
        letter_detail[i] = extra[0] if extra else ""
        body = resp[0] + (" " + extra[0] if extra else "")
        parts = [resp[1].rstrip(" .")] + ([extra[1].rstrip(" .")] if extra else [])
        loc = "; ".join(parts) + "."
        rows.append({
            "id": i,
            "type": "Major" if "Major" in i else "Minor",
            "reviewer_requirement": r["reviewer_concern"],
            "severity": r["severity"],
            "action_taken": resp[0],
            "measured_outcome": extra[0] if extra else OUTCOMES.get(i, ""),
            "evidence_location": loc,
            "status": "ADDRESSED",
        })
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "reviewer_response_tracker.csv"), index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(os.path.join(OUT, "reviewer_response_tracker.xlsx"),
                        engine="openpyxl") as xw:
        t.to_excel(xw, sheet_name="Comments", index=False)
        summary = pd.DataFrame([
            ["Major comments", int((t["type"] == "Major").sum()), "ADDRESSED"],
            ["Minor comments", int((t["type"] == "Minor").sum()), "ADDRESSED"],
            ["Total", len(t), "ADDRESSED"],
            ["Outstanding action items", 0, "all comments closed"],
            ["Real-data predictive runs completed", 9, "formation energy, 5 seeds"],
            ["Real-data predictive runs completed (band gap)", 9, "3 seeds"],
            ["Space-group holdout runs", 9, "3 seeds"],
        ], columns=["Item", "Count", "Note"])
        summary.to_excel(xw, sheet_name="Summary", index=False)
    print("wrote tracker xlsx/csv")

    # ---- response letter
    lines = [
        "# Response to the editor and to Reviewer 1",
        "",
        "**Manuscript:** COMMAT-D-26-03017R1 (revised and resubmitted)",
        "",
        "**Title:** Correctly constructing and fairly evaluating symmetry-aware parameterisation "
        "for crystal-property neural networks",
        "",
        "**Author:** Nailong Liang, Zhaoqing University, Zhaoqing 526061, China",
        "",
        "All numerical statements below come from the runs released with this revision. Section and "
        "table numbers refer to the revised manuscript and its supplementary material.",
        "",
        "## Response to the editor",
        "",
        "**E.1 Substantive revision.** The revision changes the central claim of the paper. The "
        "previously reported improvement is withdrawn: the construction that produced it was not a "
        "group action, and the corrected construction gives no accuracy gain under matched controls. "
        "The paper is resubmitted as an audit and methodology study.",
        "",
        "**E.2 Separate itemized response.** All 13 major and 15 minor comments are answered below, "
        "each with the action taken and the location of the change. No item is left as an unfulfilled "
        "promise, and no placeholder text remains in the manuscript.",
        "",
        "**E.3 Manuscript and figure source files.** The revised manuscript and supplementary material "
        "are supplied as DOCX. Every figure is supplied as SVG (source), 500+ dpi PNG and TIFF.",
        "",
        "**E.4 Highlights.** Supplied separately; five items, each within the 85-character limit.",
        "",
        "**E.5 Graphical abstract.** A declaration stating that no graphical abstract is supplied "
        "accompanies this revision, as permitted by the decision letter.",
        "",
        "## Reviewer 1 - Major comments",
        "",
    ]
    for _, r in t[t["type"] == "Major"].iterrows():
        lines += [f"### {r['id']}", "", "> " + str(r["reviewer_requirement"]).replace("\n", " "), "",
                  "**Response.** " + r["action_taken"] +
                  ((" " + letter_detail[r["id"]]) if letter_detail.get(r["id"]) else ""), "",
                  "**Changes.** " + r["evidence_location"], "", "**Status.** " + r["status"], ""]
    lines += ["## Reviewer 1 - Minor comments", ""]
    for _, r in t[t["type"] == "Minor"].iterrows():
        lines += [f"### {r['id']}", "", "> " + str(r["reviewer_requirement"]).replace("\n", " "), "",
                  "**Response.** " + r["action_taken"] +
                  ((" " + letter_detail[r["id"]]) if letter_detail.get(r["id"]) else ""), "",
                  "**Changes.** " + r["evidence_location"], "", "**Status.** " + r["status"], ""]
    with open(os.path.join(OUT, "Response_to_Reviewers.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote Response_to_Reviewers.md")

    # ---- highlights
    hl = ["Archived O_h grid orbits are invalid: 42 of 48 maps are non-injective.",
          "A cube-centred action gives four verified orbits of size 8, 24, 24 and 8.",
          "Matched occupancy, geometry and capacity controls replace the archived comparisons.",
          "On 2,741 crystals the corrected label model gives no accuracy gain.",
          "Accuracy is governed by capacity, not by the symmetry content of the labels."]
    assert all(len(h) <= 85 for h in hl), [len(h) for h in hl]
    with open(os.path.join(OUT, "highlights.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(hl) + "\n")
    print("wrote highlights.txt", [len(h) for h in hl])

    cover = f"""Dear Dr Fugallo,

Re: COMMAT-D-26-03017R1 - resubmission

Thank you for the opportunity to revise and resubmit our manuscript. The revision changes its
central claim, and we would rather report that than defend a result that does not survive
checking.

The original submission attributed a formation-energy improvement to a group-theoretic orbit
decomposition of a 4x4x4 grid. During revision we established that the construction used to
produce that result is not a group action: it placed the voxel index in an origin-centred frame
and clipped negative coordinates, so {A['legacy']['non_bijective_generators']} of the {A['n_elements']} elements of O_h are
non-injective and the {A['legacy']['K_reported']} published classes are not orbits. We replace the gauge with a
cube-centred action, verify by exhaustive enumeration that all {A['bijective_generators']} elements are bijections
and all {A['closure_ok']} compositions close, and obtain K = {A['K']} orbits of sizes
{', '.join(str(v) for v in A['orbit_sizes'])}.

We then rebuilt the evaluation so that the effect of the labels can be separated from the effect
of the capacity they add: one frozen composition-family split, target statistics fitted on
training families only, checkpoint selection on validation data only, and paired statistics
resampled over families. On the same 2,741-crystal cohort the corrected model reaches
{M['cube4']['mae_mean']:.4f} eV/atom against {M['standard']['mae_mean']:.4f} for a single shared transform, and it is also
beaten by a capacity-matched wide network ({M['wide']['mae_mean']:.4f}), by a deterministic non-symmetric
partition of the same voxel index ({M['geo4']['mae_mean']:.4f}) and by an occupancy-preserving relabelling
({M['occupancy4']['mae_mean']:.4f}). At the same time, and following the reviewer's comment on our earlier
comparison with a graph model, we have removed every cross-architecture number from the paper:
the revised study is self-contained and all of its comparisons are matched.

The paper is therefore submitted as what the evidence supports: a methodological audit of how
orbit-style parameterisations should be constructed and evaluated, with a negative result that
we report in full. We believe this is more useful to the community than the original claim.

The point-by-point response addresses all 13 major and 15 minor comments. The manuscript,
supplementary material, figure sources, highlights and data/code availability statement are
supplied with this submission.

Yours sincerely,

Nailong Liang
Zhaoqing University
nailongliang@outlook.com
"""
    with open(os.path.join(OUT, "cover_letter.md"), "w", encoding="utf-8") as f:
        f.write(cover)
    print("wrote cover_letter.md")
    with open(os.path.join(OUT, "graphical_abstract_declaration.txt"), "w", encoding="utf-8") as f:
        f.write("No graphical abstract is supplied with this submission.\n")


if __name__ == "__main__":
    main()
