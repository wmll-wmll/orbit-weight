# Corrected crystal-model evidence release 20260924

This release accompanies COMMAT-D-26-03017R1. Reported results are immutable under `results/`; reproduction outputs go to `work_results/`. Python 3.12.14 and CPU PyTorch 2.14.0 were used, with four threads. Prepared inputs allow reproduction without a database key. Install requirements into an isolated environment.

## Verification without training
`python code/audit_grid_action.py`
`python code/final_statistics.py`

## Optional full retraining
`python code/final_train_evidence.py main`
`python code/final_train_evidence.py gap`
`python code/final_train_evidence.py transfer`
`python code/final_protocol_diagnostics.py`

Training reuses the frozen partitions and writes new runs separately. Do not rerun `design` when reproducing the frozen analysis. `final_evidence_audit.py` rebuilds symmetry/structural features in work_results; the earlier hierarchy cache is retained only to quantify the shortest-image correction. `final_robustness.py` uses released checkpoints and writes a separate recheck. Hardware/version changes can affect floating-point results, symmetry tolerances and timing.

The 69 main/secondary runs contain checkpoints and full traces. Nine diagnostic runs contain validation traces and test predictions; diagnostic checkpoints were not retained. The diagnostic is not an exact reconstruction of the archived publication. Source provenance and a credential-redacted historical source are supplied solely for audit.

## Data provenance and attribution
The cohort is an archived Materials Project-derived export. Cite A. Jain et al., APL Materials 1, 011002 (2013), doi:10.1063/1.4812323. Materials Project data attribution/license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ . Snapshot redistribution guidance: https://matsci.org/t/is-it-ok-to-post-snapshots-of-materials-project-structures-on-e-g-figshare-i-e-for-manuscript-reproducibility/45437 . Content-derived mpx IDs are not database accessions. The original release identifier, retrieval date and pre-filter queue are unavailable. The data serve reproduction of this study; use the current Materials Project source for new research. No public repository DOI is claimed.
