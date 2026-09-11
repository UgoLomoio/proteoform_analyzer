"""Proteoform Analyzer: mutation & PTM effect analysis for oligomeric proteins.

A pipeline to analyze the effect of single-point mutations, post-translational
modifications (PTMs), and their pairwise combinations (proteoforms) on proteins
of arbitrary oligomeric state (monomers, dimers, tetramers, hexamers, etc.).

Based on Lomoio et al., npj Systems Biology and Applications (2025),
doi:10.1038/s41540-025-00582-2, generalized with PTM simulation via ptmpsi,
binding-site prediction (P2Rank / alpha-sphere), and proteoform combinatorics.

v3.0.0 replaces the local_pdb/AF3 structure sources with Boltz-2 folding
(locally-runnable via HPC by default, or a local binary), adds a Boltz-2
co-folding docking engine, and adds RFAntibody-based antibody/nanobody design
with a pluggable epitope-predictor interface.

v3.1.0 fixes several local-run errors (graceful HPC/GPU skips when 'biomni' is
absent, a restored chain-map helper, a DiffSBDD dependency pre-flight, a pocket
detector cascade that never emits fake zeros, and PCN community-detection
guards), wires *real* MHC epitope predictors (MHCflurry for MHC-I, IEDB
NetMHCIIpan 4.3 BA for MHC-II), and adds interactive Plotly + static SVG/PNG
visualizations for every result subsection. Ships README.md, requirements.txt,
and pyproject.toml with optional-dependency extras.

v3.1.1 focuses on honest local-mode results and GUI clarity: the composite
impact score now drops degenerate (all-equal) components and renormalizes the
weights over the informative ones, so local rankings reflect the terms that
actually differ (locally, the ESM2 sequence-embedding distance) instead of
being diluted by constants; the GUI drops all-zero proteoform / pocket / drift rows and shows an
honest note instead of a misleading all-zero table or plot; the confusing
pocket "volume is not a real measurement" wording is replaced with a clear
explanation; empty docking panels are hidden instead of showing "No data yet";
the PCN 3D viewers render a cartoon baseline (not licorice); the PCN
community-detection 0-d array crash (Louvain/Leiden/Infomap) is fixed in the
vendored code; the Molecular Dynamics tab is simplified to an RMSD/RMSF overlay
(legend-driven trace selection); the DiffSBDD pre-flight now detects
installed-but-broken torch-scatter (ABI/version mismatch) and skips cleanly with
an actionable message; and the biomni HPC "version incompatible" message is
softened to explain that GPU/HPC steps are simply skipped in local mode. See
CHANGES.md for the full list.

v3.2.0 makes Boltz-2 folding, Boltz-2 docking, and BoltzGen ligand design
runnable without the Biomni platform by routing all three through a single
backend resolver (``core/steps/_boltz_backend.resolve_backend``): (1) the
official hosted Boltz API (set ``BOLTZ_API_KEY`` or ``Boltz2Config.api_key``;
``pip install boltz_api``), (2) a local ``boltz`` / ``boltzgen`` install, then
(3) the Biomni HPC backend, and finally (4) an honest ptm-psi graft fallback for
folding only (backbone-identical, TM-score == 1.0), gated on
``allow_graft_fallback`` and only chosen when it can actually run. The GUI
surfaces how structures were produced (a prominent warning when the graft
fallback is used) and, for ``binding_site_method='reference'``, warns that
pocket prediction and pocket drift cannot change (the reference site is a fixed
box with no computed volume, and its center only moves if the backbone differs
from WT). PCN centralities can now be shown as a viridis colormap on the
structure (with a colorbar) alongside the diverging Δ-vs-WT view, plus a
top-10 bar plot of the residues with the largest |Δ centrality| (name+id,
signed +/-); PCN communities are coloured per community on the selected
structure; and the 3D structure viewer colours cartoons by chain ID with a
colorblind-safe legend. See CHANGES.md for the full list.

Local-mode caveat: without a GPU / a local boltz install / a Boltz API key,
mutant and proteoform structures are produced by side-chain grafting onto the
wild-type backbone (no folding or relaxation). The backbone is therefore
identical to WT, so TM-score == 1.0 is expected and the structural / binding /
dynamics / network impact components are 0; the composite ranking then rests on
the ESM2 sequence-embedding distance. Real structural divergence requires
Boltz-2 folding (Boltz API or a local boltz binary on a GPU).

v3.3.1 removes RFAntibody from the ligand_design engine options (the standalone
opt-in ``antibody`` step is unchanged), adds a graceful Boltz-API auth-failure
fallback (on a 401 / invalid-key error the pipeline warns once and falls back to
a local Boltz install or the PTM-Psi graft, never surfacing the raw 401 text),
and fixes the proteoform step to map mutation/PTM sites to ordinal-based
selectors via ``map_site_to_structure`` (the same fix applied to the PTM step in
v3.3.0), so author-numbered PDBs no longer cause "Residue could not be found"
errors in proteoform generation.

v3.4.0 removes the Biomni HPC backend entirely. Every step that previously
submitted GPU jobs through the Biomni HPC API now runs locally or falls back
cleanly, with no dependency on the Biomni platform:
  - The ``core/steps/_hpc.py`` shim and all ``hpc_run_tool`` / ``biomni.tool``
    calls are deleted. Backend precedence is now: hosted Boltz API -> local
    binary -> (fold only) PTM-Psi graft -> clean skip.
  - **Boltz-2 folding & docking** run via the Boltz API or a local ``boltz``
    binary (``--boltz2-local-binary``); folding keeps the backbone-identical
    graft fallback, docking skips cleanly when no backend resolves.
  - **Antibody design (RFAntibody)** runs the 3-stage pipeline
    (RFdiffusion_Ab -> ProteinMPNN -> RF2) *synchronously* on a local RFAntibody
    checkout (``--antibody-rfantibody-dir`` / ``config.antibody.local_rfantibody_dir``,
    weights via ``--antibody-weights-dir``); without it the step skips cleanly
    with install instructions. The old two-pass async HPC orchestration is gone.
  - **ddG (ThermoMPNN)** runs a local ThermoMPNN checkout when configured
    (``--thermompnn-dir`` + ``--thermompnn-checkpoint``); otherwise it falls back
    to the existing ESM2 zero-shot CPU scorer (Tier 2), so ddG always produces a
    result on CPU.
  - **BoltzGen** binder design runs via the Boltz API or a local ``boltzgen``
    install. See CHANGES.md for the full list.

Entry points:
    python -m proteoform_analyzer.cli        # command-line interface
    python -m proteoform_analyzer.gui        # Gradio web GUI
"""
from .core.config import (
    AnalysisConfig, EngineChoice, ProteoformMode, BindingSiteMethod, HotspotSource,
    PTMConfig, MDConfig, Boltz2Config, AntibodyConfig,
    hemoglobin_fast_config, ttr_fast_config, p53_fast_config,
)

__version__ = "3.7.0"
__all__ = [
    "AnalysisConfig", "EngineChoice", "ProteoformMode", "BindingSiteMethod",
    "HotspotSource", "PTMConfig", "MDConfig", "Boltz2Config", "AntibodyConfig",
    "hemoglobin_fast_config", "ttr_fast_config", "p53_fast_config",
]
