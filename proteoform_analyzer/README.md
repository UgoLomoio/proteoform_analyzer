# Proteoform Analyzer

**Mutation & PTM effect analysis for oligomeric proteins.**

A pipeline to analyze how single-point mutations, post-translational
modifications (PTMs), and their pairwise combinations (*proteoforms*) affect
proteins of arbitrary oligomeric state (monomers, dimers, tetramers,
hexamers, …).

Based on Lomoio *et al.*, *npj Systems Biology and Applications* (2025),
[doi:10.1038/s41540-025-00582-2](https://doi.org/10.1038/s41540-025-00582-2),
generalized with PTM simulation (ptmpsi), Boltz-2 structure folding & docking,
binding-site prediction (P2Rank / fpocket / alpha-sphere), protein contact
networks, ESM2 embeddings, and antibody/nanobody design (RFAntibody, via the
opt-in `antibody` step) with a pluggable epitope-predictor interface (B-cell
heuristic + **real MHC-I/II**).

---

## Contents
- [What it computes](#what-it-computes)
- [Installation](#installation)
- [Local vs. GPU execution](#local-vs-gpu-execution)
- [Quickstart](#quickstart)
- [Visualizations](#visualizations)
- [Troubleshooting](#troubleshooting)
- [Known caveats](#known-caveats)

---

## What it computes

The default pipeline runs these steps (13 built-in; `impact_score` last,
`antibody` is opt-in):

| Step | Needs GPU? | Purpose |
|------|:---:|---------|
| `sequence` | no | Fetch/prepare reference sequences |
| `structure` | *optional* | Fold structures (Boltz-2: hosted API **or** local binary; graft fallback) |
| `ptm` | no | Apply PTMs via ptmpsi |
| `proteoform` | no | Enumerate mutation × PTM combinations |
| `tmscore` | no | TM-score vs wild-type (vendored TMalign) |
| `pcn` | no | Protein contact networks (centrality + communities) |
| `md` | no* | Molecular dynamics (OpenMM); RMSD/RMSF/energy |
| `esm2` | no* | ESM2 embeddings + UMAP variant map |
| `pocket` | no | Binding-pocket prediction + drift vs WT |
| `docking` | *optional* | AutoDock Vina (local) **or** Boltz-2 co-folding (API/local) |
| `ligand_design` | *optional* | De novo ligands/binders (DiffSBDD / BoltzGen) |
| `ddg` | *optional* | ΔΔG stability (mutations: local ThermoMPNN → ESM2 zero-shot CPU fallback; PTMs: FoldX mimetic tier, opt-in) |
| `impact_score` | no | Composite ranking across all of the above |
| `antibody` | yes (local) | RFAntibody design + epitope prediction (opt-in; local RFAntibody install) |

\* `md` and `esm2` run locally on CPU but are much faster on GPU and need their
optional extras (`[md]`, `[esm]`).

---

## Installation

Requires **Python ≥ 3.9**. We recommend a fresh virtual environment (venv or
conda).

### 1. Core install

```bash
cd proteoform_analyzer
conda create -n proteoform_analyzer python=3.12.0
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
conda install -c conda-forge dgl
pip install -r requirements.txt      # core deps only
# — or, to install as a package with the CLI entry point —
pip install .
```

The core install runs: `sequence`, `ptm`, `proteoform`, `tmscore`, `pcn`,
`pocket`, `impact_score`, plus the GUI/CLI. That is enough to reproduce the
non-GPU analyses end-to-end.

### 2. Optional back-ends (extras)

Install only what you need. Extras are additive:

```bash
pip install ".[md]"       # molecular dynamics (OpenMM + mdtraj)
pip install ".[esm]"      # ESM2 embeddings / UMAP + sequence ddG
pip install ".[dock]"     # AutoDock Vina receptor/ligand prep (meeko)
pip install ".[mhc]"      # MHC-I epitope prediction (MHCflurry)
pip install ".[pcn]"      # infomap community algorithm (wurlitzer)
pip install ".[design]"   # de novo ligand design (DiffSBDD; GPU recommended)
pip install ".[boltzapi]" # hosted Boltz-2 API client (api.boltz.bio)
pip install ".[viz]"      # Plotly-native static image export (kaleido)

# everything pip-installable at once (no system binaries):
pip install ".[all]"
```

**Boltz-2 folding / docking / binder design** resolve through a shared backend
chain, tried in this order (see [Boltz backends](#boltz-2-backends-api--local--graft)):

1. **Hosted API** (`api.boltz.bio`) — `pip install ".[boltzapi]"` and set a key:
   `export BOLTZ_API_KEY=sk-...` (or `Boltz2Config.api_key`).
2. **Local install** — `pip install boltz` (folding/docking) and/or `boltzgen`
   (design); set `Boltz2Config.prefer_local=True` to try local first.
3. **PTM-Psi graft** (folding only) — no install needed beyond the bundled
   ptm-psi; used when neither the API nor a local install is available.

If none of the folding tiers is available, folding uses the **ptm-psi side-chain
graft** (backbone-identical → TM-score 1.0; see the important caveat below).
Binder design has no graft tier and skips cleanly when no API/local engine is
present.

### 3. System binaries (not pip-installable)

Some back-ends need an external executable on your `PATH`:

| Tool | Needed for | How to get it |
|------|-----------|---------------|
| **Java (JRE ≥ 11)** | P2Rank pocket detection | `conda install -c conda-forge openjdk` or your OS package manager |
| **fpocket** | pocket detection fallback | `conda install -c bioconda fpocket` (Linux/macOS) |
| **AutoDock Vina** | `docking` (Vina engine) | `conda install -c bioconda autodock-vina` or [vina releases](https://github.com/ccsb-scripps/AutoDock-Vina/releases) |
| **Boltz** binary | local Boltz-2 folding/docking | `pip install boltz` then set `Boltz2Config.local_binary` / `prefer_local=True` (or use the hosted API via `[boltzapi]` + `BOLTZ_API_KEY`) |
| **BoltzGen** | local ligand/binder design | `pip install boltzgen` (or use the hosted Boltz API) |
| **RFAntibody** | opt-in `antibody` design (GPU) | clone [RFantibody](https://github.com/RosettaCommons/RFantibody), fetch weights, then set `--antibody-rfantibody-dir` / `config.antibody.local_rfantibody_dir` (`--antibody-weights-dir` for the checkpoint) |
| **ThermoMPNN** | `ddg` Tier-1 (GPU; optional) | clone/install [ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN), then set `--thermompnn-dir` + `--thermompnn-checkpoint` (else ddG falls back to ESM2 zero-shot on CPU) |
| **FoldX** | `ddg` Tier-3: PTM ΔΔG via mimetic substitutions (optional) | obtain a [FoldX](https://foldxsuite.crg.eu/) license/binary (academic; not bundled), then set `--foldx-binary` (GUI: "FoldX binary" field) |
| **MHCflurry models** | MHC-I (after `[mhc]`) | `mhcflurry-downloads fetch models_class1_presentation` |

> **After installing `[mhc]`**, download the models once:
> ```bash
> mhcflurry-downloads fetch models_class1_presentation
> ```
> MHC-**II** (NetMHCIIpan) uses the IEDB web API and needs no local model —
> only outbound HTTPS (see the [caveat](#known-caveats)).

---

## Local vs. GPU execution

The tool is designed to **degrade gracefully** when GPU acceleration is
unavailable.

- **Steps that never need a GPU** (`sequence`, `ptm`, `proteoform`, `pcn`,
  `tmscore`, `esm2`, `pocket`) always run locally.
- **Boltz-2 folding / docking and binder design** are resolved through a single
  backend chain (see [Boltz backends](#boltz-2-backends-api--local--graft)):
  **hosted API → local install → graft** (folding only). The hosted-API path
  runs the job on Boltz's servers; the local path uses your own GPU/CPU.
- **Other GPU-accelerated steps** (`ddg`, and the opt-in `antibody` step)
  require their matching local installs. When the requirement is absent, they
  **fall back or skip cleanly with an actionable message** instead of crashing
  the whole pipeline:
  - `ddg` runs a **local ThermoMPNN** checkout (Tier 1) when
    `--thermompnn-dir` + `--thermompnn-checkpoint` are set; otherwise it falls
    back to the **ESM2 zero-shot CPU scorer** (Tier 2), so it always produces a
    result. When `--foldx-binary` is set and `run_ptm` is on, a **FoldX
    mimetic tier** (Tier 3) additionally scores PTM ΔΔG — phosphorylation
    SER/THR→GLU, acetylation LYS→GLN — on the WT background and on every
    mutation×PTM proteoform; PTMs without an accepted mimetic are skipped
    with a warning. Without a FoldX binary the tier is skipped cleanly.
  - `antibody` runs a **local RFAntibody** install synchronously
    (RFdiffusion_Ab → ProteinMPNN → RF2) when `--antibody-rfantibody-dir` is set
    and weights resolve; otherwise it **skips cleanly** with install
    instructions.

If you run the `antibody` step without a local
RFAntibody install configured it prints, e.g.:

> *“No local RFAntibody install configured. Set
> config.antibody.local_rfantibody_dir to a checkout containing
> scripts/{rfdiffusion_inference,proteinmpnn_interface_design,rf2_predict}.py and
> a weights dir with 'RFdiffusion_Ab.pt' … GPU strongly recommended.”*

…and the pipeline continues with the remaining steps.

### What mutant structures are used in local mode? (important)

When Boltz-2 folding is **not** available (no API key, no local `boltz` binary,
no GPU), the tool still needs a 3-D structure for every mutant and proteoform. It
builds them by **grafting the mutated side chains / PTMs onto the wild-type
backbone** (`ptmpsi.mutate()` and `.modify()`) — there is **no folding and no
relaxation**, so the backbone is *identical* to the wild-type.

This has direct, expected consequences for the results (they are not bugs):

| Result | Local graft (backbone-identical) | Needs Boltz-2 folding (API/GPU) |
|---|---|---|
| **TM-score vs WT** | **1.0** for every mutant (identical backbone) | < 1.0 reflects real fold change |
| **Impact component: structural / binding / dynamics / network** | 0 (nothing moved) | non-zero |
| **Impact component: ESM2 sequence distance** | **varies per proteoform** | also varies |
| **Pocket volume / drift** | no change | real change |
| **Boltz-2 docking confidence** | unavailable | pTM / ipTM values |

The composite impact score has exactly **five components**: structural
(`|1-TM|`), binding (docking-affinity shift), dynamics (RMSD), network (PCN
community change), and sequence (ESM2 embedding distance). Because of the
backbone-identical structures, in local mode the first four are all 0 and the
**composite is driven solely by the ESM2 sequence-embedding distance** — the
only component that differs per proteoform. The score automatically **drops the
degenerate (all-zero) components and renormalizes the weights** over the
informative ones (see `impact_score_components.csv`), and the GUI **hides
all-zero rows / panels** and shows a short note explaining why. To get real
structural divergence (TM < 1, pocket drift, docking confidence), run Boltz-2
folding via the hosted API or a local GPU.

> **Note:** ΔΔG stability (the separate `ddg` step) is a genuine per-proteoform
> signal, but it is **not** one of the impact-score components — it is reported
> in its own results section, not folded into the composite ranking.

---

## Boltz-2 backends (API / local / graft)

Boltz-2 folding, Boltz-2 docking, and protein binder design (BoltzGen) route
through one shared resolver
(`core/steps/_boltz_backend.py::resolve_backend`) that picks the first available
backend from a **three-tier chain**, so the same run works on a laptop or with a
cloud key:

| Tier | Backend | When it is used | Requirement |
|------|---------|-----------------|-------------|
| 1 | **Boltz API** (official `api.boltz.bio`) | an API key is available | `pip install ".[boltzapi]"` + `api_key` / `BOLTZ_API_KEY` |
| 2 | **Local install** | `boltz` (fold/dock) and, for design, `boltzgen` is importable / on `PATH` | `pip install boltz boltzgen` (GPU recommended) |
| 3 | **PTM-Psi graft** (fold only) | nothing above is available and grafting is allowed | ptm-psi (bundled) + a `local_pdb_id` seed |

The hosted-API paths call the official `boltz_api` SDK:
`client.predictions.structure_and_binding.run(...)` for folding/docking (docking
adds a `ligand_protein_binding` block when a ligand is present) and the
asynchronous **`client.protein.design`** resource
(`start` → `retrieve` → `list_results`) for binder design.

Configuration lives on `config.boltz2` (a `Boltz2Config`):

| Field | Default | Meaning |
|-------|---------|---------|
| `api_key` | `None` | Boltz API key; falls back to the `BOLTZ_API_KEY` environment variable (`resolved_api_key()`). |
| `api_base_url` | `https://api.boltz.bio` | Official Boltz endpoint. |
| `api_model` | `boltz-2.1` | Model requested from the API. |
| `prefer_local` | `False` | If `True`, try a local install **before** the API. |
| `allow_graft_fallback` | `True` | Allow the ptm-psi graft fold-only fallback as a last resort. |

> Removed in v3.4.0: `use_hpc` and `hpc_tool_id`. There is no HPC tier.

The same options are exposed in the GUI **Setup → Engine choices** panel (API-key
field, "prefer local", "allow graft fallback").

### Binder design engines (`ligand_design_engine`)

The design step supports two engines: **`diffsbdd`** (small-molecule design) and
**`boltzgen`** (protein binder design). When binder design resolves to the local
tier, the step runs the requested engine if it is installed. Binder design has
**no** graft fallback (there is nothing to graft), so it skips cleanly with an
actionable message when no API key and no local engine are present.

> **Antibody/nanobody design.** RFAntibody-based antibody/nanobody design is
> available as the separate, opt-in **`antibody`** step (**local RFAntibody
> install**), not as a `ligand_design_engine` option. See the antibody-design
> note below.

> **Docking and design scope.** Fold, dock, *and* design all use this resolver.
> Docking additionally requests Boltz-2 binding metrics from the API when a
> ligand is present. The graft tier is **fold-only**: it can build mutant
> backbones but cannot produce docking confidence or designed binders, so
> docking/design cleanly report "unavailable" on that tier rather than faking a
> result.

> **Graft fallback caveat (important).** The tier-3 graft **does not fold or
> relax** — it transplants mutated side chains / PTMs onto the wild-type
> backbone. Every mutant therefore shares the WT backbone, so **TM-score vs WT is
> 1.0 for every proteoform** and the structural / binding / dynamics / network
> impact components are all 0. In this mode the impact ranking is driven **solely
> by the ESM2 sequence-embedding distance** component. (ΔΔG is reported as a
> separate per-proteoform signal and is **not** part of the composite impact
> score in any mode.) A run that used the graft tier writes
> `structure_provenance.json` (method = `graft`, `backbone_identical = true`) and
> the GUI shows a prominent red banner so this is never silently assumed. For
> real fold divergence (TM < 1, pocket drift, docking confidence), use the API or
> a local GPU Boltz install.

### Residue numbering (graft & PTM sites)

Mutation and PTM sites are given in **UniProt sequence numbering**, but reference
PDBs (`local_pdb_id`) use **author/PDB numbering**, which is usually offset.
Since v3.3.0 the graft and PTM steps map each site to the correct structural
residue with a **name-verified mapper** (`core/steps/_resnum.py`): it matches by
author number, then ordinal, then a single corroborated per-chain offset, and
**verifies the amino-acid identity every time**. A site that cannot be located
with a matching residue is **skipped with a warning** rather than being applied
to the wrong position. This fixes earlier failures such as
*"Residue 'LEU87' could not be found"* for `L87R` on hemoglobin and
`glutathionylation` on `CYS10`.

### Mature chain (proteolytic processing)

Many proteins are synthesised as precursors and **proteolytically matured**
before assembling: TTR loses its 20-aa N-terminal signal peptide (the tetramer
is formed by four mature monomers, residues 21–147); hemoglobin subunits lose
the initiator Met (HBA 2–142, HBB 2–147). Since v3.6.0 the pipeline models
this explicitly:

- The **mature region** of each subunit is resolved as `{uniprot_id: [start,
  end]}` in UniProt (precursor) numbering, 1-based inclusive. Manual entries
  (`config.mature_regions`, the GUI textbox, or `--mature-regions`) win;
  otherwise the sequence step **auto-detects** the region from UniProt
  features — a single CHAIN/PEPTIDE feature defines it directly, otherwise
  terminal SIGNAL/PROPEP/TRANSIT peptides are trimmed (multi-chain proteins:
  terminal trimming only, with a warning). Disable with
  `mature_auto_detect=False` / `--no-mature-auto`.
- The **mature sequence** is folded (Boltz-2), embedded (ESM2), scored (ESM2
  zero-shot ddG) and co-folded with ligands (Boltz-2 docking). Steps that work
  on reference PDBs (graft fallback, PTM, proteoform) are unchanged — those
  structures already contain the mature chain.
- **Numbering stays UniProt**: input mutations/PTMs use precursor numbering
  (e.g. TTR `V50M`); positions are remapped to mature numbering internally
  (V50M → mature position 30) and results are reported in UniProt labels.
- Mutations/PTMs in **cleaved regions** are kept in the tables (flagged) but
  excluded from structural steps with explicit warnings; see
  `mature_chain_report.csv` in the results dir for the per-subunit region,
  its source, and the excluded mutations.
- The built-in presets carry the right regions: TTR `{"P02766": [21, 147]}`,
  hemoglobin `{"P69905": [2, 142], "P68871": [2, 147]}`, p53 full-length.

```bash
# manual override (also works on top of a preset)
python -m proteoform_analyzer.cli run --preset "TTR — Fast" \
    --mature-regions "P02766:21-147" --work-dir ./out

# disable auto-detection (full-length precursor sequences)
python -m proteoform_analyzer.cli run --uniprot P02766 --no-mature-auto
```

---

## Quickstart

### Command line

```bash
# list all pipeline steps
python -m proteoform_analyzer.cli list-steps

# run a built-in preset (exact preset names carry an em-dash — quote them).
# Presets: "Hemoglobin | TTR | p53" × "Fast | Standard | Production".
python -m proteoform_analyzer.cli run --preset "Hemoglobin — Fast" --work-dir ./out

# equivalent shorthand via the legacy protein selector:
python -m proteoform_analyzer.cli run --protein hemoglobin --fast --work-dir ./out

# run only selected steps (skip the GPU-heavy ones)
python -m proteoform_analyzer.cli run --preset "p53 — Fast" \
    --steps sequence,ptm,proteoform,pcn,pocket --work-dir ./out
```

Results are written under `--work-dir` (default: `./results`). If installed as
a package, the console script `proteoform-analyzer` is equivalent to
`python -m proteoform_analyzer.cli`.

### Web GUI

```bash
python -m proteoform_analyzer.gui
```

Opens a three-tab Gradio app (**Setup → Run → Results**). Configure a run in
**Setup**, launch it in **Run**, then click **Refresh all results** in
**Results** to populate the tables and interactive plots.

### Python API

```python
from proteoform_analyzer import hemoglobin_fast_config
from proteoform_analyzer.core.pipeline import run_analysis

config = hemoglobin_fast_config()
config.work_dir = "./out"          # results go to ./out/<config.name>/
results = run_analysis(config)     # returns a list[StepResult]
for r in results:
    print(r.step, r.status, r.message)
```

---

## Visualizations

Every data-driven figure is produced in **two forms**: an interactive Plotly
`.html` (embedded in the GUI Results tab) **and** static `.svg` + `.png` saved
alongside it (under `<output_dir>/_plots/` and each step's own folder).

| Result subsection | Interactive plot |
|-------------------|------------------|
| Molecular Dynamics | RMSD **and** RMSF overlays (select trajectories + metric) |
| 3D Structure Viewer | molecule-type-aware 3Dmol.js: protein cartoon **coloured by chain ID** (Okabe–Ito colour-blind-safe palette + legend), ligands as licorice, antibody complexes coloured by chain with antigen-interface sticks |
| PCN centralities | residues coloured on the 3D structure by centrality using a **viridis colormap with a colorbar** (raw view), plus a **signed diverging** delta view (mutant − WT) |
| PCN top-Δ residues | horizontal bar plot of the **top-10 residues by \|Δ centrality\|**, labelled by residue name + id (+ chain), **red = increase / blue = decrease** vs WT with a signed legend |
| PCN communities | each structure's detected communities coloured directly on the 3D structure (per-structure view, not a WT−mutant difference) |
| Docking | (A) Vina affinity bar chart; (B) Boltz-2 pTM vs ipTM scatter |
| Pocket Prediction | pocket-volume bars (failed detectors greyed out — never faked as zero) |
| Pocket Drift | volume-change vs pocket-centre-displacement scatter |
| Impact / TM-score | composite-ranking and TM-score summary bars |
| Designed Ligands (DiffSBDD) | sortable per-ligand table: QED, SA, logP, MW, HBD/HBA, rotatable bonds, TPSA, Lipinski violations (sorted by QED) |

**PCN centrality & community views.** In the Results tab, the PCN panel lets you
pick a structure, a centrality measure, and a community algorithm. The centrality
view colours residues on the 3D structure with a **viridis colormap and a
colorbar** (a "raw" absolute view), and a toggle switches to a **diverging
signed** view of the change vs WT. Alongside it, a **top-10 \|Δ centrality\|**
bar plot names the most-shifted residues (name + id + chain) and marks each as a
positive (red) or negative (blue) change; residues whose centrality does not
change are excluded so the plot never shows empty bars. Communities are rendered
per structure (each community a distinct colour on the cartoon), avoiding the
ambiguous "difference between two community partitions" view.

> Static export uses **matplotlib** by default and needs no extra package.
> Installing `[viz]` (kaleido) additionally enables Plotly-native static
> rendering; the tool falls back automatically if kaleido is absent.

MD numeric arrays are also exported as `rmsd.csv` (`time_ps`, `rmsd_A`) and
`rmsf.csv` (`residue`, `rmsf_A`) per structure, so the plots build quickly
without re-loading trajectories.
---

## Project layout

```
proteoform_analyzer/
├── __init__.py            # package metadata, public config API
├── cli.py                 # command-line interface (`main`)
├── gui.py                 # Gradio web GUI (Setup / Run / Results)
├── core/
│   ├── config.py          # AnalysisConfig + presets
│   ├── pipeline.py        # StepResult, STEP_REGISTRY, run_analysis
│   ├── viz.py             # shared interactive+static plotting helpers
│   └── steps/             # one module per pipeline step
└── _vendored/             # bundled back-ends (DiffSBDD, P2Rank, pcn_miner,
                           #   ptmpsi, TMalign) — shipped as package data
```

For the full change history see `CHANGES.md`.

On the results-esm2umap there is a space for another plot or a table on the left, under the dropdown.
Same thing on the right of "3D Struture Viewer PDB", fill it with something useful. Also, regarding the 3d visualization of docked structures, why only chains relative to the target proteins are visibile? I want to see also the ligand (in licorice) or the antibody created. TM-scores with boltz are 1, is not possible... are you using round function? even a small difference of 0.01 is important for us. Predicted pocket volumes and drift are zero... 
Give me the modified code as a zip file.