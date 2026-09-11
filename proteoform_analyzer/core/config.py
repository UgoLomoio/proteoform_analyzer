"""Analysis configuration for the Proteoform Analyzer pipeline.

A single ``AnalysisConfig`` dataclass captures every user choice and is the single
entry point shared by the CLI and the Gradio GUI.  No pipeline function reads from
``input()``; everything comes from this config.

Generalized from the original tetramer-only design to support arbitrary oligomeric
states (monomers, dimers, trimers, tetramers, hexamers, etc.) and pairwise
proteoform combinatorics (mutation × PTM combinations).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json
import logging
import os
import re
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

log = logging.getLogger("proteoform_analyzer.config")


def parse_mature_regions(text: str, uids: list | None = None) -> dict:
    """Parse a manual mature-region spec into ``{uniprot_id: [start, end]}``.

    Format: ``"P02766:21-147; P69905:2-142"`` (entries separated by ``;`` or
    ``,``; ranges are 1-based, inclusive, UniProt precursor numbering). A bare
    ``"21-147"`` is accepted only when a single UniProt ID is configured.
    Unparseable entries are skipped with a warning.
    """
    text = (text or "").strip()
    if not text:
        return {}
    uids = [u for u in (uids or []) if u]
    regions: dict[str, list[int]] = {}
    # Split on ';' first; commas only separate entries when every comma-part
    # carries a "UID:" prefix (a bare "2-142, 2-147" would be ambiguous).
    parts: list[str] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "," in chunk and all(":" in c for c in chunk.split(",") if c.strip()):
            parts.extend(c.strip() for c in chunk.split(",") if c.strip())
        else:
            parts.append(chunk)
    for part in parts:
        if ":" in part:
            uid, _, rng = part.partition(":")
            uid = uid.strip()
        else:
            if len(uids) != 1:
                log.warning("Mature region '%s': bare START-END is only allowed "
                            "with a single UniProt ID; skipped", part)
                continue
            uid, rng = uids[0], part
        m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", rng.strip())
        if not m:
            log.warning("Mature region '%s': expected START-END (e.g. 21-147); "
                        "skipped", part)
            continue
        s, e = int(m.group(1)), int(m.group(2))
        if s < 1 or e < s:
            log.warning("Mature region '%s': invalid range %d-%d; skipped",
                        part, s, e)
            continue
        regions[uid] = [s, e]
    return regions


class EngineChoice(str, Enum):
    """User-selectable engines."""
    # MD
    MD_OPENMM = "openmm"
    MD_GROMACS = "gromacs"
    # Structure source (Boltz-2 folds structures from sequence via the hosted
    # Boltz API or a local `boltz` binary; graft fallback if neither available).
    STRUCT_BOLTZ2 = "boltz2"
    # Docking
    DOCK_VINA = "vina"
    DOCK_BOLTZ2 = "boltz2"        # protein+ligand co-folding (confidence proxy)
    DOCK_DIFFDOCK = "diffdock"    # legacy; documented no-op if service unavailable
    # Ligand design
    DESIGN_DIFFSBDD = "diffsbdd"
    DESIGN_BOLTZGEN = "boltzgen"
    # Antibody framework (RFAntibody frameworks bundled with a local install)
    AB_NANOBODY = "nanobody"
    AB_SCFV = "scfv"


class ProteoformMode(str, Enum):
    """How to combine mutations and PTMs into proteoforms."""
    SINGLE = "single"      # mutations and PTMs analyzed separately (no combinations)
    PAIRWISE = "pairwise"  # each mutation × each PTM type


class BindingSiteMethod(str, Enum):
    """Binding-site prediction strategy (tried in tier order)."""
    AUTO = "auto"          # tiered: reference → P2Rank → alpha-sphere → geometric
    REFERENCE = "reference"  # use known co-crystal structure only
    P2RANK = "p2rank"      # P2Rank pocket prediction only
    ALPHASPHERE = "alphasphere"  # pure-Python alpha-sphere detection only
    USER = "user"          # user-specified residues or coordinates


class HotspotSource(str, Enum):
    """Where antibody-design epitope hotspot residues come from."""
    USER = "user"          # user-selected residues (or binding_site_residues fallback)
    MHC_I = "mhc_i"        # AI-predicted MHC class I (T-cell) epitopes
    MHC_II = "mhc_ii"      # AI-predicted MHC class II (T-cell) epitopes
    BCELL = "bcell"        # AI-predicted B-cell (antibody) epitopes


@dataclass
class PTMConfig:
    """PTM simulation settings (ptmpsi).

    Three ways to specify PTMs, in precedence order (see core/steps/ptm.py):

    1. ``pairs`` — explicit (residue, ptm_type, uniprot_id|None) triples, e.g.
       from the GUI PTM builder. Only residue/PTM combinations that ptmpsi can
       physically model are accepted (see core/ptm_rules.py). A triple with a
       UniProt ID is applied only to that subunit's chains (via chain_map).
    2. ``residues`` x ``ptm_types`` — legacy cross-product; each combination is
       validated against the compatibility rules and invalid ones are skipped
       with a warning instead of failing at runtime.
    3. both empty — the ptm step auto-fetches *experimentally observed* PTMs
       from UniProt feature annotations, keeps only ptmpsi-modelable ones, and
       caps the total number of sites at ``AnalysisConfig.max_mutations``. If
       no observed/modelable PTM exists, the PTM step is skipped.
    """
    # residues to modify, e.g. ["CYS10"]; legacy cross-product mode
    residues: list[str] = field(default_factory=lambda: [])
    # PTM types to apply; ptmpsi PTMs: carbamoylation, sulfhydration,
    # sulfenylation, sulfinylation, sulfonylation, nitrosylation,
    # glutathionylation, cysteinylation, phosphorylation, acetylation,
    # methylation, dimethylation, trimethylation, ...
    ptm_types: list[str] = field(
        default_factory=lambda: []
    )
    # chains to modify (default all chains; auto-set by config based on n_subunits)
    chains: list[str] = field(default_factory=list)  # empty = all chains
    # Explicit (residue_spec, ptm_type, uniprot_id|None) triples from the GUI
    # PTM builder, e.g. [("CYS10", "glutathionylation", "P69905")]. Tuples or
    # 3-element lists are both accepted (JSON round-trip produces lists).
    pairs: list = field(default_factory=list)


@dataclass
class MDConfig:
    """Molecular dynamics settings."""
    engine: str = EngineChoice.MD_OPENMM.value       # "openmm" | "gromacs"
    # Fast config: short simulation for pipeline validation
    timestep_fs: float = 1.0
    nvt_steps: int = 1000          # ~1 ps
    npt_steps: int = 1000          # ~1 ps
    production_steps: int = 10000  # ~10 ps (fast); raise for real runs
    temperature_k: float = 300.0
    pressure_bar: float = 1.0
    ionic_strength_molar: float = 0.15
    forcefield: str = "amber14-all.xml"
    water_model: str = "amber14/tip3p.xml"
    # In vacuo (no solvent) for speed: ~20x fewer atoms, ~10x faster.
    # Set False for explicit solvent (water box, accurate but slow).
    implicit_solvent: bool = True
    # For GROMACS path
    gromacs_mdp_dir: Optional[str] = None


@dataclass
class Boltz2Config:
    """Boltz-2 structure-prediction / docking settings.

    Boltz-2 folds structures from sequence and runs on GPU. There are **three**
    possible backends, resolved in this order per job (see
    ``core/steps/_boltz_backend.resolve_backend``):

      1. **Hosted API** — the official Boltz API (``api.boltz.bio``). Enabled by
         setting ``api_key`` (or the ``BOLTZ_API_KEY`` env var). Covers structure
         + binding (folding & docking) and protein binder design (the BoltzGen
         replacement). No local GPU needed; predictions are billed to the user's
         key.
      2. **Local binary / package** — a local ``boltz`` binary on PATH (folding /
         docking), or an importable/PATH ``boltzgen`` (binder design), used when
         ``prefer_local`` is set or the API is unavailable. This runs
         **synchronously** on the local machine (a GPU is strongly recommended).
      3. **Graft fallback** — folding only: build backbone-identical structures
         with ptm-psi side-chain grafting (see ``allow_graft_fallback``). This
         is a *last resort*: TM-scores are 1.0 by construction and there is no
         structural / pocket / docking signal, so a prominent warning is raised.


    Docking and binder design have **no** graft equivalent (they need a real
    co-fold); when no fold/dock/design backend is available those steps skip
    cleanly.
    """
    # --- Hosted Boltz API (official api.boltz.bio) ---
    # If set (or BOLTZ_API_KEY is in the environment) and the `boltz_api` client
    # is importable, folding/docking/design use the hosted API.
    api_key: Optional[str] = None
    api_base_url: str = "https://api.boltz.bio"
    api_model: str = "boltz-2.1"
    # Reserved for future multi-provider support; only "boltz_api" is wired now.
    provider: str = "boltz_api"

    # --- Local boltz binary ---
    # If set (or 'boltz' is on PATH and prefer_local=True), shell out locally
    # instead of using the hosted API.
    local_binary: Optional[str] = None
    prefer_local: bool = True           # if True, use local_binary/PATH boltz when available

    # --- Graft fallback (folding only) ---
    # If True (default), and no API/local fold backend is available, fold by
    # ptm-psi side-chain grafting onto the WT backbone and raise a big warning
    # (TM=1.0, no structural signal). If False, structure folding is skipped
    # instead of producing backbone-identical grafts.
    allow_graft_fallback: bool = True

    # MSA generation is REQUIRED for protein sequences (Boltz-2 errors without it)
    use_msa_server: bool = True
    # Local Boltz weight cache dir (passed to `boltz predict --cache <dir>`).
    # Defaults to the standard local user cache; created on first use by boltz.
    cache_dir: str = os.path.expanduser("~/.cache/boltz")
    # Extra CLI flags appended verbatim (advanced use)
    extra_flags: list[str] = field(default_factory=list)
    # Docking: whether to run MSA server for the receptor when co-folding a ligand
    dock_use_msa_server: bool = True

    def resolved_api_key(self) -> Optional[str]:
        """Return the API key from config or the BOLTZ_API_KEY env var."""
        if self.api_key:
            return self.api_key
        return os.environ.get("BOLTZ_API_KEY") or None


@dataclass
class AntibodyConfig:
    """RFAntibody de novo antibody/nanobody design settings.

    Opt-in (enabled=False by default; the 'antibody' step is not in the default
    step list). Designs binders against a chosen proteoform structure at hotspot
    epitope residues that are either user-selected or AI-predicted.
    """
    enabled: bool = True
    # Framework baked into the RFAntibody install: "nanobody" or "scfv"
    framework: str = "nanobody"
    # Which structure to target: "wt" or an explicit mutant/proteoform name stem
    # (e.g. "Mut_P69905_D75A-assembly"). Default WT.
    target_structure: str = "wt"
    # Target chain (in the source PDB) to relabel as RFAntibody target chain 'T'
    target_chain: str = "A"
    # Where hotspot residues come from: "bcell" (default; predicted B-cell
    # epitopes) | "user" | "mhc_i" | "mhc_ii"
    hotspot_source: str = HotspotSource.BCELL.value
    # User-selected hotspot residues, e.g. ["A:62", "A:96"] (chain:resid, source
    # numbering). If empty and source == user, falls back to binding_site_residues.
    hotspot_residues: list[str] = field(default_factory=list)
    # For AI sources: how many top-scoring predicted epitope residues -> hotspots
    epitope_top_k: int = 6
    # CDR loop length ranges for RFdiffusion. None -> framework-appropriate
    # default (nanobody: "H1:7,H2:6,H3:5-13"; scfv adds the light-chain loops).
    design_loops: Optional[str] = None
    # Number of backbone designs (pilot scale by default)
    num_designs: int = 20
    # ProteinMPNN: sequences designed per backbone
    mpnn_seqs_per_struct: int = 2
    # RF2: number of recycles for the complex prediction
    rf2_recycles: int = 10
    # Crop target to ~this many Angstrom around hotspots (0 = no crop)
    crop_radius: float = 10.0
    # RF2 filtering thresholds (applied when results are retrieved)
    rf2_pae_max: float = 10.0
    rmsd_max: float = 2.0

    # --- Local RFAntibody execution ---
    # Path to a local RFAntibody checkout root (the dir containing ``scripts/``
    # with rfdiffusion_inference.py, proteinmpnn_interface_design.py,
    # rf2_predict.py). If set and the scripts + weights resolve, the 'antibody'
    # step runs the 3-stage pipeline synchronously on the local machine (GPU
    # strongly recommended). If unset/unresolvable, the step skips cleanly.
    local_rfantibody_dir: Optional[str] = f"{base_dir}/_vendored/RFantibody"
    # Dir containing the RFdiffusion_Ab checkpoint (RFdiffusion_Ab.pt) and any
    # other required weights. If None, defaults to <local_rfantibody_dir>/weights.
    local_weights_dir: Optional[str] = None
    # Optional Python interpreter to launch the RFAntibody scripts with (e.g. a
    # conda env python). Defaults to the current interpreter (sys.executable).
    local_python: Optional[str] = None
    # Optional path to a local nanobody/scFv framework PDB. If None, the step
    # looks for the framework baked into <local_rfantibody_dir> example inputs.
    local_framework_pdb: Optional[str] = None
    # If True (default), a missing RFantibody environment is bootstrapped
    # automatically on first use: `uv sync` creates the checkout's .venv and
    # the model weights (~750 MB) are downloaded from the IPD servers. If
    # False, the step only verifies and skips with instructions.
    auto_bootstrap: bool = True


@dataclass
class AnalysisConfig:
    """Top-level configuration for a single analysis run.

    Generalized to support arbitrary oligomeric states and proteoform combinatorics.
    """

    # --- Identity ---
    # For homo: 1 id; for hetero: N ids (e.g. ["P69905","P68871"] = HBA,HBB)
    uniprot_ids: list[str] = field(default_factory=list)
    # Optional display name; defaults to joined ids
    name: Optional[str] = None

    # --- Structural choices (generalized) ---
    # Number of subunits in the assembly (1=monomer, 2=dimer, 4=tetramer, 6=hexamer)
    n_subunits: int = 4
    # Stoichiometry as a list of copy counts, aligned with uniprot_ids.
    # Homo tetramer: [4]; hetero 2:2: [2,2]; hetero 4:2: [4,2]
    subunit_stoichiometry: list[int] = field(default_factory=lambda: [4])
    # Explicit chain map: {uniprot_id: [chain_ids]}. If empty, inferred from PDB.
    chain_map: dict[str, list[str]] = field(default_factory=dict)

    # --- Mutations ---
    # per-subunit mutation lists, aligned with uniprot_ids.
    # Each mutation "XnnY" (1-based). "WT" is always included automatically.
    mutations: list[list[str]] = field(default_factory=list)
    numbering_offset: int = 0
    max_mutations: Optional[int] = None   # cap total (fast test: 10)

    # --- Mature chain (proteolytic processing) ---
    # Per-subunit mature region as {uniprot_id: [start, end]} in UniProt
    # (precursor) numbering, 1-based, inclusive — e.g. {"P02766": [21, 147]}
    # for TTR (signal peptide 1-20 cleaved). The mature sequence
    # seq[start-1:end] is what gets folded / embedded / docked; mutations and
    # PTM sites outside the region are flagged in mature_chain_report.csv and
    # excluded from structural steps with a warning. Manual entries here win
    # over auto-detection. Default {} = full-length for every subunit.
    mature_regions: dict[str, list[int]] = field(default_factory=dict)
    # If True (default), the sequence step auto-detects the mature region from
    # UniProt features (single CHAIN, else terminal SIGNAL/PROPEP/TRANSIT
    # trimming) for every UniProt ID without a manual entry.
    mature_auto_detect: bool = True

    # --- Proteoform combinatorics ---
    proteoform_mode: str = ProteoformMode.PAIRWISE.value  # "single" | "pairwise"
    proteoform_cap: int = 100  # max proteoforms to generate

    # --- Structure source ---
    # Boltz-2 folds WT + mutant structures from sequence. Backend resolved as
    # hosted Boltz API -> local `boltz` binary -> graft fallback (see
    # Boltz2Config). This is the only structure source.
    structure_source: str = EngineChoice.STRUCT_BOLTZ2.value
    # Reference/template PDB ID used ONLY for known-binding-site alignment (e.g.
    # 5E83 voxelotor, 4DST tafamidis) and optional validation — NOT the structure
    # source. Kept for the internal reference-PDB fetch helper.
    local_pdb_id: Optional[str] = None

    # --- Boltz-2 (structure + docking) ---
    boltz2: Boltz2Config = field(default_factory=Boltz2Config)

    # --- Antibody design (RFAntibody; opt-in) ---
    antibody: AntibodyConfig = field(default_factory=AntibodyConfig)

    # --- PTM ---
    ptm: PTMConfig = field(default_factory=PTMConfig)
    run_ptm: bool = True

    # --- MD ---
    md: MDConfig = field(default_factory=MDConfig)
    run_md: bool = True

    # --- Engine choices ---
    docking_engine: str = EngineChoice.DOCK_VINA.value
    ligand_design_engine: str = EngineChoice.DESIGN_DIFFSBDD.value

    # --- Binding-site prediction ---
    binding_site_method: str = BindingSiteMethod.AUTO.value
    binding_site_residues: Optional[list[str]] = None  # user-specified, e.g. ["A:62","A:96"]
    binding_site_center: Optional[list[float]] = None  # user-specified [x,y,z]
    binding_site_box_size: float = 24.0

    # --- Multi-ligand screening ---
    # Multiple ligand SDF paths for screening; None = auto-select single ligand
    ligand_sdfs: Optional[list[str]] = None
    # Single ligand (backward compat; if set, used as the only ligand)
    ligand_sdf: Optional[str] = None

    # --- Ensemble docking ---
    ensemble_docking: bool = False
    ensemble_n_snapshots: int = 3

    # --- Impact score weights ---
    # Weights for composite impact score (must sum to 1.0; auto-normalized if not)
    impact_weights: dict[str, float] = field(default_factory=lambda: {
        "structural": 0.2, "binding": 0.3, "dynamics": 0.2,
        "network": 0.15, "sequence": 0.15,
    })

    # --- DiffSBDD ---
    diffsbdd_checkpoint: Optional[str] = f"{base_dir}/_vendored/diffsbdd/checkpoints/crossdocked_ca_cond.ckpt"
    diffsbdd_n_samples: int = 5
    diffsbdd_pocket_residues: Optional[list[int]] = None

    # --- ddG / ThermoMPNN local execution ---
    # Path to a local ThermoMPNN checkout/dir containing a ``custom_inference.py``
    # (or set thermompnn_script directly). If set with a checkpoint, the ddG step
    # Tier 1 runs ThermoMPNN locally (GPU recommended); otherwise it falls back
    # automatically to the CPU ESM2 zero-shot ddG scorer (no GPU needed).
    thermompnn_dir: Optional[str] = f"{base_dir}/_vendored/ThermoMPNN"
    # Explicit path to the ThermoMPNN inference script (overrides <dir>/custom_inference.py).
    thermompnn_script: Optional[str] = f"{base_dir}/_vendored/ThermoMPNN/analysis/custom_inference.py"
    # Path to the ThermoMPNN model checkpoint (e.g. thermoMPNN_default.pt).
    thermompnn_checkpoint: Optional[str] = f"{base_dir}/_vendored/ThermoMPNN/models/thermoMPNN_default.pt"
    # Optional Python interpreter for the ThermoMPNN script (defaults to sys.executable).
    thermompnn_python: Optional[str] = None

    # --- ddG / FoldX local execution (PTM stability via mimetic substitutions) ---
    # Path to a local FoldX executable (FoldX 4/5). FoldX is academic-licensed
    # and cannot be redistributed, so the pipeline never ships it: set this path
    # (or $FOLDX_BINARY) to enable the FoldX tier of the ddG step, which scores
    # PTM stability effects as PTM-mimetic substitutions (e.g. phosphomimetic
    # SER/THR->GLU, acetyl-mimetic LYS->GLN; see core/ptm_rules.py). When unset,
    # PTM ddG is skipped cleanly and mutation ddG is unaffected.
    foldx_binary: Optional[str] = None
    # Path to the FoldX rotabase.txt. When None, it is looked up next to the
    # FoldX binary (the standard FoldX distribution layout).
    foldx_rotabase: Optional[str] = None
    # NumberOfRuns for FoldX BuildModel (replicate runs averaged per mutation).
    foldx_n_runs: int = 1

    # --- Which steps to run ---
    steps: list[str] = field(default_factory=lambda: [
        "sequence", "structure", "ptm", "proteoform", "tmscore", "pcn", "md",
        "esm2", "pocket", "docking", "ligand_design", "ddg", "impact_score", "antibody"
    ])

    @staticmethod
    def default_steps() -> list[str]:
        """Default pipeline steps (antibody is opt-in and NOT included)."""
        return [
            "sequence", "structure", "ptm", "proteoform", "tmscore", "pcn", "md",
            "esm2", "pocket", "docking", "ligand_design", "ddg", "impact_score", "antibody"
        ]

    # --- Paths ---
    work_dir: str = "results"
    input_dir: str = "input"

    # --- Callback for progress (CLI/GUI set this) ---
    progress_callback: Optional[object] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.name is None:
            self.name = "_".join(self.uniprot_ids) if self.uniprot_ids else "analysis"
        # Auto-set PTM chains if not specified
        if not self.ptm.chains:
            self.ptm.chains = [chr(65 + i) for i in range(self.n_subunits)]

    @property
    def is_homo(self) -> bool:
        """True if all subunits are the same (single UniProt ID)."""
        return len(self.subunit_stoichiometry) <= 1

    @property
    def is_hetero(self) -> bool:
        return not self.is_homo

    @property
    def n_unique_subunits(self) -> int:
        """Number of distinct subunit types."""
        return len(self.subunit_stoichiometry)

    @property
    def total_chains(self) -> int:
        """Total number of chains in the assembly."""
        return sum(self.subunit_stoichiometry)

    @property
    def is_monomer(self) -> bool:
        return self.n_subunits == 1

    def results_dir(self) -> str:
        return os.path.join(self.work_dir, self.name)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("progress_callback", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "AnalysisConfig":
        d = json.loads(s)
        if "ptm" in d and isinstance(d["ptm"], dict):
            d["ptm"] = PTMConfig(**d["ptm"])
        if "md" in d and isinstance(d["md"], dict):
            d["md"] = MDConfig(**d["md"])
        if "boltz2" in d and isinstance(d["boltz2"], dict):
            d["boltz2"] = Boltz2Config(**d["boltz2"])
        if "antibody" in d and isinstance(d["antibody"], dict):
            d["antibody"] = AntibodyConfig(**d["antibody"])
        d.pop("progress_callback", None)
        return cls(**d)


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def hemoglobin_fast_config() -> AnalysisConfig:
    """Fast-test config for the hemoglobin heterotetramer.

    HBA = P69905 (alpha), HBB = P68871 (beta).  10 biologically-informative
    mutations (5 per subunit) including HbS (E7V, sickle-cell).

    Mutations use UniProt numbering (includes initiator Met).
    """
    hba_muts = []
    hbb_muts = []

    return AnalysisConfig(
        uniprot_ids=["P69905", "P68871"],
        name="hemoglobin",
        n_subunits=4,
        subunit_stoichiometry=[2, 2],
        chain_map={"P69905": ["A", "C"], "P68871": ["B", "D"]},
        mutations=[hba_muts, hbb_muts],
        max_mutations=10,
        proteoform_mode=ProteoformMode.SINGLE.value,
        proteoform_cap=100,
        structure_source=EngineChoice.STRUCT_BOLTZ2.value,
        local_pdb_id="1A3N",
        # Mature chains: initiator Met cleaved in both subunits (UniProt CHAIN
        # P69905 2-142, P68871 2-147).
        mature_regions={"P69905": [2, 142], "P68871": [2, 147]},
        run_ptm=False,  # no PTMs for hemoglobin
        run_md=True,
        md=MDConfig(engine=EngineChoice.MD_OPENMM.value, production_steps=1000,
                    implicit_solvent=True),
        docking_engine=EngineChoice.DOCK_VINA.value,
        ligand_design_engine=EngineChoice.DESIGN_DIFFSBDD.value,
        binding_site_method=BindingSiteMethod.AUTO.value,
        diffsbdd_n_samples=5,
        steps=["sequence", "structure", "ptm", "proteoform", "tmscore", "pcn", "md",
               "esm2", "pocket", "docking", "ligand_design", "ddg", "impact_score", "antibody"],
    )


def ttr_fast_config() -> AnalysisConfig:
    """Fast config for transthyretin (TTR) homotetramer.

    TTR = P02766.  10 clinically relevant amyloidogenic mutations.
    Tafamidis is the FDA-approved TTR stabilizer (reference: PDB 1F41).
    In vacuo MD, 1000 steps (~1 ps) for rapid pipeline validation.
    """
    ttr_muts = []
    return AnalysisConfig(
        uniprot_ids=["P02766"],
        name="ttr",
        n_subunits=4,
        subunit_stoichiometry=[4],
        chain_map={"P02766": ["A", "B", "C", "D"]},
        mutations=[ttr_muts],
        max_mutations=10,
        proteoform_mode=ProteoformMode.SINGLE.value,
        proteoform_cap=100,
        structure_source=EngineChoice.STRUCT_BOLTZ2.value,
        local_pdb_id="1F41",
        # Mature chain: TTR signal peptide (residues 1-20) is cleaved; the
        # tetramer is formed by four mature monomers (21-147).
        mature_regions={"P02766": [21, 147]},
        run_ptm=False,  # no PTMs for TTR
        run_md=True,
        md=MDConfig(engine=EngineChoice.MD_OPENMM.value, production_steps=1000,
                    implicit_solvent=True),
        docking_engine=EngineChoice.DOCK_VINA.value,
        ligand_design_engine=EngineChoice.DESIGN_DIFFSBDD.value,
        binding_site_method=BindingSiteMethod.AUTO.value,
        diffsbdd_n_samples=5,
        steps=["sequence", "structure", "ptm", "proteoform", "tmscore", "pcn", "md",
               "esm2", "pocket", "docking", "ligand_design", "ddg", "impact_score", "antibody"],
    )


def p53_fast_config() -> AnalysisConfig:
    """Fast config for p53 tetramerization domain (homotetramer).

    p53 = P04637.  PDB 1PES has the full tetramer (4 chains, residues 325-356).
    10 cancer-associated mutations in/near the tetramerization domain.
    No known small-molecule stabilizer → uses P2Rank pocket prediction + DiffSBDD.
    In vacuo MD, 1000 steps (~1 ps) for rapid pipeline validation.
    """
    p53_muts = []
    return AnalysisConfig(
        uniprot_ids=["P04637"],
        name="p53",
        n_subunits=1,
        subunit_stoichiometry=[1],
        chain_map={"P04637": ["A"]},
        mutations=[p53_muts],
        max_mutations=10,
        proteoform_mode=ProteoformMode.SINGLE.value,
        proteoform_cap=100,
        structure_source=EngineChoice.STRUCT_BOLTZ2.value,
        local_pdb_id="1PES",
        run_ptm=False,  # no PTMs for p53 tetramerization domain
        run_md=True,
        md=MDConfig(engine=EngineChoice.MD_OPENMM.value, production_steps=1000,
                    implicit_solvent=True),
        docking_engine=EngineChoice.DOCK_VINA.value,
        ligand_design_engine=EngineChoice.DESIGN_DIFFSBDD.value,
        binding_site_method=BindingSiteMethod.P2RANK.value,  # no reference ligand
        diffsbdd_n_samples=5,
        steps=["sequence", "structure", "ptm", "proteoform", "tmscore", "pcn", "md",
               "esm2", "pocket", "docking", "ligand_design", "ddg", "impact_score", "antibody"],
    )


# ---------------------------------------------------------------------------
# Standard and Production presets (speed tiers)
# ---------------------------------------------------------------------------

_ALL_STEPS = ["sequence", "structure", "ptm", "proteoform", "tmscore", "pcn", "md",
              "esm2", "pocket", "docking", "ligand_design", "ddg", "impact_score", "antibody"]


def hemoglobin_standard_config() -> AnalysisConfig:
    """Standard config for hemoglobin: in vacuo MD 5000 steps, 10 curated mutations."""
    hba_muts = []
    hbb_muts = []
    return AnalysisConfig(
        uniprot_ids=["P69905", "P68871"],
        name="hemoglobin",
        n_subunits=4,
        subunit_stoichiometry=[2, 2],
        chain_map={"P69905": ["A", "C"], "P68871": ["B", "D"]},
        mutations=[hba_muts, hbb_muts],
        max_mutations=10,
        proteoform_mode=ProteoformMode.SINGLE.value,
        proteoform_cap=100,
        structure_source=EngineChoice.STRUCT_BOLTZ2.value,
        local_pdb_id="1A3N",
        # Mature chains: initiator Met cleaved in both subunits (UniProt CHAIN
        # P69905 2-142, P68871 2-147).
        mature_regions={"P69905": [2, 142], "P68871": [2, 147]},
        run_ptm=False,
        run_md=True,
        md=MDConfig(engine=EngineChoice.MD_OPENMM.value, production_steps=5000,
                    implicit_solvent=True),
        docking_engine=EngineChoice.DOCK_VINA.value,
        ligand_design_engine=EngineChoice.DESIGN_DIFFSBDD.value,
        binding_site_method=BindingSiteMethod.AUTO.value,
        diffsbdd_n_samples=5,
        steps=list(_ALL_STEPS),
    )


def hemoglobin_production_config() -> AnalysisConfig:
    """Production config for hemoglobin: explicit solvent MD 50000 steps, all UniProt variants."""
    hba_muts = [
        "V2E", "L3R", "A6D", "A6P", "D7A", "D7G", "D7N", "D7V", "D7Y",
        "K8E", "N10T", "K12E", "A13D", "A14P", "W15R", "G16R", "K17M",
        "K17N", "G19D", "G19R", "A20D", "A20E", "H21Q", "H21R", "Y25H",
        "E24G", "E24K", "E28D", "E28G", "E28V", "E31K", "R32K", "R32S",
        "L35R", "K41M", "P38R", "T42S", "F44L", "P45L", "P45R", "D48A",
        "D48G", "D48H", "D48Y", "L49R", "S50R", "G52D", "G52R", "A54D",
        "V63M", "K61N", "K62N", "K62T", "A64D", "K57R", "K57T", "G58R",
        "G60D", "G60V", "D65Y", "A72E", "A72V", "H73R", "D75A", "D75G",
        "D75N", "D76A", "D76H", "M77K", "M77T", "L81R", "S82C", "S85R",
        "D86V", "D86Y", "L87R", "K91M", "L92F", "L92P", "R93Q", "R93W",
        "P96A", "P96T", "D95A", "D95Y", "S103R", "K100E", "H104R",
        "H104Y", "L110R", "A111D", "H113D", "N69K", "N79H", "N79K",
        "N98K",
    ]

    hbb_muts = [
        "V2A", "H3L", "H3Q", "H3R", "H3Y", "S10C", "A11D", "A11V",
        "P6R", "V12D", "V12I", "A14D", "E7A", "E7K", "E7Q", "E7V",
        "E8G", "E8K", "K9E", "K9Q", "K9T", "L15P", "L15R", "W16G",
        "W16R",
    ]
    return AnalysisConfig(
        uniprot_ids=["P69905", "P68871"],
        name="hemoglobin",
        n_subunits=4,
        subunit_stoichiometry=[2, 2],
        chain_map={"P69905": ["A", "C"], "P68871": ["B", "D"]},
        mutations=[hba_muts, hbb_muts],
        max_mutations=50,
        proteoform_mode=ProteoformMode.SINGLE.value,
        proteoform_cap=500,
        structure_source=EngineChoice.STRUCT_BOLTZ2.value,
        local_pdb_id="1A3N",
        # Mature chains: initiator Met cleaved in both subunits (UniProt CHAIN
        # P69905 2-142, P68871 2-147).
        mature_regions={"P69905": [2, 142], "P68871": [2, 147]},
        run_ptm=False,
        run_md=True,
        md=MDConfig(engine=EngineChoice.MD_OPENMM.value, production_steps=5000,
                    nvt_steps=5000, npt_steps=5000, implicit_solvent=False),
        docking_engine=EngineChoice.DOCK_VINA.value,
        ligand_design_engine=EngineChoice.DESIGN_DIFFSBDD.value,
        binding_site_method=BindingSiteMethod.AUTO.value,
        diffsbdd_n_samples=5,
        steps=list(_ALL_STEPS),
    )


def ttr_standard_config() -> AnalysisConfig:
    """Standard config for TTR: in vacuo MD 5000 steps, 10 curated mutations."""
    ttr_muts = []
    return AnalysisConfig(
        uniprot_ids=["P02766"],
        name="ttr",
        n_subunits=4,
        subunit_stoichiometry=[4],
        chain_map={"P02766": ["A", "B", "C", "D"]},
        mutations=[ttr_muts],
        max_mutations=10,
        proteoform_mode=ProteoformMode.SINGLE.value,
        proteoform_cap=100,
        structure_source=EngineChoice.STRUCT_BOLTZ2.value,
        local_pdb_id="1F41",
        # Mature chain: TTR signal peptide (residues 1-20) is cleaved; the
        # tetramer is formed by four mature monomers (21-147).
        mature_regions={"P02766": [21, 147]},
        run_ptm=False,  # no PTMs for TTR
        run_md=True,
        md=MDConfig(engine=EngineChoice.MD_OPENMM.value, production_steps=5000,
                    implicit_solvent=True),
        docking_engine=EngineChoice.DOCK_VINA.value,
        ligand_design_engine=EngineChoice.DESIGN_DIFFSBDD.value,
        binding_site_method=BindingSiteMethod.AUTO.value,
        diffsbdd_n_samples=5,
        steps=list(_ALL_STEPS),
    )


def ttr_production_config() -> AnalysisConfig:
    """Production config for TTR: explicit solvent MD 50000 steps, all UniProt variants."""
    ttr_muts = []
    return AnalysisConfig(
        uniprot_ids=["P02766"],
        name="ttr",
        n_subunits=4,
        subunit_stoichiometry=[4],
        chain_map={"P02766": ["A", "B", "C", "D"]},
        mutations=[ttr_muts],
        max_mutations=None,
        proteoform_mode=ProteoformMode.SINGLE.value,
        proteoform_cap=500,
        structure_source=EngineChoice.STRUCT_BOLTZ2.value,
        local_pdb_id="1F41",
        # Mature chain: TTR signal peptide (residues 1-20) is cleaved; the
        # tetramer is formed by four mature monomers (21-147).
        mature_regions={"P02766": [21, 147]},
        run_ptm=False,
        run_md=True,
        md=MDConfig(engine=EngineChoice.MD_OPENMM.value, production_steps=5000,
                    nvt_steps=5000, npt_steps=5000, implicit_solvent=False),
        docking_engine=EngineChoice.DOCK_VINA.value,
        ligand_design_engine=EngineChoice.DESIGN_DIFFSBDD.value,
        binding_site_method=BindingSiteMethod.AUTO.value,
        diffsbdd_n_samples=5,
        steps=list(_ALL_STEPS),
    )

def ttr_production_ptm() -> AnalysisConfig:
    """Production config for TTR: explicit solvent MD 5000 steps, PTMs."""
    ptm_config = PTMConfig(
        residues=[],
        ptm_types=[],
        chains=[],  # all chains A,B,C,D
        pairs=[
            ("CYS10", "nitrosylation", None),
            ("THR23", "phosphorylation", None),
            ("THR25", "phosphorylation", None),
            ("SER72", "phosphorylation", None),
            ("TYR125", "phosphorylation", None),
            ("SER137", "phosphorylation", None),
            ("THR138", "phosphorylation", None),
        ],
    )
    return AnalysisConfig(
        uniprot_ids=["P02766"],
        name="ttr_ptms",
        n_subunits=4,
        subunit_stoichiometry=[4],
        chain_map={"P02766": ["A", "B", "C", "D"]},
        mutations=[],
        max_mutations=0,
        proteoform_mode=ProteoformMode.SINGLE.value,
        proteoform_cap=500,
        structure_source=EngineChoice.STRUCT_BOLTZ2.value,
        local_pdb_id="1F41",
        # Mature chain: TTR signal peptide (residues 1-20) is cleaved; the
        # tetramer is formed by four mature monomers (21-147).
        mature_regions={"P02766": [21, 147]},
        ptm=ptm_config,
        run_ptm=True,
        run_md=True,
        md=MDConfig(engine=EngineChoice.MD_OPENMM.value, production_steps=5000,
                    nvt_steps=5000, npt_steps=5000, implicit_solvent=False),
        docking_engine=EngineChoice.DOCK_VINA.value,
        ligand_design_engine=EngineChoice.DESIGN_DIFFSBDD.value,
        binding_site_method=BindingSiteMethod.AUTO.value,
        diffsbdd_n_samples=5,
        steps=list(_ALL_STEPS),
    )

def p53_standard_config() -> AnalysisConfig:
    """Standard config for p53: in vacuo MD 5000 steps, 10 curated mutations."""
    p53_muts = []
    return AnalysisConfig(
        uniprot_ids=["P04637"],
        name="p53",
        n_subunits=1,
        subunit_stoichiometry=[1],
        chain_map={"P04637": ["A"]},
        mutations=[p53_muts],
        max_mutations=10,
        proteoform_mode=ProteoformMode.SINGLE.value,
        proteoform_cap=100,
        structure_source=EngineChoice.STRUCT_BOLTZ2.value,
        local_pdb_id="1PES",
        run_ptm=False,
        run_md=True,
        md=MDConfig(engine=EngineChoice.MD_OPENMM.value, production_steps=5000,
                    implicit_solvent=True),
        docking_engine=EngineChoice.DOCK_VINA.value,
        ligand_design_engine=EngineChoice.DESIGN_DIFFSBDD.value,
        binding_site_method=BindingSiteMethod.P2RANK.value,
        diffsbdd_n_samples=5,
        steps=list(_ALL_STEPS),
    )


def p53_production_config() -> AnalysisConfig:
    """Production config for p53: explicit solvent MD 50000 steps, all UniProt variants."""
    p53_muts = []
    return AnalysisConfig(
        uniprot_ids=["P04637"],
        name="p53",
        n_subunits=1,
        subunit_stoichiometry=[1],
        chain_map={"P04637": ["A"]},
        mutations=p53_muts,
        max_mutations=None,
        proteoform_mode=ProteoformMode.SINGLE.value,
        proteoform_cap=500,
        structure_source=EngineChoice.STRUCT_BOLTZ2.value,
        local_pdb_id="1PES",
        run_ptm=False,
        run_md=True,
        md=MDConfig(engine=EngineChoice.MD_OPENMM.value, production_steps=5000,
                    nvt_steps=5000, npt_steps=5000, implicit_solvent=False),
        docking_engine=EngineChoice.DOCK_VINA.value,
        ligand_design_engine=EngineChoice.DESIGN_DIFFSBDD.value,
        binding_site_method=BindingSiteMethod.P2RANK.value,
        diffsbdd_n_samples=5,
        steps=list(_ALL_STEPS),
    )


# ---------------------------------------------------------------------------
# Preset registry (used by GUI + CLI)
# ---------------------------------------------------------------------------

PRESETS = {
    "Hemoglobin — Fast": hemoglobin_fast_config,
    "Hemoglobin — Standard": hemoglobin_standard_config,
    "Hemoglobin — Production": hemoglobin_production_config,
    "TTR — Fast": ttr_fast_config,
    "TTR — Standard": ttr_standard_config,
    "TTR — Production": ttr_production_config,
    "TTR — PTMs": ttr_production_ptm,
    "p53 — Fast": p53_fast_config,
    "p53 — Standard": p53_standard_config,
    "p53 — Production": p53_production_config,
}
