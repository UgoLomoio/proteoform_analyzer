"""Pipeline orchestration: run_analysis() drives every step from AnalysisConfig.

Each step is a function (config, paths, log) -> StepResult.  Steps are registered
in STEP_REGISTRY so the CLI/GUI can list and select them.  No step uses input();
all choices come from the config.
"""
from __future__ import annotations

import os
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import AnalysisConfig

log = logging.getLogger("proteoform_analyzer")


@dataclass
class StepResult:
    step: str
    status: str            # "ok" | "skipped" | "failed"
    message: str = ""
    outputs: list[str] = field(default_factory=list)   # file paths produced
    data: object = None    # optional structured result (df/dict)
    elapsed_s: float = 0.0


def _make_paths(config: AnalysisConfig) -> dict:
    """Create the standard results directory tree and return a paths dict."""
    base = config.results_dir()
    # if the result path already exists we delete all contents to avoid confusion with old results
    if os.path.exists(base):
        import shutil
        shutil.rmtree(base)
    os.makedirs(base, exist_ok=True)
    paths = {
        "results": base,
        "input": config.input_dir,
        "fasta": os.path.join(base, "fasta_sequences"),
        "pdbs": os.path.join(base, "pdbs", "tetramer"),
        "pdbs_monomer": os.path.join(base, "pdbs", "monomer"),
        "sdfs": os.path.join(base, "sdfs"),
        "docking": os.path.join(base, "docked-outputs"),
        "json_jobs": os.path.join(base, "json_jobs"),
        "af_output": os.path.join(base, "af_output"),
        "cifs": os.path.join(base, "cifs"),
        "molecular_dynamics": os.path.join(base, "molecular_dynamics"),
        "ptms": os.path.join(base, "ptms"),
        "embeddings": os.path.join(base, "embeddings"),
        "tmalign": os.path.join(base, "tmalign"),
        "pcn_outputs": os.path.join(base, "pcn_outputs"),
        "ligand_design": os.path.join(base, "ligand_design"),
        "ddg": os.path.join(base, "ddg"),
        "proteoforms": os.path.join(base, "proteoforms"),
        "pockets": os.path.join(base, "pockets"),
        "impact": os.path.join(base, "impact_scores"),
        "boltz_structures": os.path.join(base, "boltz_structures"),
        "antibody": os.path.join(base, "antibody"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _emit(config: AnalysisConfig, step: str, status: str, message: str):
    """Send progress to the callback (if any) and the logger."""
    msg = f"[{step}] {status}: {message}"
    if status == "ok":
        log.info(msg)
    elif status == "skipped":
        log.warning(msg)
    else:
        log.error(msg)
    if config.progress_callback is not None:
        try:
            config.progress_callback(step, status, message)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Step implementations (imported lazily so optional deps don't break import)
# ---------------------------------------------------------------------------

def step_sequence(config: AnalysisConfig, paths: dict) -> StepResult:
    """Fetch sequences + mutation consequences from UniProt (cached locally)."""
    from .steps.sequence import fetch_sequences_and_mutations
    return fetch_sequences_and_mutations(config, paths)


def step_structure(config: AnalysisConfig, paths: dict) -> StepResult:
    """Prepare WT + mutant structures by folding sequences with Boltz-2."""
    from .steps.structure import prepare_structures
    return prepare_structures(config, paths)


def step_ptm(config: AnalysisConfig, paths: dict) -> StepResult:
    """Generate PTM-modified structures via ptmpsi."""
    from .steps.ptm import generate_ptms
    return generate_ptms(config, paths)


def step_tmscore(config: AnalysisConfig, paths: dict) -> StepResult:
    """Compute pairwise TM-scores between WT and mutant structures."""
    from .steps.tmscore import compute_tmscores
    return compute_tmscores(config, paths)


def step_pcn(config: AnalysisConfig, paths: dict) -> StepResult:
    """Protein Contact Network centrality + community detection."""
    from .steps.pcn import run_pcn
    return run_pcn(config, paths)


def step_md(config: AnalysisConfig, paths: dict) -> StepResult:
    """Molecular dynamics (OpenMM or GROMACS)."""
    from .steps.md import run_md
    return run_md(config, paths)


def step_esm2(config: AnalysisConfig, paths: dict) -> StepResult:
    """ESM2 embeddings + UMAP variant classification."""
    from .steps.esm2 import run_esm2
    return run_esm2(config, paths)


def step_docking(config: AnalysisConfig, paths: dict) -> StepResult:
    """Docking (AutoDock Vina or DiffDock)."""
    from .steps.docking import run_docking
    return run_docking(config, paths)


def step_ligand_design(config: AnalysisConfig, paths: dict) -> StepResult:
    """Ligand design (DiffSBDD or BoltzGen)."""
    from .steps.ligand_design import run_ligand_design
    return run_ligand_design(config, paths)


def step_ddg(config: AnalysisConfig, paths: dict) -> StepResult:
    """DeltaDeltaG stability prediction (DynaMut2 web service)."""
    from .steps.ddg import run_ddg
    return run_ddg(config, paths)


def step_proteoform(config: AnalysisConfig, paths: dict) -> StepResult:
    """Generate pairwise mutation+PTM proteoform structures."""
    from .steps.proteoform import generate_proteoforms
    return generate_proteoforms(config, paths)


def step_pocket(config: AnalysisConfig, paths: dict) -> StepResult:
    """Binding-site prediction + pocket drift analysis."""
    from .steps.pocket import run_pocket
    return run_pocket(config, paths)


def step_impact_score(config: AnalysisConfig, paths: dict) -> StepResult:
    """Composite proteoform impact score."""
    from .steps.impact_score import run_impact_score
    return run_impact_score(config, paths)


def step_antibody(config: AnalysisConfig, paths: dict) -> StepResult:
    """De novo antibody/nanobody design against a binding site (RFAntibody)."""
    from .steps.antibody import run_antibody
    return run_antibody(config, paths)


# Registry: name -> (description, function)
STEP_REGISTRY: dict[str, tuple[str, Callable]] = {
    "sequence":      ("Sequence + mutation retrieval (UniProt)", step_sequence),
    "structure":     ("Structure preparation (Boltz-2 folding from sequence)", step_structure),
    "ptm":           ("PTM generation (PTM-Psi / ptmpsi)", step_ptm),
    "tmscore":       ("TM-score structural comparison (TM-align)", step_tmscore),
    "pcn":           ("Protein Contact Network analysis (PCN-Miner)", step_pcn),
    "md":            ("Molecular dynamics (OpenMM / GROMACS)", step_md),
    "esm2":          ("ESM2 + UMAP variant classification", step_esm2),
    "docking":       ("Docking (AutoDock Vina / Boltz-2 co-folding)", step_docking),
    "ligand_design": ("Ligand design (DiffSBDD / BoltzGen)", step_ligand_design),
    "ddg":           ("DeltaDeltaG stability (ThermoMPNN / ESM2 zero-shot)", step_ddg),
    "proteoform":    ("Proteoform generation (pairwise mutation+PTM)", step_proteoform),
    "pocket":        ("Binding-site prediction + pocket drift", step_pocket),
    "impact_score":  ("Composite proteoform impact score", step_impact_score),
    "antibody":      ("Antibody/nanobody design (RFAntibody; opt-in)", step_antibody),
}


def run_analysis(config: AnalysisConfig) -> list[StepResult]:
    """Run the configured pipeline steps in order. Returns per-step results."""
    # Configure logging once
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    paths = _make_paths(config)
    results: list[StepResult] = []
    _emit(config, "pipeline", "ok", f"Starting analysis '{config.name}' with steps: {config.steps}")
    for step_name in config.steps:
        if step_name not in STEP_REGISTRY:
            r = StepResult(step_name, "skipped", f"Unknown step '{step_name}'")
            results.append(r)
            _emit(config, step_name, "skipped", f"Unknown step")
            continue
        desc, fn = STEP_REGISTRY[step_name]
        _emit(config, step_name, "ok", f"Running: {desc}")
        t0 = time.time()
        try:
            r = fn(config, paths)
            r.elapsed_s = time.time() - t0
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            r = StepResult(step_name, "failed", f"{type(e).__name__}: {e}", elapsed_s=time.time() - t0)
            log.error(tb)
        results.append(r)
        _emit(config, step_name, r.status, f"{r.message} ({r.elapsed_s:.1f}s)")
    # Summary
    ok = sum(1 for r in results if r.status == "ok")
    sk = sum(1 for r in results if r.status == "skipped")
    fl = sum(1 for r in results if r.status == "failed")
    _emit(config, "pipeline", "ok", f"Done. ok={ok} skipped={sk} failed={fl}")
    return results
