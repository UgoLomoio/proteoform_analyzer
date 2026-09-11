"""Step: structure preparation via Boltz-2 (replaces local_pdb / AF3).

Boltz-2 folds WT + mutant structures from sequence. The backend is chosen by
``_boltz_backend.resolve_backend(config, "fold")`` in this order:

  - **Hosted API:** the official Boltz API (``api.boltz.bio``), used when
    ``boltz2.api_key`` / ``$BOLTZ_API_KEY`` is set.

  - **Local binary:** if ``config.boltz2.local_binary`` is set (or ``boltz`` is
    on PATH and ``prefer_local=True``), shell out to the local ``boltz`` binary
    **synchronously** and convert its mmCIF output in place (GPU recommended).

  - **Graft fallback (folding only):** if neither API nor a local binary is
    available and ``boltz2.allow_graft_fallback`` is set, build
    backbone-identical structures by PTM-Psi side-chain grafting onto a
    reference backbone (TM=1.0, no structural signal; a warning is surfaced).

  - Otherwise the step **skips cleanly** with an actionable message.

As of v3.4.0 the Biomni HPC backend has been removed: this step no longer
submits GPU jobs to any external cluster and runs entirely locally / via the
hosted API.

Output PDBs are written to the SAME canonical locations the old local_pdb path
used (``pdbs/tetramer/wt-<uid>-tetramer.pdb``, ``Mut_<uid>_<mut>-tetramer.pdb``, or the
monomer equivalents) so PTM-Psi, TM-align, PCN, MD, pocket, docking, ddG and
impact all keep working without changes.
"""
from __future__ import annotations

import os
import shutil
import logging
import subprocess

from ..pipeline import StepResult
from ._pdb_utils import cif_to_pdb, apply_mutation

log = logging.getLogger("proteoform_analyzer.boltz2_fold")


# ---------------------------------------------------------------------------
# YAML construction
# ---------------------------------------------------------------------------

def _chain_ids(n: int) -> list[str]:
    """Chain ids A, B, C, ... for n chains (Boltz-2 uses single-letter ids)."""
    return [chr(65 + i) for i in range(n)]


def _remap_mutation_to_mature(config, subunit_idx: int, mut: str) -> str | None:
    """Remap a UniProt-numbered mutation to mature-chain numbering.

    Returns the mutation string unchanged when the subunit is full-length,
    the remapped ``X<mature_pos>Y`` string when a mature region applies, or
    None when the site lies in a cleaved region (not present in the mature
    protein — callers exclude these from the worklist).
    """
    if not mut or mut.upper() == "WT":
        return mut
    from .sequence import resolve_mature_regions, to_mature_pos, parse_mutation_pos
    uids = list(getattr(config, "uniprot_ids", []) or [])
    if subunit_idx >= len(uids):
        return mut
    region = resolve_mature_regions(config).get(uids[subunit_idx])
    pos = parse_mutation_pos(mut)
    if pos is None:
        return mut
    mpos = to_mature_pos(region, pos)
    if mpos is None:
        return None
    return f"{mut[0]}{mpos}{mut[-1]}"


def _build_yaml_spec(config, mutated_idx: int, mut: str) -> dict:
    """Build a Boltz-2 YAML spec (as a dict) for one proteoform.

    ``mutated_idx`` is the index into config.uniprot_ids whose subunit carries
    ``mut``; all other subunits stay WT. Chains are expanded per stoichiometry.

    Sequences are the **mature** chains (signal/pro-peptides removed when a
    mature region is configured or auto-detected), and the mutation position
    is remapped to mature-chain numbering, so the folded assembly matches the
    biologically processed protein (e.g. the TTR tetramer without the 20-aa
    signal peptide on each monomer).
    """
    from .sequence import get_mature_sequences
    sequences = (get_mature_sequences(config)
                 or getattr(config, "_sequences", None) or [])
    stoich = config.subunit_stoichiometry or [config.n_subunits]
    mut_mature = _remap_mutation_to_mature(config, mutated_idx, mut)
    if mut_mature is None:
        # Cleaved-region mutation (normally filtered from the worklist):
        # fold the WT mature sequence rather than a wrong position.
        mut_mature = "WT"

    # One chain entry per copy, assigning sequential chain ids.
    chains = []
    chain_letters = _chain_ids(config.total_chains if not config.is_monomer else 1)
    ci = 0
    if config.is_monomer:
        seq = sequences[mutated_idx] if mutated_idx < len(sequences) else sequences[0]
        seq = apply_mutation(seq, mut_mature)
        chains.append({"protein": {"id": chain_letters[0], "sequence": seq}})
    else:
        for j, seq in enumerate(sequences):
            use_seq = apply_mutation(seq, mut_mature) if j == mutated_idx else seq
            copies = stoich[j] if j < len(stoich) else 1
            for _ in range(copies):
                if ci >= len(chain_letters):
                    break
                chains.append({"protein": {"id": chain_letters[ci], "sequence": use_seq}})
                ci += 1
    return {"version": 1, "sequences": chains}


def _write_yaml(spec: dict, path: str) -> str:
    import yaml
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
    return path


# ---------------------------------------------------------------------------
# Output naming (canonical, matches old local_pdb path)
# ---------------------------------------------------------------------------

def _canonical_pdb_path(config, paths, uid: str, mut: str) -> str:
    if config.is_monomer:
        d = paths["pdbs_monomer"]
        os.makedirs(d, exist_ok=True)
        if mut.upper() == "WT":
            return os.path.join(d, f"wt-{uid}-monomer.pdb")
        return os.path.join(d, f"Mut_{uid}_{mut}-monomer.pdb")
    d = paths["pdbs"]
    os.makedirs(d, exist_ok=True)
    if mut.upper() == "WT":
        return os.path.join(d, f"wt-{uid}-tetramer.pdb")
    return os.path.join(d, f"Mut_{uid}_{mut}-tetramer.pdb")


def _job_label(config, uid: str, mut: str) -> str:
    tag = "monomer" if config.is_monomer else "assembly"
    return f"{('wt-'+uid if mut.upper()=='WT' else 'Mut_'+uid+'_'+mut)}-{tag}"


# ---------------------------------------------------------------------------
# Local boltz binary path
# ---------------------------------------------------------------------------

def _resolve_local_binary(config) -> str | None:
    b = config.boltz2.local_binary
    if b and (os.path.exists(b) or shutil.which(b)):
        return b
    if config.boltz2.prefer_local:
        found = shutil.which("boltz")
        if found:
            return found
    return None


def _run_local_boltz(binary, yaml_path, out_dir, config) -> str | None:
    """Run boltz locally (synchronous). Returns the predicted mmCIF path or None."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [binary, "predict", yaml_path, "--out_dir", out_dir,
           "--cache", config.boltz2.cache_dir, "--num_workers", "0", "--no_kernels"]
    if config.boltz2.use_msa_server:
        cmd.append("--use_msa_server")
    cmd += list(config.boltz2.extra_flags)
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    log.info("Local boltz: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, env=env, timeout=3600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log.warning("Local boltz run failed: %s", e)
        return None
    return _find_boltz_cif(out_dir)


def _find_boltz_cif(out_dir: str) -> str | None:
    """Find the predicted mmCIF (``*_model_0.cif``) in a boltz output tree."""
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.endswith("_model_0.cif") or f.endswith(".cif"):
                return os.path.join(root, f)
    return None


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _fold_api(config, paths, worklist) -> StepResult:
    """API folding backend: fold WT + each mutant via the hosted Boltz API.

    On an auth failure (401 / invalid key), logs ONE warning and falls back to
    the local boltz binary or PTM-Psi graft for ALL structures (never surfaces
    the 401 text as a step error). On non-auth per-structure failures, logs a
    warning per structure and continues; if all fail, falls back too.
    """
    from . import _boltz_backend as bb

    outputs = []
    n_done = 0
    errors = []
    api_dir = os.path.join(paths["boltz_structures"], "api")
    auth_failed = False

    for item in worklist:
        if len(item) == 2:
            uid, mut = item
            mutated_idx = 0
        else:
            uid, mut, mutated_idx = item
        pdb_out = _canonical_pdb_path(config, paths, uid, mut)
        if os.path.exists(pdb_out):
            outputs.append(pdb_out)
            n_done += 1
            continue
        label = _job_label(config, uid, mut)
        spec = _build_yaml_spec(config, mutated_idx, mut)
        out_dir = os.path.join(api_dir, label)
        try:
            # Pass a unique name per structure so the SDK never refuses with
            # "belongs to a different request".
            cif = bb.run_api_fold(config, spec, out_dir, name=label)
            cif_to_pdb(cif, pdb_out)
            outputs.append(pdb_out)
            n_done += 1
            log.info("Boltz API folded %s", os.path.basename(pdb_out))
        except Exception as e:
            if bb.is_auth_error(e):
                auth_failed = True
                log.warning("Boltz API key invalid or unauthorized; falling back to "
                            "local boltz / graft for remaining structures.")
                break
            log.error("Boltz API fold failed for %s: %s", label, e)
            errors.append(f"{label}: {e}")

    # --- Auth failure: fall back to local/graft for the ENTIRE worklist -----
    if auth_failed:
        return _fold_fallback_after_api(config, paths, worklist, outputs, n_done)

    if n_done:
        bb.write_structure_provenance(paths, "boltz_api", backbone_identical=False,
                                      detail=f"Hosted Boltz API ({config.boltz2.api_model})")
        status = "ok"
        msg = f"Boltz API prepared {n_done} structures"
        if errors:
            msg += f" ({len(errors)} failed)"
    else:
        # All structures failed with non-auth errors: try local/graft too.
        if errors:
            log.warning("Boltz API failed for all structures; attempting local/graft fallback.")
            return _fold_fallback_after_api(config, paths, worklist, outputs, n_done)
        status = "skipped"
        msg = "Boltz API produced no structures"
    return StepResult("structure", status, msg, outputs=outputs)


def _fold_fallback_after_api(config, paths, worklist, outputs, n_done) -> StepResult:
    """Fall back to local boltz or graft after an API failure (auth or otherwise).

    Re-resolves the backend EXCLUDING the API tier. If a local ``boltz`` binary
    is available, folds the remaining structures locally; otherwise uses the
    PTM-Psi graft fallback (with GRAFT_WARNING). Never surfaces API error text.
    """
    from . import _boltz_backend as bb

    local_bin = _resolve_local_binary(config)
    if local_bin:
        return _fold_local_fallback(config, paths, worklist, outputs, n_done, local_bin)
    # No local boltz: try graft (folding-only last resort).
    if bb.graft_available(config):
        log.info("Falling back to PTM-Psi graft (no local boltz, API unavailable).")
        return _fold_graft(config, paths, worklist)
    # Nothing available at all: clean skip with actionable message (no 401 text).
    hint = ("Boltz API unavailable and no local 'boltz' binary or graft fallback "
            "is configured. Provide a valid Boltz API key (boltz2.api_key or "
            "$BOLTZ_API_KEY), install a local 'boltz' binary "
            "(boltz2.prefer_local=True), or set boltz2.allow_graft_fallback=True "
            "for an offline backbone-identical graft.")
    return StepResult("structure", "skipped",
                      f"Boltz-2 folding has no available backend. {hint}")


def _fold_local_fallback(config, paths, worklist, outputs, n_done, local_bin) -> StepResult:
    """Fold remaining (not-yet-built) structures with a local boltz binary."""
    from . import _boltz_backend as bb
    yaml_dir = os.path.join(paths["boltz_structures"], "yaml")
    os.makedirs(yaml_dir, exist_ok=True)

    for item in worklist:
        if len(item) == 2:
            uid, mut = item
            mutated_idx = 0
        else:
            uid, mut, mutated_idx = item
        pdb_out = _canonical_pdb_path(config, paths, uid, mut)
        if os.path.exists(pdb_out):
            if pdb_out not in outputs:
                outputs.append(pdb_out)
            continue
        label = _job_label(config, uid, mut)
        spec = _build_yaml_spec(config, mutated_idx, mut)
        yaml_path = _write_yaml(spec, os.path.join(yaml_dir, f"{label}.yaml"))
        out_dir = os.path.join(paths["boltz_structures"], "local", label)
        cif = _run_local_boltz(local_bin, yaml_path, out_dir, config)
        if cif:
            try:
                cif_to_pdb(cif, pdb_out)
                outputs.append(pdb_out)
                n_done += 1
                log.info("Boltz-2 (local fallback) built %s", os.path.basename(pdb_out))
            except Exception as e:
                log.error("CIF->PDB failed for %s: %s", label, e)
        else:
            log.error("Local boltz produced no structure for %s", label)

    if n_done:
        bb.write_structure_provenance(paths, "boltz_local", backbone_identical=False,
                                      detail="Local boltz binary (API fallback)")
        status = "ok"
        msg = (f"Boltz-2 (local fallback after API failure) prepared "
               f"{n_done} structures")
    else:
        status = "skipped"
        msg = "Boltz-2 local fallback produced no structures"
    return StepResult("structure", status, msg, outputs=outputs)


def _fold_graft(config, paths, worklist) -> StepResult:
    """Graft fallback backend: reference-seed WT + PTM-Psi mutant grafting.

    Backbone-identical by construction (TM=1.0). Records provenance and returns
    the GRAFT_WARNING so the caller/GUI can surface it prominently.
    """
    from . import _boltz_backend as bb
    from . import _graft_seed

    res = _graft_seed.build_graft_structures(
        config, paths, worklist, _canonical_pdb_path)
    outputs = list(res["outputs"])
    n_done = res["n_done"]

    if n_done:
        bb.write_structure_provenance(
            paths, "graft", backbone_identical=True,
            detail=f"PTM-Psi side-chain graft onto {config.local_pdb_id} backbone. "
                   + bb.GRAFT_WARNING)
        status = "ok"
        msg = ("Graft fallback: prepared "
               f"{n_done} backbone-identical structures (WT seed {config.local_pdb_id} "
               f"+ PTM-Psi mutants). " + bb.GRAFT_WARNING)
        if res["errors"]:
            msg += f" [{len(res['errors'])} graft issue(s)]"
    else:
        status = "skipped"
        detail = res["errors"][0] if res["errors"] else "no structures produced"
        msg = f"Graft fallback produced no structures: {detail}"
    return StepResult("structure", status, msg, outputs=outputs)


def _fold_local(config, paths, worklist) -> StepResult:
    """Local folding backend: fold WT + each mutant with a local ``boltz`` binary.

    Runs **synchronously** (one subprocess per structure). If the local binary
    cannot be resolved, or produces no structures, falls back to graft (if
    enabled) or skips cleanly. Never submits to any external cluster.
    """
    from . import _boltz_backend as bb

    local_bin = _resolve_local_binary(config)
    if not local_bin:
        # Resolver said "local" but the binary is not actually runnable here.
        if bb.graft_available(config):
            log.info("Local 'boltz' binary not found; using PTM-Psi graft fallback.")
            return _fold_graft(config, paths, worklist)
        hint = ("The local backend was selected but no runnable 'boltz' binary "
                "was found. Set boltz2.local_binary to its path, put 'boltz' on "
                "PATH with boltz2.prefer_local=True, provide a Boltz API key "
                "(boltz2.api_key or $BOLTZ_API_KEY), or set "
                "boltz2.allow_graft_fallback=True for an offline backbone-identical graft.")
        return StepResult("structure", "skipped",
                          f"Boltz-2 folding has no runnable local backend. {hint}")

    yaml_dir = os.path.join(paths["boltz_structures"], "yaml")
    os.makedirs(yaml_dir, exist_ok=True)
    outputs: list[str] = []
    n_done = 0

    for item in worklist:
        if len(item) == 2:
            uid, mut = item
            mutated_idx = 0
        else:
            uid, mut, mutated_idx = item

        pdb_out = _canonical_pdb_path(config, paths, uid, mut)
        if os.path.exists(pdb_out):
            if pdb_out not in outputs:
                outputs.append(pdb_out)
            n_done += 1
            continue

        label = _job_label(config, uid, mut)
        spec = _build_yaml_spec(config, mutated_idx, mut)
        yaml_path = _write_yaml(spec, os.path.join(yaml_dir, f"{label}.yaml"))
        #outputs.append(yaml_path)

        out_dir = os.path.join(paths["boltz_structures"], "local", label)
        cif = _run_local_boltz(local_bin, yaml_path, out_dir, config)
        if cif:
            try:
                cif_to_pdb(cif, pdb_out)
                outputs.append(pdb_out)
                n_done += 1
                log.info("Boltz-2 (local) built %s", os.path.basename(pdb_out))
            except Exception as e:
                log.error("CIF->PDB failed for %s: %s", label, e)
        else:
            log.error("Local boltz produced no structure for %s", label)

    if n_done:
        bb.write_structure_provenance(paths, "boltz_local", backbone_identical=False,
                                      detail="Local boltz binary")
        return StepResult("structure", "ok",
                          f"Boltz-2 (local) prepared {n_done} structures",
                          outputs=outputs)
    # Local binary ran but produced nothing: try graft, else skip.
    if bb.graft_available(config):
        log.warning("Local boltz produced no structures; using PTM-Psi graft fallback.")
        return _fold_graft(config, paths, worklist)
    return StepResult("structure", "skipped",
                      "Boltz-2 (local) produced no structures and no graft fallback "
                      "is available.", outputs=outputs)


def build_structures(config, paths: dict) -> StepResult:
    """Fold WT + mutant structures with Boltz-2.

    Backend is chosen by ``_boltz_backend.resolve_backend(config, 'fold')``:
    hosted API (key) -> local ``boltz`` binary (synchronous) -> PTM-Psi graft
    fallback (if ``boltz2.allow_graft_fallback``) -> clean skip. ``prefer_local``
    flips the API/local tier order. The Biomni HPC backend was removed in v3.4.0.
    """
    from . import _boltz_backend as bb

    sequences = getattr(config, "_sequences", None)
    mutation_lists = getattr(config, "_mutation_lists", None)
    if not sequences:
        return StepResult("structure", "skipped",
                          "No sequences available (run 'sequence' step first)")

    yaml_dir = os.path.join(paths["boltz_structures"], "yaml")
    os.makedirs(yaml_dir, exist_ok=True)

    # Resolve mature-chain regions once (manual config > sequence-step stash >
    # persisted report) and stash the mature sequences for the YAML builder.
    from .sequence import (get_mature_sequences, resolve_mature_regions,
                           parse_mutation_pos, is_cleaved_site)
    config._mature_regions = resolve_mature_regions(config, paths)
    mature_seqs = get_mature_sequences(config, paths)
    if mature_seqs:
        config._mature_sequences = mature_seqs

    # Build the (uid, mut) worklist: WT once (from subunit 0) + each mutant on its subunit.
    # Mutations in cleaved (non-mature) regions are excluded: the residue does
    # not exist in the mature protein that gets folded.
    worklist = [(config.uniprot_ids[0], "WT")]
    for idx, uid in enumerate(config.uniprot_ids):
        muts = mutation_lists[idx] if (mutation_lists and idx < len(mutation_lists)) else []
        region = config._mature_regions.get(uid)
        for mut in muts:
            if mut.upper() == "WT":
                continue
            pos = parse_mutation_pos(mut)
            if pos is not None and is_cleaved_site(region, pos):
                log.warning("Mutation %s (%s) lies in the cleaved region (mature "
                            "chain %d-%d); not folded — excluded from structural steps",
                            mut, uid, region[0], region[1])
                continue
            worklist.append((uid, mut, idx))

    backend = bb.resolve_backend(config, "fold")
    log.info("Boltz folding backend resolved to: %s", backend)

    # --- Dispatch on resolved backend -------------------------------------
    if backend == "api":
        return _fold_api(config, paths, worklist)
    if backend == "local":
        return _fold_local(config, paths, worklist)
    if backend == "graft":
        return _fold_graft(config, paths, worklist)

    # backend == "none": no backend at all (graft disabled or unavailable).
    hint = ("Provide a Boltz API key (boltz2.api_key or $BOLTZ_API_KEY), "
            "install a local 'boltz' binary (boltz2.prefer_local=True), or set "
            "boltz2.allow_graft_fallback=True for an offline backbone-identical "
            "graft (TM=1.0, no structural signal).")
    return StepResult("structure", "skipped",
                      f"Boltz-2 folding has no available backend. {hint}")
