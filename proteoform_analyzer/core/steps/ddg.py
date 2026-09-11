"""Step: DeltaDeltaG stability prediction (local ThermoMPNN + ESM2 zero-shot fallback).

Tier 1: ThermoMPNN — GPU-based graph neural network that predicts ddG from PDB
        structure. Run **locally** via a configured ThermoMPNN checkout
        (``config.thermompnn_dir``/``thermompnn_script`` + ``thermompnn_checkpoint``).
        Output is a CSV with all 19 point mutations per position; we filter to
        the configured mutations. GPU strongly recommended.

Tier 2: ESM2 zero-shot — log-likelihood ratio between WT and mutant sequences.
        Runs locally on CPU. Measures sequence fitness (not thermodynamic ddG)
        but correlates with stability and requires no GPU.

Tier 3: FoldX PTM mimetics — PTMs are scored as their accepted mimetic
        substitutions (phospho SER/THR->GLU, acetyl LYS->GLN) with a
        user-provided FoldX binary (``config.foldx_binary``), on the WT
        background and on each mutation × PTM proteoform. Optional: skipped
        gracefully when no FoldX binary is configured. See ``_foldx.py``.

If a local ThermoMPNN install is not configured (or the run fails), the step
automatically falls back to ESM2 zero-shot scores. As of v3.4.0 the Biomni HPC
backend has been removed — ThermoMPNN now runs locally or not at all.
"""
from __future__ import annotations

import os
import shutil
import logging
import subprocess
import pandas as pd
import numpy as np

from ..pipeline import StepResult

log = logging.getLogger("proteoform_analyzer.ddg")


# ---------------------------------------------------------------------------
# Tier 1: ThermoMPNN (local binary/script, GPU recommended)
# ---------------------------------------------------------------------------

def _resolve_thermompnn(config):
    """Resolve the local ThermoMPNN inference script + checkpoint + interpreter.

    Returns ``(script_path, checkpoint_path, python_exe)`` if a runnable local
    install is configured, else ``None``. Side-effect free.

    Resolution:
      - script: ``config.thermompnn_script`` if set, else
        ``<config.thermompnn_dir>/custom_inference.py``.
      - checkpoint: ``config.thermompnn_checkpoint`` (required).
      - python: ``config.thermompnn_python`` if set, else the current interpreter.
    """
    import sys as _sys

    script = getattr(config, "thermompnn_script", None)
    tdir = getattr(config, "thermompnn_dir", None)
    if not script and tdir:
        script = os.path.join(tdir, "custom_inference.py")
    ckpt = getattr(config, "thermompnn_checkpoint", None)

    if not script or not os.path.isfile(script):
        return None
    if not ckpt or not os.path.isfile(ckpt):
        log.info("ThermoMPNN script found but checkpoint missing/unset "
                 "(config.thermompnn_checkpoint); using ESM2 fallback.")
        return None

    py = getattr(config, "thermompnn_python", None)
    if py:
        if not (os.path.isfile(py) or shutil.which(py)):
            log.warning("Configured thermompnn_python '%s' not found; using %s",
                        py, _sys.executable)
            py = _sys.executable
    else:
        py = _sys.executable
    return script, ckpt, py


def _thermompnn_for_chain_local(script, checkpoint, python_exe,
                                wt_pdb_path: str, chain: str, out_dir: str) -> str | None:
    """Run the local ThermoMPNN inference script for one chain (synchronous).

    Returns the per-chain output dir on success, else None. Mirrors the CLI the
    ThermoMPNN ``custom_inference.py`` exposes:

        python custom_inference.py --pdb <pdb> --chain <chain>
            --model_path <ckpt> --out_dir <out_dir>
    """
    chain_out = os.path.join(out_dir, f"chain_{chain}")
    os.makedirs(chain_out, exist_ok=True)
    cmd = [
        python_exe, script,
        "--pdb", wt_pdb_path,
        "--chain", chain,
        "--model_path", checkpoint,
        "--out_dir", chain_out,
    ]
    log.info("Local ThermoMPNN (chain %s): %s", chain, " ".join(cmd))
    try:
        subprocess.run(cmd, timeout=3600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log.warning("Local ThermoMPNN run failed (chain %s): %s", chain, e)
        return None
    return chain_out


def _parse_thermompnn_results(output_dir: str, chain: str,
                              wanted_mutations: list[str]) -> list[dict]:
    """Parse ThermoMPNN output CSV and filter to wanted mutations.

    ThermoMPNN output columns: Model, Dataset, ddG_pred, position, wildtype, mutation
    Mutation format in output: single-letter AA (e.g. 'A' for alanine).
    We reconstruct 'XnnY' format from wildtype/position/mutation columns.
    """
    rows = []
    csv_path = None
    candidates = []

    for root, _, files in os.walk(output_dir):
        for filename in files:
            if filename.endswith(".csv"):
                candidates.append(os.path.join(root, filename))

    if candidates:
        # Preferisce il formato prodotto da custom_inference.py
        preferred = [
            p for p in candidates
            if os.path.basename(p).startswith("ThermoMPNN_inference_")
        ]
        csv_path = sorted(preferred or candidates)[0]

    df = pd.read_csv(csv_path)
    wanted_set = set(wanted_mutations)

    for _, row in df.iterrows():
        wt_aa = str(row.get("wildtype", "")).strip()
        raw_pos = int(row.get("position", 0))
        mut_aa = str(row.get("mutation", "")).strip()
        ddg = row.get("ddG_pred")

        pos = raw_pos + 1  
        if not wt_aa or not mut_aa or pos == 0 or ddg is None:
            continue
        mut_str = f"{wt_aa}{pos}{mut_aa}"
        if mut_str in wanted_set:
            try:
                rows.append({
                    "mutation": mut_str,
                    "ddg_kcal_mol": float(ddg),
                    "method": "thermompnn",
                })
            except (ValueError, TypeError):
                continue
    return rows


def _run_thermompnn(config, wt_pdb_path: str, chains: list[str],
                    wanted_mutations: list[str]) -> list[dict] | None:
    """Run local ThermoMPNN for the given chains and return filtered ddG rows.

    Returns None if no local ThermoMPNN install is configured or the run
    produces no matching mutations (caller falls back to ESM2). Returns a list
    of dicts with mutation/ddg_kcal_mol/method keys.
    """
    resolved = _resolve_thermompnn(config)
    if resolved is None:
        log.info("No local ThermoMPNN install configured; using ESM2 fallback.")
        return None
    script, ckpt, py = resolved

    base_out = os.path.join(os.path.dirname(os.path.abspath(wt_pdb_path)),
                            "thermompnn_local")
    os.makedirs(base_out, exist_ok=True)

    all_rows = []
    for chain in chains:
        chain_out = _thermompnn_for_chain_local(script, ckpt, py,
                                                wt_pdb_path, chain, base_out)
        if not chain_out:
            continue
        rows = _parse_thermompnn_results(chain_out, chain, wanted_mutations)
        all_rows.extend(rows)

    if not all_rows:
        log.warning("Local ThermoMPNN produced no matching mutations")
        return None
    return all_rows


# ---------------------------------------------------------------------------
# Tier 2: ESM2 zero-shot (local CPU fallback)
# ---------------------------------------------------------------------------

def _esm2_zeroshot_ddg(wt_seq: str, mutations: list[str],
                       model_name: str = "esm2_t33_650M_UR50D") -> dict[str, float]:
    """Compute ESM2 log-likelihood ratio for each mutation.

    Returns a dict {mutation: ddg_score} where positive = destabilizing
    (following the convention that higher ddG = less stable).

    The score is the negative log-likelihood ratio: -[log P(mut) - log P(wt)]
    at the mutated position. A mutation that the model finds less likely
    (lower P(mut) than P(wt)) gets a positive score (destabilizing).
    """
    try:
        import torch
        import esm
    except ImportError as e:
        log.warning("fair-esm not available for zero-shot ddG: %s", e)
        return {}

    from ._device import torch_device
    # CPU by default; CUDA on a ZeroGPU Space when a GPU is actually allocated
    # (the Spaces app.py wraps this function with @spaces.GPU).
    device = torch_device()

    log.info("Loading ESM2 model %s for zero-shot ddG (device=%s)...", model_name, device)
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # Score WT sequence once
    _, _, wt_tokens = batch_converter([("wt", wt_seq)])
    wt_tokens = wt_tokens.to(device)
    with torch.no_grad():
        wt_logits = model(wt_tokens, repr_layers=[], return_contacts=False)["logits"]

    scores = {}
    for mut in mutations:
        if mut.upper() == "WT":
            continue
        try:
            pos = int(mut[1:-1]) - 1  # 0-indexed
            new_aa = mut[-1]
            if pos < 0 or pos >= len(wt_seq):
                continue
            wt_aa = mut[0]
            if wt_seq[pos] != wt_aa:
                log.warning("WT aa mismatch at pos %d: seq=%s, mut=%s", pos + 1, wt_seq[pos], wt_aa)
                continue

            # Get logit at the mutated position for both WT and mutant AA
            wt_aa_idx = alphabet.get_idx(wt_aa)
            new_aa_idx = alphabet.get_idx(new_aa)

            # Log-likelihood ratio: log P(new_aa | context) - log P(wt_aa | context)
            # at the mutated position (offset +1 for BOS token)
            wt_logit = wt_logits[0, pos + 1, wt_aa_idx].item()
            new_logit = wt_logits[0, pos + 1, new_aa_idx].item()

            # Negative LLR: positive = mutant less likely = destabilizing
            llr = new_logit - wt_logit
            scores[mut] = -llr
        except (ValueError, IndexError) as e:
            log.warning("Failed to score mutation %s: %s", mut, e)
            continue

    return scores


def _run_esm2_zeroshot(sequences: list[str], mutation_lists: list[str],
                       uniprot_ids: list[str]) -> list[dict]:
    """Run ESM2 zero-shot ddG for all mutations across all subunits."""
    all_rows = []
    for idx, (uid, seq) in enumerate(zip(uniprot_ids, sequences)):
        muts = mutation_lists[idx] if idx < len(mutation_lists) else []
        muts = [m for m in muts if m.upper() != "WT"]
        if not muts:
            continue
        log.info("ESM2 zero-shot ddG for %s: %d mutations", uid, len(muts))
        scores = _esm2_zeroshot_ddg(seq, muts)
        for mut, score in scores.items():
            all_rows.append({
                "uniprot_id": uid,
                "mutation": mut,
                "ddg_kcal_mol": round(score, 4),
                "method": "esm2_zeroshot",
            })
    return all_rows


# ---------------------------------------------------------------------------
# Main step function
# ---------------------------------------------------------------------------

def run_ddg(config, paths: dict) -> StepResult:
    """Run ΔΔG prediction for all mutations.

    Tries a locally-configured ThermoMPNN (GPU recommended) first; falls back to
    the CPU ESM2 zero-shot scorer if no local ThermoMPNN is configured or it fails.
    """
    out_dir = paths["ddg"]
    os.makedirs(out_dir, exist_ok=True)

    mutation_lists = getattr(config, "_mutation_lists", None)
    sequences = getattr(config, "_sequences", None)
    if not mutation_lists:
        mutation_lists = []
        for uid in config.uniprot_ids:
            mf = os.path.join(paths["results"], f"{uid}_mutations.txt")
            if os.path.exists(mf):
                with open(mf) as f:
                    mutation_lists.append([l.strip() for l in f if l.strip()])
            else:
                mutation_lists.append([])
        config._mutation_lists = mutation_lists
    if not sequences:
        from .sequence import read_fasta
        sequences = []
        for uid in config.uniprot_ids:
            fp = os.path.join(paths["input"], f"{uid}.fasta")
            if os.path.exists(fp):
                sequences.append(read_fasta(fp))
            else:
                sequences.append("")
        config._sequences = sequences

    # Mature-chain regions are needed by both the mutation tiers (remapping)
    # and the FoldX PTM tier, so resolve them up front.
    from .sequence import (get_mature_sequences, resolve_mature_regions,
                           to_mature_pos, parse_mutation_pos)
    mature_regions = resolve_mature_regions(config, paths)

    def _find_wt_pdb() -> str | None:
        """Locate the WT PDB (monomer preferred — single chain).

        Structures resolve through boltz-experiments first (see
        ``_structure_source``).
        """
        from ._structure_source import resolve_structure_pdb
        for d_key in ("pdbs_monomer", "pdbs"):
            d = paths.get(d_key, "")
            if d and os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith(".pdb") and f.lower().startswith("wt"):
                        return (resolve_structure_pdb(paths, f[:-4], "ddg")
                                or os.path.join(d, f))
        return None

    def _run_ptm_tier(wt_pdb_path: str | None):
        """Tier 3: FoldX PTM-mimetic ddG (PTMs only). Returns (rows, note)."""
        if not getattr(config, "run_ptm", False) or not wt_pdb_path:
            return [], ""
        try:
            from ._foldx import run_foldx_ptm_ddg
            return run_foldx_ptm_ddg(config, paths, wt_pdb_path,
                                     mutation_lists, mature_regions)
        except Exception as e:
            log.warning("FoldX PTM ddG tier failed: %s", e)
            return [], f"FoldX PTM ddG failed: {e}"

    def _finalize(mut_rows, ptm_rows, ptm_note, method_used) -> StepResult | None:
        """Write ddg_summary.csv and build the StepResult (None if no rows)."""
        all_rows = list(mut_rows or []) + list(ptm_rows or [])
        if not all_rows:
            return None
        df = pd.DataFrame(all_rows)
        # Consistent column order; mutation rows leave the PTM columns empty
        cols = ["uniprot_id", "mutation", "ptm", "mimetic_mutation",
                "structure", "ddg_kcal_mol", "method"]
        df = df[[c for c in cols if c in df.columns]]
        csv = os.path.join(out_dir, "ddg_summary.csv")
        df.to_csv(csv, index=False)
        n_ok = len(df[df["ddg_kcal_mol"].notna()])
        parts = []
        if mut_rows:
            parts.append(f"{len(mut_rows)} mutations via {method_used}")
        if ptm_rows:
            parts.append(f"{len(ptm_rows)} PTM/proteoform mimetics via foldx_mimetic")
        msg = (f"ΔΔG predictions: {n_ok}/{len(df)} entries scored ("
               + "; ".join(parts) + ")")
        if ptm_note and not ptm_rows:
            msg += f". {ptm_note}"
        return StepResult("ddg", "ok", msg, outputs=[csv], data=df)

    if not mutation_lists or all(len(m) <= 1 for m in mutation_lists):
        # No mutations — PTM-only ddG (FoldX mimetic tier) may still apply.
        ptm_rows, ptm_note = _run_ptm_tier(_find_wt_pdb())
        res = _finalize([], ptm_rows, ptm_note, "foldx_mimetic")
        return res or StepResult("ddg", "skipped", "No mutations available")

    # Collect all wanted mutations (across all subunits)
    all_wanted = []
    for idx, uid in enumerate(config.uniprot_ids):
        muts = mutation_lists[idx] if idx < len(mutation_lists) else []
        for m in muts:
            if m.upper() != "WT":
                all_wanted.append(m)
    if not all_wanted:
        ptm_rows, ptm_note = _run_ptm_tier(_find_wt_pdb())
        res = _finalize([], ptm_rows, ptm_note, "foldx_mimetic")
        return res or StepResult("ddg", "skipped", "No non-WT mutations found")

    # Mature-chain handling: folded structures and mature sequences use mature
    # numbering, so both tiers score mature-remapped mutations; results are
    # translated back to UniProt labels for the output table. Mutations in
    # proteolytically cleaved regions are excluded (absent from the mature
    # protein) with an explicit warning.
    wanted_backmap: dict[str, str] = {}  # mature label -> original UniProt label
    all_wanted_mature: list[str] = []
    for idx, uid in enumerate(config.uniprot_ids):
        region = mature_regions.get(uid)
        muts = mutation_lists[idx] if idx < len(mutation_lists) else []
        for m in muts:
            if m.upper() == "WT":
                continue
            pos = parse_mutation_pos(m)
            if pos is not None and region is not None:
                mpos = to_mature_pos(region, pos)
                if mpos is None:
                    log.warning(
                        "ddg: mutation %s (%s) lies in the proteolytically "
                        "cleaved region (mature chain %d-%d); excluded from ddG",
                        m, uid, region[0], region[1])
                    continue
                mm = f"{m[0]}{mpos}{m[-1]}"
                wanted_backmap[mm] = m
                all_wanted_mature.append(mm)
            else:
                all_wanted_mature.append(m)
    if not all_wanted_mature:
        ptm_rows, ptm_note = _run_ptm_tier(_find_wt_pdb())
        res = _finalize([], ptm_rows, ptm_note, "foldx_mimetic")
        return res or StepResult(
            "ddg", "skipped",
            "All mutations lie in cleaved regions - nothing to score")

    # Find WT PDB (monomer preferred for ThermoMPNN — single chain).
    wt_pdb = _find_wt_pdb()

    # ── Tier 1: ThermoMPNN (local, GPU recommended) ──
    rows = None
    method_used = "thermompnn"
    if wt_pdb:
        log.info("Attempting local ThermoMPNN ddG prediction...")
        # For homo-oligomers, chain A is sufficient (all chains identical)
        # For hetero-oligomers, we'd need separate runs per subunit type
        chains_to_analyze = ["A"]
        if config.is_hetero:
            chains_to_analyze = [chr(65 + i) for i in range(config.n_unique_subunits)]

        try:
            rows = _run_thermompnn(config, wt_pdb, chains_to_analyze, all_wanted_mature)
        except Exception as e:
            log.warning("Local ThermoMPNN failed: %s", e)
            rows = None

        if rows:
            # Translate mature-numbered labels back to UniProt numbering
            for r in rows:
                r["mutation"] = wanted_backmap.get(r["mutation"], r["mutation"])
            # Add uniprot_id to each row
            for r in rows:
                # Match mutation to subunit
                for idx, uid in enumerate(config.uniprot_ids):
                    muts = mutation_lists[idx] if idx < len(mutation_lists) else []
                    if r["mutation"] in muts:
                        r["uniprot_id"] = uid
                        break
                else:
                    r["uniprot_id"] = config.uniprot_ids[0]

    # ── Tier 2: ESM2 zero-shot fallback ──
    if not rows:
        method_used = "esm2_zeroshot"
        log.info("Falling back to ESM2 zero-shot ddG prediction...")
        # Score the mature sequences with mature-numbered mutations.
        esm_seqs = get_mature_sequences(config, paths) or sequences
        esm_mut_lists: list[list[str]] = []
        for idx, uid in enumerate(config.uniprot_ids):
            region = mature_regions.get(uid)
            muts = mutation_lists[idx] if idx < len(mutation_lists) else []
            remapped: list[str] = []
            for m in muts:
                pos = parse_mutation_pos(m)
                if pos is not None and region is not None:
                    mpos = to_mature_pos(region, pos)
                    if mpos is None:
                        continue  # cleaved site; already warned above
                    m = f"{m[0]}{mpos}{m[-1]}"
                remapped.append(m)
            esm_mut_lists.append(remapped)
        rows = _run_esm2_zeroshot(esm_seqs, esm_mut_lists, config.uniprot_ids)
        for r in rows:
            r["mutation"] = wanted_backmap.get(r["mutation"], r["mutation"])

    # ── Tier 3: FoldX PTM-mimetic ddG (PTMs only; optional) ──
    ptm_rows, ptm_note = _run_ptm_tier(wt_pdb)

    res = _finalize(rows, ptm_rows, ptm_note, method_used)
    if res is None:
        return StepResult("ddg", "skipped",
                          "Both ThermoMPNN and ESM2 zero-shot failed — no ddG predictions")
    return res
