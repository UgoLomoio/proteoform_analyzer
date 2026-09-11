"""Gradio web GUI for the Proteoform Analyzer.

Three tabs (Setup / Run / Results) built with gradio.Blocks so it can be deployed
as a Hugging Face Space / webapp with no refactor.  Shares the same
``run_analysis(config)`` entry point as the CLI.

The Results tab is organized into subsections per result type, each with a
searchable dataframe, plus an embedded 3Dmol.js PDB viewer, MD trajectory overlay,
and interactive PCN centrality/community visualization.
"""
from __future__ import annotations

import html
import os
import sys
import io
import ast
import json
import re
import time
import base64
import logging
import tempfile
import threading
import traceback

import gradio as gr
import pandas as pd

from .core.config import (
    AnalysisConfig, EngineChoice, ProteoformMode, BindingSiteMethod,
    PTMConfig, MDConfig, Boltz2Config, AntibodyConfig, HotspotSource,
    hemoglobin_fast_config, ttr_fast_config, p53_fast_config,
    PRESETS,
)
from .core.pipeline import run_analysis, STEP_REGISTRY
from .core import viz
from .core.steps.sequence import _fetch_mutations_from_uniprot

log = logging.getLogger("proteoform_analyzer.gui")


def _html_iframe(html_path, height=560):
    """Embed a saved interactive Plotly .html file in a sandboxed iframe."""
    if not html_path or not os.path.exists(html_path):
        return "<p>No plot available yet. Run the pipeline and click Refresh.</p>"
    with open(html_path, "r", encoding="utf-8") as f:
        doc = f.read()
    escaped = html.escape(doc, quote=True)
    return (f'<iframe srcdoc="{escaped}" width="100%" height="{height}" '
            f'style="border:1px solid #ccc; border-radius:6px;" '
            f'sandbox="allow-scripts allow-same-origin"></iframe>')


# ---------------------------------------------------------------------------
# Guided PTM builder helpers
# ---------------------------------------------------------------------------

# Cache dir for FASTA / observed-PTM downloads used by the Setup-tab builder.
_PTM_GUI_CACHE = os.path.join(tempfile.gettempdir(), "proteoform_analyzer_ptm_gui")

PTM_PAIR_HEADERS = ["residue", "ptm_type", "uniprot_id"]


def _gui_fetch_sequence(uid: str) -> str:
    """Download (cached) FASTA sequence for a UniProt ID; '' on failure."""
    try:
        from .core.steps.sequence import _download_fasta, read_fasta
        path = _download_fasta(uid, _PTM_GUI_CACHE)
        return read_fasta(path)
    except Exception as e:
        log.warning("Could not fetch sequence for %s: %s", uid, e)
        return ""


def _list_modifiable_residues(uniprot_ids_text):
    """Populate the residue dropdown with PTM-modifiable residues.

    Only residues that can carry at least one ptmpsi-modelable PTM are listed
    (plus position 1, which can always be alpha-acetylated).  Dropdown values
    are ``"<uid>|<RES><pos>"``; labels are human-readable.
    """
    from .core.ptm_rules import is_modifiable_residue, one_to_three
    ids = [u.strip() for u in (uniprot_ids_text or "").split(",") if u.strip()]
    if not ids:
        return (gr.update(choices=[], value=None),
                "Enter at least one UniProt ID above, then click again.")
    choices, msgs = [], []
    for uid in ids:
        seq = _gui_fetch_sequence(uid)
        if not seq:
            msgs.append(f"{uid}: sequence download failed (check the ID / network)")
            continue
        n = 0
        for i, aa in enumerate(seq, start=1):
            aa3 = one_to_three(aa)
            if not aa3:
                continue
            if is_modifiable_residue(aa3) or i == 1:
                choices.append((f"{uid} · {aa3}{i}", f"{uid}|{aa3}{i}"))
                n += 1
        msgs.append(f"{uid}: {n} modifiable residues")
    if not choices:
        return gr.update(choices=[], value=None), " — ".join(msgs)
    return gr.update(choices=choices, value=choices[0][1]), " — ".join(msgs)


def _compatible_ptm_choices(residue_value):
    """Filter the PTM dropdown to PTMs compatible with the selected residue.

    PTMs experimentally observed in UniProt at that exact residue are tagged
    "(observed in UniProt)" and sorted first.
    """
    from .core.ptm_rules import compatible_ptms, parse_residue_spec
    if not residue_value or "|" not in str(residue_value):
        return gr.update(choices=[], value=None)
    uid, spec = str(residue_value).split("|", 1)
    parsed = parse_residue_spec(spec)
    if parsed is None:
        return gr.update(choices=[], value=None)
    aa3, pos = parsed
    compat = compatible_ptms(aa3, n_terminal=(pos == 1))
    observed: set[str] = set()
    try:
        from .core.steps.ptm import fetch_observed_ptms
        seq = _gui_fetch_sequence(uid)
        observed = {ptm for res, ptm in fetch_observed_ptms(uid, seq, _PTM_GUI_CACHE)
                    if res == f"{aa3}{pos}"}
    except Exception as e:
        log.info("Observed-PTM lookup failed for %s: %s", uid, e)
    ordered = sorted(compat, key=lambda p: (p not in observed, p))
    choices = [(f"{p}  (observed in UniProt)", p) if p in observed else (p, p)
               for p in ordered]
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def _ptm_pairs_to_records(ptm_pairs):
    """Normalize a gr.Dataframe value (DataFrame | list | None) to row lists."""
    if ptm_pairs is None:
        return []
    if isinstance(ptm_pairs, pd.DataFrame):
        return ptm_pairs.values.tolist()
    return [list(r) for r in ptm_pairs if r]


def _add_ptm_pair(residue_value, ptm_type, current):
    """Append the selected (residue, PTM) pair to the pairs table."""
    rows = _ptm_pairs_to_records(current)
    if residue_value and ptm_type and "|" in str(residue_value):
        uid, spec = str(residue_value).split("|", 1)
        row = [spec, str(ptm_type), uid]
        if row not in [[str(c) for c in r] for r in rows]:
            rows.append(row)
    return pd.DataFrame(rows, columns=PTM_PAIR_HEADERS)


def _clear_ptm_pairs():
    return pd.DataFrame(columns=PTM_PAIR_HEADERS)


def _parse_ptm_pairs(ptm_pairs) -> list:
    """Convert the pairs-table widget value to PTMConfig.pairs triples."""
    pairs = []
    for row in _ptm_pairs_to_records(ptm_pairs):
        cells = [str(c).strip() if c is not None else "" for c in row]
        if len(cells) >= 2 and cells[0] and cells[1]:
            pairs.append((cells[0], cells[1], cells[2] if len(cells) > 2 and cells[2] else None))
    return pairs


# ---------------------------------------------------------------------------
# Mature chain (proteolytic processing) helpers
# ---------------------------------------------------------------------------

MATURE_HEADERS = ["uniprot_id", "mature_start", "mature_end", "source", "warning"]


def _parse_mature_regions_text(text, uids) -> dict:
    """Parse the manual mature-regions textbox into ``{uid: [start, end]}``.

    Delegates to ``core.config.parse_mature_regions`` (shared with the CLI).
    Format: ``"P02766:21-147; P69905:2-142"``; a bare ``"21-147"`` is accepted
    only when a single UniProt ID is configured.
    """
    from .core.config import parse_mature_regions
    return parse_mature_regions(text, uids)


def _detect_mature_regions(uniprot_ids_text):
    """Preview UniProt-derived mature-chain regions for the Setup tab.

    Returns a Dataframe with one row per UniProt ID (region, detection source,
    and any warning, e.g. multi-chain proteins where only terminal signal /
    propeptide trimming is applied).
    """
    from .core.steps.sequence import fetch_mature_region
    ids = [u.strip() for u in (uniprot_ids_text or "").split(",") if u.strip()]
    if not ids:
        return pd.DataFrame(columns=MATURE_HEADERS)
    rows = []
    for uid in ids:
        try:
            res = fetch_mature_region(uid, cache_dir=_PTM_GUI_CACHE)
        except Exception as e:
            rows.append([uid, "", "", "", f"detection failed: {e}"])
            continue
        region = res.get("region")
        source = res.get("source", "")
        warning = res.get("warning") or ""
        if region:
            rows.append([uid, int(region[0]), int(region[1]), source, warning])
        else:
            rows.append([uid, "", "", source or "full-length", warning])
    return pd.DataFrame(rows, columns=MATURE_HEADERS)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_config(
    uniprot_ids, n_subunits, stoichiometry, proteoform_mode, proteoform_cap,
    mutations_text, max_mutations, structure_source, local_pdb_id,
    ptm_pairs, md_engine, md_steps,
    docking_engine, ligand_design_engine, binding_site_method,
    ensemble_docking, selected_steps,
    antibody_enabled=True, antibody_framework="nanobody",
    antibody_hotspot_source="bcell", antibody_hotspots="", antibody_num_designs=20,
    boltz_api_key="", boltz_prefer_local=True,
    boltz_allow_graft=True,
    mature_auto=True, mature_regions_text="",
    foldx_binary="",
):
    """Build an AnalysisConfig from GUI widget values."""
    ids = [u.strip() for u in uniprot_ids.split(",") if u.strip()]
    mutations = []
    if mutations_text.strip():
        for block in mutations_text.split("|"):
            mutations.append([m.strip() for m in block.split() if m.strip()])
    stoich = [int(x.strip()) for x in stoichiometry.split(",") if x.strip()] if stoichiometry.strip() else [int(n_subunits)]
    ptm_pair_list = _parse_ptm_pairs(ptm_pairs)
    steps = [s.strip() for s in selected_steps] if selected_steps else list(STEP_REGISTRY.keys())

    # Antibody (RFAntibody) — opt-in; add the 'antibody' step when enabled.
    hotspots = [h.strip() for h in (antibody_hotspots or "").split(",") if h.strip()]
    antibody_cfg = AntibodyConfig(
        enabled=bool(antibody_enabled),
        framework=antibody_framework,
        hotspot_source=antibody_hotspot_source,
        hotspot_residues=hotspots,
        num_designs=int(antibody_num_designs),
    )
    if antibody_enabled and "antibody" not in steps:
        steps = steps + ["antibody"]

    # Boltz-2 backend config (folding, docking, binder design share one resolver:
    # API key -> local -> graft fallback (folding only). 
    boltz2_cfg = Boltz2Config(
        api_key=(boltz_api_key or None),
        prefer_local=bool(boltz_prefer_local),
        allow_graft_fallback=bool(boltz_allow_graft),
    )
    if max_mutations is None:
        max_mutations = 0
        mutations = []
    else:
        if max_mutations > 0:
            max_mutations = int(max_mutations)
            mutations = mutations[:max_mutations]
        else: #<= 0
            max_mutations = 0
            mutations = []

    config = AnalysisConfig(
        uniprot_ids=ids,
        n_subunits=int(n_subunits),
        subunit_stoichiometry=stoich,
        mutations=mutations,
        max_mutations=max_mutations,
        proteoform_mode=proteoform_mode,
        proteoform_cap=int(proteoform_cap),
        structure_source=structure_source,
        local_pdb_id=local_pdb_id or None,
        antibody=antibody_cfg,
        boltz2=boltz2_cfg,
        run_ptm="ptm" in steps,
        run_md="md" in steps,
        md=MDConfig(engine=md_engine, production_steps=int(md_steps)),
        docking_engine=docking_engine,
        ligand_design_engine=ligand_design_engine,
        binding_site_method=binding_site_method,
        ensemble_docking=ensemble_docking,
        ptm=PTMConfig(pairs=ptm_pair_list),
        steps=steps,
        mature_auto_detect=bool(mature_auto),
        mature_regions=_parse_mature_regions_text(mature_regions_text, ids),
        foldx_binary=(foldx_binary or "").strip() or None,
    )
    return config


# ---------------------------------------------------------------------------
# Preset selector callback
# ---------------------------------------------------------------------------

def _load_preset(preset_name):
    """Load a preset config and return values for all Setup widgets."""
    if not preset_name or preset_name == "Custom (manual)":
        return (gr.update(),) * 20  # no change to any widget

    fn = PRESETS.get(preset_name)
    if not fn:
        return (gr.update(),) * 20

    cfg = fn()
    # Build widget values from config
    uniprot_str = ",".join(cfg.uniprot_ids)
    stoich_str = ",".join(str(s) for s in cfg.subunit_stoichiometry)
    # Mutations: pipe-separated per subunit
    mut_blocks = []
    for mut_list in cfg.mutations:
        mut_blocks.append(" ".join(mut_list))
    mutations_str = "|".join(mut_blocks) if mut_blocks else ""
    max_mut = str(cfg.max_mutations) if cfg.max_mutations is not None else "-1"
    # Legacy preset PTM fields -> pairs-table rows (presets normally leave
    # these empty, which means "auto-fetch observed PTMs from UniProt").
    pair_rows = []
    if cfg.ptm.pairs:
        for triple in cfg.ptm.pairs:
            pair_rows.append([triple[0], triple[1],
                              triple[2] if len(triple) > 2 and triple[2] else ""])
    else:
        for r in cfg.ptm.residues:
            for t in cfg.ptm.ptm_types:
                pair_rows.append([r, t, ""])
    ptm_pairs_df = pd.DataFrame(pair_rows, columns=PTM_PAIR_HEADERS)
    md_steps = str(cfg.md.production_steps)
    binding_site = cfg.binding_site_method
    ensemble = cfg.ensemble_docking
    steps = cfg.steps
    # Mature-chain widgets: reflect the preset's resolved regions (if any) in
    # the manual-override textbox so the user sees exactly what will be used.
    mature_auto = bool(getattr(cfg, "mature_auto_detect", True))
    mature_regions_str = "; ".join(
        f"{uid}:{int(reg[0])}-{int(reg[1])}"
        for uid, reg in (getattr(cfg, "mature_regions", None) or {}).items()
        if reg
    )

    return (
        uniprot_str,              # uniprot_ids
        cfg.n_subunits,           # n_subunits
        stoich_str,               # stoichiometry
        cfg.proteoform_mode,      # proteoform_mode
        cfg.proteoform_cap,       # proteoform_cap
        mutations_str,            # mutations_text
        int(max_mut),             # max_mutations
        cfg.structure_source,     # structure_source
        cfg.local_pdb_id or "",   # local_pdb_id
        ptm_pairs_df,             # ptm_pairs
        cfg.md.engine,            # md_engine
        int(md_steps),            # md_steps
        cfg.docking_engine,       # docking_engine
        cfg.ligand_design_engine, # ligand_design_engine
        binding_site,             # binding_site_method
        ensemble,                 # ensemble_docking
        steps,                    # selected_steps
        mature_auto,              # mature_auto
        mature_regions_str,       # mature_regions_text
        getattr(cfg, "foldx_binary", None) or "",  # foldx_binary
    )



def _find_structure_pdb(results_dir, structure_name):
    """Find a PDB file for a structure across all structure directories.

    Searches canonical structures (pdbs/tetramer, pdbs/monomer), PTM-modified
    structures (ptms/ptms) and proteoform structures (proteoforms). The
    ``use_ptms`` flag is kept for backward compatibility but PTM/proteoform
    directories are always searched: the structure name itself is unique to
    the current job, so including these directories cannot pick up stale
    files from other runs.
    """
    search_paths = [
        os.path.join(results_dir, "pdbs/tetramer"),
        os.path.join(results_dir, "pdbs/monomer"),
        os.path.join(results_dir, "ptms/ptms"),
        os.path.join(results_dir, "proteoforms"),
    ]
    
    
    for subdir in search_paths:
        path = os.path.join(subdir, f"{structure_name}.pdb")
        if os.path.exists(path):
            return path
    
    return None

# ---------------------------------------------------------------------------
# Fix 1: Live log streaming via generator + background thread
# ---------------------------------------------------------------------------

def _run_pipeline_threaded(config, log_buf, status_state):
    """Run the pipeline in a background thread, appending logs to log_buf."""
    def _emit(step, status, message):
        flag = {"ok": "[+]", "skipped": "[~]", "failed": "[!]"}.get(status, "[?]")
        log_buf.append(f"{flag} {step}: {message}")

    # Remove old results dir if it exists, then create a fresh one for this run.
    if os.path.exists(config.results_dir()):
        import shutil
        shutil.rmtree(config.results_dir())

    config.progress_callback = _emit
    try:
        results = run_analysis(config)
        rows = []
        for r in results:
            flag = {"ok": "OK", "skipped": "SKIP", "failed": "FAIL"}.get(r.status, "?")
            rows.append([flag, r.step, r.message, f"{r.elapsed_s:.1f}s", len(r.outputs)])
        status_state["results"] = rows
        status_state["done"] = True
        pd.DataFrame(rows, columns=["status", "step", "message", "elapsed_s", "n_outputs"]).to_csv(
            os.path.join(config.results_dir(), "step_status.csv"), index=False)
        status_state["results_dir"] = config.results_dir()
    except Exception as e:
        log_buf.append(f"[!] FATAL: {e}")
        log_buf.append(traceback.format_exc())
        status_state["done"] = True
        status_state["error"] = str(e)


def run_from_gui(
    preset_name,
    uniprot_ids, n_subunits, stoichiometry, proteoform_mode, proteoform_cap,
    mutations_text, max_mutations, structure_source, local_pdb_id,
    ptm_pairs, md_engine, md_steps,
    docking_engine, ligand_design_engine, binding_site_method,
    ensemble_docking, selected_steps,
    antibody_enabled, antibody_framework, antibody_hotspot_source,
    antibody_hotspots, antibody_num_designs,
    boltz_api_key, boltz_prefer_local, boltz_allow_graft,
    mature_auto, mature_regions_text,
    foldx_binary,
    state,
):
    """Gradio generator handler: streams live log + status table in real time."""
    # If a preset is selected, use it directly (avoids re-parsing text fields).
    # Antibody design is opt-in and not part of any preset, so honour the antibody
    # widgets even when a preset is chosen by enabling it on the preset config.
    if preset_name and preset_name != "Custom (manual)" and preset_name in PRESETS:
        config = PRESETS[preset_name]()
        # Honour the PTM pairs table on top of the preset too: a non-empty
        # table overrides the preset's (usually empty = auto-fetch) PTM config.
        ptm_pair_list = _parse_ptm_pairs(ptm_pairs)
        if ptm_pair_list:
            config.ptm.pairs = ptm_pair_list
        if antibody_enabled:
            hotspots = [h.strip() for h in (antibody_hotspots or "").split(",") if h.strip()]
            config.antibody = AntibodyConfig(
                enabled=True, framework=antibody_framework,
                hotspot_source=antibody_hotspot_source, hotspot_residues=hotspots,
                num_designs=int(antibody_num_designs),
            )
            if "antibody" not in config.steps:
                config.steps = list(config.steps) + ["antibody"]
        # Honour the Boltz backend widgets on top of the preset too, so users can
        # supply an API key / toggle prefer-local / graft without leaving the
        # preset.
        config.boltz2.api_key = (boltz_api_key or None)
        config.boltz2.prefer_local = bool(boltz_prefer_local)
        config.boltz2.allow_graft_fallback = bool(boltz_allow_graft)
        # Honour the mature-chain widgets on top of the preset: the auto-detect
        # toggle always applies, and the regions textbox ALWAYS replaces the
        # preset's regions (it is pre-filled with them by _load_preset, so this
        # is a no-op unless the user edited it; clearing the textbox is the
        # explicit opt-out and yields full-length chains).
        config.mature_auto_detect = bool(mature_auto)
        config.mature_regions = _parse_mature_regions_text(mature_regions_text,
                                                           config.uniprot_ids)
        # Honour the FoldX widget on top of the preset too.
        config.foldx_binary = (foldx_binary or "").strip() or None
    else:
        config = _build_config(
            uniprot_ids, n_subunits, stoichiometry, proteoform_mode, proteoform_cap,
            mutations_text, max_mutations, structure_source, local_pdb_id,
            ptm_pairs, md_engine, md_steps,
            docking_engine, ligand_design_engine, binding_site_method,
            ensemble_docking, selected_steps,
            antibody_enabled, antibody_framework, antibody_hotspot_source,
            antibody_hotspots, antibody_num_designs,
            boltz_api_key, boltz_prefer_local, boltz_allow_graft,
            mature_auto, mature_regions_text,
            foldx_binary,
        )
    log_buf = []
    new_state = {"done": False, "results": [], "results_dir": config.results_dir()}

    # Start pipeline in background thread
    thread = threading.Thread(target=_run_pipeline_threaded,
                              args=(config, log_buf, new_state), daemon=True)
    thread.start()

    # Stream logs until done
    while True:
        time.sleep(0.5)
        log_text = "\n".join(log_buf)
        results_df = new_state.get("results", [])
        yield log_text, results_df, new_state
        if new_state.get("done"):
            break

    thread.join(timeout=5)
    log_text = "\n".join(log_buf)
    results_df = new_state.get("results", [])
    yield log_text, results_df, new_state


# ---------------------------------------------------------------------------
# Results tab helpers
# ---------------------------------------------------------------------------

def _find_csv(results_dir, *patterns):
    if not results_dir or not os.path.isdir(results_dir):
        return None
    for root, _, fnames in os.walk(results_dir):
        for f in fnames:
            if f.endswith(".csv"):
                for pat in patterns:
                    if pat in f:
                        return os.path.join(root, f)
    return None


def _find_files(results_dir, ext):
    out = []
    if not results_dir or not os.path.isdir(results_dir):
        return out
    for root, _, fnames in os.walk(results_dir):
        for f in fnames:
            if f.endswith(ext):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, results_dir)
                out.append((rel, full))
    return sorted(out)


def _load_csv(path):
    if not path or not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return [list(df.columns)] + df.values.tolist()


def _load_csv_fmt(path, decimals_by_col=None):
    """Like ``_load_csv`` but formats chosen numeric columns to a fixed number
    of decimals for DISPLAY only (the CSV on disk keeps full precision).

    ``decimals_by_col`` maps column name -> number of decimals. Used so metrics
    with small meaningful differences (TM-scores) show enough resolution in the
    table instead of being visually rounded.
    """
    if not path or not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    decimals_by_col = decimals_by_col or {}
    for col, nd in decimals_by_col.items():
        if col in df.columns:
            num = pd.to_numeric(df[col], errors="coerce")
            df[col] = [format(v, f".{nd}f") if pd.notna(v) else df[col].iloc[i]
                       for i, v in enumerate(num)]
    return [list(df.columns)] + df.values.tolist()


def _df_to_table_fmt(df, decimals_by_col=None):
    """Convert a DataFrame to the ``[[headers], [row], ...]`` table format used
    by ``gr.Dataframe``, formatting chosen numeric columns to a fixed number of
    decimals for DISPLAY (the CSV on disk is untouched)."""
    if df is None or len(df) == 0:
        return []
    d = df.copy()
    decimals_by_col = decimals_by_col or {}
    for col, nd in decimals_by_col.items():
        if col in d.columns:
            num = pd.to_numeric(d[col], errors="coerce")
            d[col] = [format(v, f".{nd}f") if pd.notna(v) else d[col].iloc[i]
                      for i, v in enumerate(num)]
    return [list(d.columns)] + d.values.tolist()


def _pdb_viewer_label(rel_path):
    """Build a readable, grouped dropdown label for a discovered PDB.

    Distinguishes docked receptor+ligand complexes and designed antibody
    complexes from plain (receptor-only) structures so the user can pick the
    one that actually contains a ligand / antibody. The dropdown VALUE stays the
    relative path; only the shown label changes.
    """
    rp = rel_path.replace("\\", "/")
    base = os.path.basename(rp).replace(".pdb", "")
    low = rp.lower()

    # Designed target+drug complex (written by the ligand-design step to
    # ligand_design/complexes/<target>__<ligand>_complex.pdb)
    if "ligand_design/complexes/" in low and base.endswith("_complex"):
        stem = base[: -len("_complex")]
        if "__" in stem:
            struct, lig = stem.split("__", 1)
            return f"[designed ligand · DiffSBDD] {struct} + {lig}"
        return f"[designed ligand · DiffSBDD] {stem}"

    # Antibody–target complex (written by the antibody step to
    # antibody/complexes/<design>_complex.pdb or <design>_fullassembly.pdb).
    # MUST come before the generic "complexes/" branch below, which would
    # otherwise mislabel these as docked ligand complexes.
    if "antibody/complexes/" in low:
        if base.endswith("_fullassembly"):
            stem = base[: -len("_fullassembly")]
            return f"[antibody–target complex · full assembly] {stem}"
        stem = base[: -len("_complex")] if base.endswith("_complex") else base
        return f"[antibody–target complex] {stem}"

    # Docked receptor+ligand complex (written by the docking step to
    # docking*/complexes/<structure>__<ligand>_complex.pdb)
    if "complexes/" in low and base.endswith("_complex"):
        stem = base[: -len("_complex")]
        engine = "Boltz-2" if "boltz" in low else "Vina"
        if "__" in stem:
            struct, lig = stem.split("__", 1)
            return f"[docked · {engine}] {struct} + {lig}"
        return f"[docked · {engine}] {stem}"

    # Designed antibody complex (antibody/designs/*.pdb from RFAntibody)
    if "antibody/designs/" in low or "/designs/" in low and "antibody" in low:
        return f"[designed antibody] {base}"

    # Reference / co-crystal PDB (4-char PDB id like 5e83, 4dst)
    if re.match(r"^[0-9][a-z0-9]{3}$", base.lower()):
        return f"[reference PDB] {base.upper()}"

    # Plain structures grouped by subdir
    if "pdbs/tetramer/" in low:
        return f"[structure · tetramer] {base}"
    if "pdbs/monomer/" in low:
        return f"[structure · monomer] {base}"
    if "ptms/ptms/" in low:
        return f"[PTM] {base}"
    if "proteoforms/" in low:
        return f"[proteoform] {base}"
    return base


def _pdb_viewer_choices(pdb_files):
    """Return ``[(label, rel_path), …]`` for the structure-viewer dropdown.

    Complexes (docked / antibody) are listed FIRST so a user looking for the
    ligand or designed antibody finds them immediately, then plain structures,
    then reference PDBs. ``pdb_files`` is the output of ``_find_files(rd, '.pdb')``.
    """
    def _group_rank(rel):
        low = rel.lower()
        base = os.path.basename(low).replace(".pdb", "")
        if "antibody/complexes/" in low:
            return 0  # antibody–target complexes first
        if "complexes/" in low and base.endswith("_complex"):
            return 0  # docked + designed-ligand complexes first
        if "/designs/" in low:
            return 1  # designed antibodies
        if re.match(r"^[0-9][a-z0-9]{3}$", base):
            return 3  # reference PDBs last
        return 2      # plain structures
    choices = []
    for rel, _full in sorted(pdb_files, key=lambda t: (_group_rank(t[0]), t[0])):
        choices.append((_pdb_viewer_label(rel), rel))
    return choices


# ---------------------------------------------------------------------------
# Fix 5: 3Dmol.js viewer with proper script execution
# ---------------------------------------------------------------------------

_3DMOL_VIEWER_COUNTER = [0]

def _3dmol_html(pdb_text, style_script, width=500, height=400):
    if not pdb_text:
        return "<p>No PDB data.</p>"

    lines = pdb_text.split("\n")
    if len(lines) > 8000:
        pdb_text = "\n".join(lines[:8000]) + "\nEND"
    b64 = base64.b64encode(pdb_text.encode()).decode()

    _3DMOL_VIEWER_COUNTER[0] += 1
    vid = f"viewer_{_3DMOL_VIEWER_COUNTER[0]}_{int(time.time()*1000) % 1000000}"

    inner_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
      <style>html,body{{margin:0;padding:0;}}</style>
    </head>
    <body>
      <div id="{vid}" style="width:{width}px; height:{height}px;"></div>
      <script>
        window.onload = function() {{
          try {{
            var element = document.getElementById('{vid}');
            var viewer = $3Dmol.createViewer(element, {{backgroundColor: 'white'}});
            var pdbData = atob('{b64}');
            viewer.addModel(pdbData, 'pdb');
            {style_script}
            viewer.zoomTo();
            viewer.render();
            viewer.zoom(1.2, 800);
          }} catch(e) {{
            document.body.innerHTML = '<p style="color:red;">3Dmol error: ' + e.message + '</p>';
          }}
        }};
      </script>
    </body>
    </html>
    """

    escaped = html.escape(inner_doc, quote=True)
    return f'<iframe srcdoc="{escaped}" width="{width}" height="{height}" style="border:1px solid #ccc; border-radius:6px;" sandbox="allow-scripts allow-same-origin"></iframe>'


_STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "HSD", "HSE", "HSP",
}
_WATER = {"HOH", "WAT", "TIP", "TIP3", "SOL", "H2O"}
# Immunoglobulin chain IDs commonly emitted by RFAntibody/ImmuneBuilder outputs
_AB_CHAINS = {"H", "L"}

# Colorblind-safe qualitative palette (Okabe-Ito, 8 colors) for coloring protein
# cartoons by chain ID. Distinguishable under the common forms of colour-vision
# deficiency. Cycled if a structure has more chains than colours.
_CHAIN_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]


def _chain_color_map(chains):
    """Map an iterable of chain IDs -> hex colors using the colorblind-safe
    ``_CHAIN_PALETTE`` (cycled). Returns an ordered dict keyed by chain ID
    (sorted) so the 3D coloring and the legend agree."""
    out = {}
    for i, ch in enumerate(sorted(str(c) for c in chains if str(c).strip())):
        out[ch] = _CHAIN_PALETTE[i % len(_CHAIN_PALETTE)]
    return out


def _chain_legend_html(chain_colors, extra_items=None):
    """Build a compact HTML legend (colored chips + labels) for a chain->color
    map. ``extra_items`` is an optional list of (label, color) tuples appended
    after the chains (e.g. ligands)."""
    chips = []
    for ch, col in chain_colors.items():
        chips.append(
            f"<span style=\"display:inline-flex;align-items:center;margin:2px 10px 2px 0;"
            f"font-size:0.85em;\"><span style=\"display:inline-block;width:12px;height:12px;"
            f"background:{col};border:1px solid #888;border-radius:2px;margin-right:5px;\">"
            f"</span>Chain {ch}</span>")
    for label, col in (extra_items or []):
        chips.append(
            f"<span style=\"display:inline-flex;align-items:center;margin:2px 10px 2px 0;"
            f"font-size:0.85em;\"><span style=\"display:inline-block;width:12px;height:12px;"
            f"background:{col};border:1px solid #888;border-radius:2px;margin-right:5px;\">"
            f"</span>{label}</span>")
    if not chips:
        return ""
    return ("<div style=\"margin-top:6px;padding:4px 2px;line-height:1.6;\">"
            "<b style=\"font-size:0.85em;color:#444;\">Chains:</b>&nbsp;"
            + "".join(chips) + "</div>")


def _classify_pdb_contents(pdb_text, path_hint=""):
    """Inspect a PDB and report which molecule types are present.

    Returns dict with sets: ``protein_chains``, ``ligand_resns``,
    ``ligand_chains``, plus booleans ``has_protein``, ``has_ligand``,
    ``looks_like_antibody``.

    ``looks_like_antibody`` is True when both immunoglobulin chains H and L
    are present, OR — for nanobody (heavy-chain-only) designs, the default
    RFAntibody framework — when an H chain is present together with a target
    chain and the file comes from the antibody step (``path_hint`` contains
    "antibody"). The path hint avoids misclassifying ordinary structures
    that happen to have a chain named H.
    """
    protein_chains, ligand_resns, ligand_chains = set(), set(), set()
    for line in pdb_text.splitlines():
        rec = line[:6].strip()
        if rec == "ATOM":
            ch = line[21:22].strip()
            if ch:
                protein_chains.add(ch)
        elif rec == "HETATM":
            resn = line[17:20].strip()
            ch = line[21:22].strip()
            if resn in _WATER:
                continue
            if resn in _STD_AA:  # modified residue treated as protein
                if ch:
                    protein_chains.add(ch)
                continue
            ligand_resns.add(resn)
            if ch:
                ligand_chains.add(ch)
    looks_ab = bool(protein_chains) and (
        _AB_CHAINS.issubset(protein_chains)
        or ("H" in protein_chains
            and bool(protein_chains & {"L", "T"})
            and "antibody" in path_hint.replace("\\", "/").lower())
    )
    return {
        "protein_chains": protein_chains,
        "ligand_resns": ligand_resns,
        "ligand_chains": ligand_chains,
        "has_protein": bool(protein_chains),
        "has_ligand": bool(ligand_resns),
        "looks_like_antibody": looks_ab,
    }


def _pdb_to_html_viewer(pdb_path, width=500, height=400):
    """Molecule-type-aware 3Dmol viewer.

    Styling rules:
      * protein            -> cartoon coloured **by chain ID** (colorblind-safe
                              palette) with a chain legend
      * ligands / hetero   -> licorice/stick + coloured by element
      * antibody complexes  -> cartoon by chain (H / L / antigen) + sticks on the
                              antigen interface residues near the H/L chains

    Cartoons are coloured per chain (user request: "color cartoons by chain ID").
    A small HTML legend below the viewer maps each chain to its colour.
    """
    if not pdb_path or not os.path.exists(pdb_path):
        return "<p>No PDB file selected.</p>"
    with open(pdb_path, "r") as f:
        pdb_text = f.read()

    info = _classify_pdb_contents(pdb_text, path_hint=pdb_path)
    parts = []
    legend_html = ""

    if info["looks_like_antibody"]:
        # Antibody: semantic per-chain colours (H / L / antigen) + interface
        # sticks. This is still "colour by chain", with meaningful labels.
        antigen_chains = sorted(info["protein_chains"] - _AB_CHAINS)
        parts.append("viewer.setStyle({}, {cartoon: {color: 'spectrum'}});")
        parts.append("viewer.setStyle({chain: 'H'}, {cartoon: {color: '#0279EE'}});")
        parts.append("viewer.setStyle({chain: 'L'}, {cartoon: {color: '#75A025'}});")
        legend_items = {"H (heavy)": "#0279EE", "L (light)": "#75A025"}
        if antigen_chains:
            ag = "[" + ",".join(f"'{c}'" for c in antigen_chains) + "]"
            parts.append(f"viewer.setStyle({{chain: {ag}}}, {{cartoon: {{color: '#FF9400'}}}});")
            # interface sticks: antigen residues within 5A of the H/L chains
            parts.append(
                f"viewer.addStyle({{chain: {ag}, within: {{distance: 5, sel: {{chain: ['H','L']}}}}}}, "
                "{stick: {radius: 0.2, colorscheme: 'orangeCarbon'}});")
            legend_items["antigen (" + ",".join(antigen_chains) + ")"] = "#FF9400"
        # Build a legend from the semantic antibody colours.
        extra = [("ligand", "#2ca02c")] if info["has_ligand"] else None
        legend_html = _chain_legend_html(dict(legend_items), extra_items=extra)
    else:
        # General protein: colour each chain a distinct colorblind-safe colour.
        chain_colors = _chain_color_map(info["protein_chains"])
        if chain_colors:
            # default so any unclassified atoms still get a cartoon
            parts.append("viewer.setStyle({}, {cartoon: {color: 'spectrum'}});")
            for ch, col in chain_colors.items():
                parts.append(
                    f"viewer.setStyle({{chain: '{ch}'}}, {{cartoon: {{color: '{col}'}}}});")
        else:
            parts.append("viewer.setStyle({}, {cartoon: {color: 'spectrum'}});")
        extra = [("ligand", "#2ca02c")] if info["has_ligand"] else None
        legend_html = _chain_legend_html(chain_colors, extra_items=extra)

    if info["has_ligand"]:
        resns = "[" + ",".join(f"'{r}'" for r in sorted(info["ligand_resns"])) + "]"
        # ligands as licorice (thick sticks) + ball, coloured by element
        parts.append(
            f"viewer.setStyle({{resn: {resns}}}, "
            "{stick: {radius: 0.25, colorscheme: 'greenCarbon'}, "
            "sphere: {scale: 0.28}});")

    style = "\n          ".join(parts)
    viewer_html = _3dmol_html(pdb_text, style, width, height)
    if legend_html:
        return (f"<div style=\"width:{width}px;\">{viewer_html}{legend_html}</div>")
    return viewer_html


# ---------------------------------------------------------------------------
# Fix 3: PCN interactive visualization helpers
# ---------------------------------------------------------------------------

# PCN file path patterns (generated by pcn_miner's save functions)
_CENTRALITY_MEASURES = ["betweenness", "closeness", "degree_c", "eigenvector_c"]
_CENTRALITY_LABELS = {
    "betweenness": "Betweenness",
    "closeness": "Closeness",
    "degree_c": "Degree",
    "eigenvector_c": "Eigenvector",
}
_COMMUNITY_ALGOS = ["louvain", "leiden", "infomap"]

# Color palette for communities (up to 20 distinct colors)
_COMMUNITY_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
]


def _parse_pcn_dict(filepath):
    """Parse a PCN output file (Python dict repr with np.str_ keys)."""
    if not filepath or not os.path.exists(filepath):
        return {}
    with open(filepath) as f:
        content = f.read()
    # Strip np.str_('...') -> '...'
    cleaned = re.sub(r"np\.str_\(['\"]([^'\"]+)['\"]\)", r"'\1'", content)
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        return {}


def _parse_residue_label(label):
    """Parse 'VAL1 A' -> ('VAL', 1, 'A') or None."""
    m = re.match(r"([A-Z]+)(\d+)\s+([A-Z])", label.strip())
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None


def _find_centrality_file(pcn_dir, measure, structure):
    """Find the centrality file for a given measure and structure."""
    # Pattern: Centralities/{measure}{measure}Centralities/{measure}/Txt/{structure}_{measure}.txt
    path = os.path.join(pcn_dir, "Centralities", f"{measure}{measure}Centralities",
                        measure, "Txt", f"{structure}_{measure}.txt")
    if os.path.exists(path):
        return path
    # Fallback: search for it
    for root, _, fnames in os.walk(os.path.join(pcn_dir, "Centralities")):
        target = f"{structure}_{measure}.txt"
        if target in fnames:
            return os.path.join(root, target)
    return None


def _find_community_file(pcn_dir, algo, structure):
    """Find the community labels file for a given algorithm and structure."""
    # Pattern: Communities/{algo}{algo}/Communities/{structure}_Communities_{algo}_ncoms{N}.txt
    base = os.path.join(pcn_dir, "Communities", f"{algo}{algo}", "Communities")
    if not os.path.isdir(base):
        # Fallback: search
        for root, _, fnames in os.walk(os.path.join(pcn_dir, "Communities")):
            for f in fnames:
                if f.startswith(f"{structure}_Communities_{algo}_ncoms"):
                    return os.path.join(root, f)
        return None
    # There may be multiple files (different ncoms from re-runs); pick the first
    candidates = [f for f in os.listdir(base)
                  if f.startswith(f"{structure}_Communities_{algo}_ncoms")]
    if candidates:
        return os.path.join(base, sorted(candidates)[0])
    return None


def _find_wt_structure(structures):
    """Find the WT structure name from a list of structure names."""
    for name in structures:
        if "wt" in name.lower().split("-")[0]:
            return name
    for name in structures:
        if name.lower().startswith("wt"):
            return name
    # Fallback: look for a PDB ID-like name (e.g. 1a3n)
    for name in structures:
        if re.match(r"^[0-9][a-z0-9]{3}$", name.lower()):
            return name
    return structures[0] if structures else None


def _diverging_color(value):
    """Map a value in [-1, 1] to a blue-white-red diverging color hex.

    -1 = blue (#2166ac), 0 = white (#f7f7f7), +1 = red (#b2182b).
    """
    v = max(-1.0, min(1.0, value))
    if v >= 0:
        # white -> red
        r = int(247 + (178 - 247) * v)
        g = int(247 + (24 - 247) * v)
        b = int(247 + (43 - 247) * v)
    else:
        # white -> blue
        r = int(247 + (33 - 247) * (-v))
        g = int(247 + (102 - 247) * (-v))
        b = int(247 + (172 - 247) * (-v))
    return f"#{r:02x}{g:02x}{b:02x}"


def _structure_provenance_banner(rd):
    """Return a Markdown warning banner (or "") describing how structures were
    produced. When structures were built by ptm-psi side-chain grafting (no
    folding backend available), show a prominent red/bold warning so users know
    the backbone is identical to WT (TM-scores == 1.0 by construction).

    Reads ``structure_provenance.json`` (keys: method, backbone_identical,
    detail) written by the folding step via ``write_structure_provenance``.
    """
    if not rd:
        return ""
    p = os.path.join(rd, "structure_provenance.json")
    if not os.path.exists(p):
        return ""
    try:
        with open(p) as f:
            prov = json.load(f)
    except Exception:
        return ""
    method = str(prov.get("method", "")).lower()
    detail = str(prov.get("detail", "")).strip()
    if method == "graft":
        try:
            from .core.steps._boltz_backend import GRAFT_WARNING
        except Exception:
            GRAFT_WARNING = (
                "Structures were built by ptm-psi side-chain grafting onto the "
                "wild-type backbone because no folding backend was available. "
                "The backbone is IDENTICAL to WT for every variant, so "
                "TM-scores are 1.0 by construction and there is no structural, "
                "pocket, or docking signal.")
        msg = (
            "<div style=\"background:#fdecea;border:2px solid #b2182b;"
            "border-radius:6px;padding:10px 14px;\">"
            "\u26a0\ufe0f&nbsp;<b style=\"color:#b2182b;\">GRAFT FALLBACK "
            "\u2014 backbone-identical structures.</b><br>"
            f"<span style=\"color:#7a1a12;\">{GRAFT_WARNING}</span>")
        if detail:
            msg += f"<br><span style=\"color:#7a1a12;font-size:0.9em;\">{detail}</span>"
        msg += "</div>"
        return msg
    # Real folding backends: a short, non-alarming provenance note.
    label = {"api": "Boltz-2 API (api.boltz.bio)",
             "local": "local boltz install"}.get(method, method or "unknown")
    return (
        f"<div style=\"background:#14283a;border:1px solid #2e7d32;"
        f"border-radius:6px;padding:8px 12px;color:#e6f0ff;\">\u2705 Structures folded via "
        f"<b>{label}</b>. TM-score / pocket / docking signals are meaningful."
        + (f"<br><span style=\"font-size:0.9em;\">{detail}</span>" if detail else "")
        + "</div>"
    )


def _pocket_warning_banner(rd):
    """Return a Markdown warning banner (or "") for the pocket step.

    Reads ``pocket_warnings.json`` (keys: binding_site_method, all_volumes_zero,
    warning) written by the pocket step when ``binding_site_method == 'reference'``
    (fixed reference box -> volume/drift always 0) or when every detector
    produced a zero volume. Surfaces the same warning shown in the step message
    inside the Pocket Prediction and Pocket Drift sections of the Results tab.
    """
    if not rd:
        return ""
    # pocket_warnings.json is written next to pocket_predictions.csv. Look in the
    # results dir and common pocket subdirectories.
    candidates = [os.path.join(rd, "pocket_warnings.json"),
                  os.path.join(rd, "pocket", "pocket_warnings.json"),
                  os.path.join(rd, "pockets", "pocket_warnings.json")]
    warn_path = next((c for c in candidates if os.path.exists(c)), None)
    if warn_path is None:
        # Fall back to a recursive search (results trees are small).
        for root, _, fnames in os.walk(rd):
            if "pocket_warnings.json" in fnames:
                warn_path = os.path.join(root, "pocket_warnings.json")
                break
    if warn_path is None:
        return ""
    try:
        with open(warn_path) as f:
            w = json.load(f)
    except Exception:
        return ""
    text = str(w.get("warning", "")).strip()
    if not text:
        return ""
    method = w.get("binding_site_method")
    heading = ("Binding-site method = 'reference'"
               if method == "reference" else "Pocket detection limitation")
    return ("<div style=\"background:#fff4e5;border:2px solid #FF9400;"
            "border-radius:6px;padding:10px 14px;\">"
            f"\u26a0\ufe0f&nbsp;<b style=\"color:#a15c00;\">{heading} \u2014 "
            "pocket prediction and pocket drift may not change.</b><br>"
            f"<span style=\"color:#7a4a00;\">{text}</span></div>")


def _get_cmap(name):
    """Return a matplotlib colormap by name, compatible across matplotlib
    versions. ``matplotlib.cm.get_cmap`` was removed in matplotlib 3.9; the
    modern access is ``matplotlib.colormaps[name]``."""
    import matplotlib
    try:
        return matplotlib.colormaps[name]          # matplotlib >= 3.5 (preferred)
    except Exception:
        import matplotlib.cm as _cm
        return _cm.get_cmap(name)                   # legacy fallback


def _viridis_color(value):
    """Map a value in [0, 1] to a viridis hex color (perceptually-uniform,
    colorblind-friendly). Uses matplotlib's viridis colormap."""
    import matplotlib.colors as _mcolors
    v = max(0.0, min(1.0, float(value)))
    rgba = _get_cmap("viridis")(v)
    return _mcolors.to_hex(rgba)

def _colorbar_svg(vmin, vmax, label, cmap="viridis", diverging=False, width=260, height=54):
    """Return a small inline SVG colorbar (gradient + min/mid/max ticks).

    ``diverging=True`` uses the blue-white-red delta scale (symmetric about 0);
    otherwise a sequential viridis ramp from ``vmin`` to ``vmax``.
    """
    n = 32
    stops = []
    for i in range(n + 1):
        f = i / n
        if diverging:
            col = _diverging_color(2.0 * f - 1.0)  # map [0,1] -> [-1,1]
        else:
            col = _viridis_color(f)
        stops.append(f'<stop offset="{f*100:.1f}%" stop-color="{col}"/>')

    grad = "".join(stops)

    if diverging:
        lo, mid, hi = f"{vmin:.3g}", "0", f"{vmax:.3g}"
    else:
        lo, mid, hi = f"{vmin:.3g}", f"{(vmin + vmax) / 2:.3g}", f"{vmax:.3g}"

    bar_w = width - 20
    fam = "Liberation Sans, Arimo, DejaVu Sans, sans-serif"
    text_color = "#ffffff"

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="font-family:{fam};">'
        f'<defs>'
        f'<linearGradient id="cbgrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'{grad}'
        f'</linearGradient>'
        f'</defs>'
        f'<text x="10" y="12" font-size="11" font-weight="bold" '
        f'fill="{text_color}">{html.escape(label)}</text>'
        f'<rect x="10" y="18" width="{bar_w}" height="14" fill="url(#cbgrad)" '
        f'stroke="#888" stroke-width="0.5"/>'
        f'<text x="10" y="46" font-size="10" fill="{text_color}">{lo}</text>'
        f'<text x="{10 + bar_w / 2:.0f}" y="46" font-size="10" '
        f'text-anchor="middle" fill="{text_color}">{mid}</text>'
        f'<text x="{10 + bar_w:.0f}" y="46" font-size="10" '
        f'text-anchor="end" fill="{text_color}">{hi}</text>'
        f'</svg>'
    )

def _pcn_centrality_html(structure, measure, state, width=480, height=400, view_mode="raw"):
    """3Dmol.js HTML colouring residues by PCN centrality.

    ``view_mode='raw'`` colours by the structure's own centrality on a viridis
    scale (works for WT and mutants). ``view_mode='delta'`` colours by
    (mutant - WT) on a blue-white-red diverging scale (mutants only). A small
    inline SVG colorbar is prepended so the mapping is legible.
    """
    rd = state.get("results_dir", "") if state else ""
    if not rd:
        return "<p>No results directory.</p>"

    pcn_dir = os.path.join(rd, "pcn_outputs")
    if not os.path.isdir(pcn_dir):
        return "<p>PCN results not found. Run the PCN step first.</p>"

    pdb_dir = os.path.join(rd, "pdbs", "tetramer")
    if not os.path.isdir(pdb_dir):
        pdb_dir = os.path.join(rd, "pdbs", "monomer")
    all_structures = [f.replace(".pdb", "") for f in sorted(os.listdir(pdb_dir))
                      if f.endswith(".pdb")] if os.path.isdir(pdb_dir) else []
    wt_name = _find_wt_structure(all_structures)

    label_text = _CENTRALITY_LABELS.get(measure, measure)
    struct_pdb_path = _find_structure_pdb(rd, structure)
    if not struct_pdb_path:
        return f"<p>PDB file not found: {structure}.pdb</p>"
    with open(struct_pdb_path) as f:
        pdb_text = f.read()

    baseline = "viewer.setStyle({}, {cartoon: {color: 'lightgray'}});"

    # ---- RAW mode: colour by the structure's own centrality (viridis) --------
    if view_mode == "raw":
        cent = _parse_pcn_dict(_find_centrality_file(pcn_dir, measure, structure))
        if not cent:
            return f"<p>Centrality data not found for {measure} ({structure}).</p>"
        vals = [float(v) for v in cent.values()]
        vmin, vmax = min(vals), max(vals)
        span = (vmax - vmin) or 1.0
        style_lines = []
        for lab, val in cent.items():
            parsed = _parse_residue_label(lab)
            if not parsed:
                continue
            _resn, resi, chain = parsed
            norm = (float(val) - vmin) / span
            color = _viridis_color(norm)
            style_lines.append(
                f"viewer.setStyle({{chain:'{chain}',resi:{resi}}},"
                f"{{cartoon:{{color:'{color}'}}}});")
        style_script = "\n          ".join([baseline] + style_lines)
        viewer = _3dmol_html(pdb_text, style_script, width, height)
        cbar = _colorbar_svg(vmin, vmax, f"{label_text} centrality ({structure})",
                             diverging=False, width=width - 10)
        return f'<div style="margin-bottom:4px;">{cbar}</div>' + viewer

    # ---- DELTA mode: colour by (mutant - WT) on a diverging scale -----------
    if structure == wt_name:
        return ("<p>Delta view needs a mutant (WT minus WT = 0). "
                "Switch to 'Raw centrality' to view WT.</p>")
    wt_cent = _parse_pcn_dict(_find_centrality_file(pcn_dir, measure, wt_name))
    mut_cent = _parse_pcn_dict(_find_centrality_file(pcn_dir, measure, structure))
    if not wt_cent or not mut_cent:
        return f"<p>Centrality data not found for {measure} ({wt_name} or {structure}).</p>"
    deltas = {}
    for lab, val in mut_cent.items():
        if lab in wt_cent:
            deltas[lab] = float(val) - float(wt_cent[lab])
    if not deltas:
        return "<p>No matching residues between WT and mutant.</p>"
    max_abs = max(abs(v) for v in deltas.values()) or 1.0
    style_lines = []
    for lab, delta in deltas.items():
        parsed = _parse_residue_label(lab)
        if not parsed:
            continue
        _resn, resi, chain = parsed
        color = _diverging_color(delta / max_abs)
        style_lines.append(
            f"viewer.setStyle({{chain:'{chain}',resi:{resi}}},"
            f"{{cartoon:{{color:'{color}'}}}});")
    style_script = "\n          ".join([baseline] + style_lines)
    viewer = _3dmol_html(pdb_text, style_script, width, height)
    cbar = _colorbar_svg(-max_abs, max_abs,
                         f"\u0394 {label_text} vs WT (blue=down, red=up)",
                         diverging=True, width=width - 10)
    return f'<div style="margin-bottom:4px;">{cbar}</div>' + viewer


def _pcn_community_html(structure, algo, state, width=480, height=400):
    """Generate 3Dmol.js HTML showing community colors + changed residues vs WT."""
    rd = state.get("results_dir", "") if state else ""
    if not rd:
        return "<p>No results directory.</p>"

    pcn_dir = os.path.join(rd, "pcn_outputs")
    if not os.path.isdir(pcn_dir):
        return "<p>PCN results not found. Run the PCN step first.</p>"

    # Colour the SELECTED structure by its own community assignment (single
    # structure; no WT-difference overlay/white sticks — those were confusing and
    # are dropped per the community-view request). WT can be viewed like any
    # other structure.
    comm = _parse_pcn_dict(_find_community_file(pcn_dir, algo, structure))
    if not comm:
        return f"<p>Community data not found for {algo} ({structure}).</p>"

    struct_pdb_path = _find_structure_pdb(rd, structure)
    if not struct_pdb_path:
        return f"<p>PDB file not found: {structure}.pdb</p>"
    with open(struct_pdb_path) as f:
        pdb_text = f.read()

    style_lines = []
    for label, comm_id in comm.items():
        parsed = _parse_residue_label(label)
        if not parsed:
            continue
        _resn, resi, chain = parsed
        color = _COMMUNITY_COLORS[int(comm_id) % len(_COMMUNITY_COLORS)]
        style_lines.append(
            f"viewer.setStyle({{chain:'{chain}',resi:{resi}}},"
            f"{{cartoon:{{color:'{color}'}}}});")

    # Global cartoon baseline FIRST so the whole protein is cartoon (not the
    # 3Dmol default lines/licorice); per-community colored cartoon is layered on
    # top.
    baseline = "viewer.setStyle({}, {cartoon: {color: 'lightgray'}});"
    style_script = "\n          ".join([baseline] + style_lines)

    n_comms = len(set(comm.values()))
    fam = "Liberation Sans, Arimo, DejaVu Sans, sans-serif"
    caption = (f'<div style="font-family:{fam};font-size:11px;margin-bottom:4px;">'
               f'<b>{structure}</b>: {n_comms} communities ({algo}), '
               f'coloured by community id</div>')
    viewer = _3dmol_html(pdb_text, style_script, width, height)
    return caption + viewer


def _centrality_delta_bar_file(rd, structure, measure):
    """Build (and cache) the top-10 |Δ centrality| signed bar plot PNG for a
    structure vs WT. Returns the PNG path, or None when data is unavailable."""
    if not rd:
        return None
    pcn_dir = os.path.join(rd, "pcn_outputs")
    if not os.path.isdir(pcn_dir):
        return None
    pdb_dir = os.path.join(rd, "pdbs", "tetramer")
    if not os.path.isdir(pdb_dir):
        pdb_dir = os.path.join(rd, "pdbs", "monomer")
    all_structures = [f.replace(".pdb", "") for f in sorted(os.listdir(pdb_dir))
                      if f.endswith(".pdb")] if os.path.isdir(pdb_dir) else []
    wt_name = _find_wt_structure(all_structures)
    if not wt_name or structure == wt_name:
        return None
    wt_cent = _parse_pcn_dict(_find_centrality_file(pcn_dir, measure, wt_name))
    mut_cent = _parse_pcn_dict(_find_centrality_file(pcn_dir, measure, structure))
    if not wt_cent or not mut_cent:
        return None
    plots_dir = os.path.join(pcn_dir, "delta_bars")
    os.makedirs(plots_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{structure}_{measure}")
    stem = os.path.join(plots_dir, f"delta_{safe}")
    try:
        out = viz.centrality_delta_bars(wt_cent, mut_cent, measure, stem,
                                        structure=structure, top_n=10)
    except Exception as e:
        log.warning("centrality_delta_bars failed for %s/%s: %s", structure, measure, e)
        return None
    return out.get("png") if out and out.get("n") else None


def pcn_visualize(structure, measure, algo, state, view_mode="raw"):
    """Callback for PCN visualize button.

    Returns (centrality_html, community_html, legend, delta_bar_png).
    """
    if not structure:
        return ("<p>Select a structure.</p>", "<p>Select a structure.</p>",
                "Select a structure.", None)

    rd = state.get("results_dir", "") if state else ""
    pcn_dir = os.path.join(rd, "pcn_outputs") if rd else ""

    mode = "delta" if (view_mode or "").lower().startswith("d") else "raw"
    cent_html = _pcn_centrality_html(structure, measure, state, view_mode=mode)
    comm_html = _pcn_community_html(structure, algo, state)

    # Top-10 |Δ centrality| bar plot (signed: red=increase, blue=decrease vs WT).
    bar_file = _centrality_delta_bar_file(rd, structure, measure)

    # Build legend text
    measure_label = _CENTRALITY_LABELS.get(measure, measure)
    pdb_dir = os.path.join(rd, "pdbs", "tetramer") if rd else ""
    if not os.path.isdir(pdb_dir):
        pdb_dir = os.path.join(rd, "pdbs", "monomer") if rd else ""
    all_structures = [f.replace(".pdb", "") for f in sorted(os.listdir(pdb_dir))
                      if f.endswith(".pdb")] if os.path.isdir(pdb_dir) else []
    wt_name = _find_wt_structure(all_structures)

    comm = _parse_pcn_dict(_find_community_file(pcn_dir, algo, structure)) if pcn_dir else {}
    n_communities = len(set(comm.values())) if comm else 0

    if mode == "raw":
        cent_desc = (f"Left: {measure_label} centrality of {structure} "
                     f"(viridis: purple = low, yellow = high).")
    else:
        cent_desc = (f"Left: {measure_label} centrality \u0394 vs WT "
                     f"(red = increased, blue = decreased).")
    legend = (f"{cent_desc}  Right: {algo} communities of {structure} "
              f"({n_communities} communities, coloured by community id).  "
              f"Bar plot: top-10 residues by |\u0394 centrality| vs WT.")
    return cent_html, comm_html, legend, bar_file


# ---------------------------------------------------------------------------
# MD trajectory overlay
# ---------------------------------------------------------------------------

def _clean_md_label(traj_rel):
    """Derive a readable structure label from a trajectory's relative path."""
    label = os.path.basename(os.path.dirname(traj_rel))
    label = label.replace("Mut_", "").replace("-tetramer", "").replace("_", " ")
    if label.lower() in ("1a3n", "1f41", "1aie"):
        label = "WT"
    return label


def _md_series_from_selection(traj_rels, results_dir, kind):
    """Build viz.md_overlay series for the selected trajectories.

    Prefers the per-structure ``rmsd.csv`` / ``rmsf.csv`` written by md.py
    (fast, no re-load); falls back to recomputing from ``trajectory.pdb`` with
    mdtraj if the CSV is absent (older runs).
    """
    import numpy as np
    series = []
    csv_name = "rmsf.csv" if kind == "rmsf" else "rmsd.csv"
    for traj_rel in traj_rels:
        traj_dir = os.path.dirname(os.path.join(results_dir, traj_rel))
        label = _clean_md_label(traj_rel)
        csv_path = os.path.join(traj_dir, csv_name)
        if os.path.exists(csv_path):
            try:
                d = pd.read_csv(csv_path)
                if kind == "rmsf":
                    series.append({"label": label, "x": d["residue"].to_numpy(),
                                   "y": d["rmsf_A"].to_numpy()})
                else:
                    series.append({"label": label, "x": d["time_ps"].to_numpy(),
                                   "y": d["rmsd_A"].to_numpy()})
                continue
            except Exception as e:
                log.warning("Failed to read %s: %s", csv_path, e)
        # Fallback: recompute from trajectory.pdb
        full = os.path.join(results_dir, traj_rel)
        if not os.path.exists(full):
            continue
        try:
            import mdtraj
            traj = mdtraj.load(full)
            if kind == "rmsf":
                y = mdtraj.rmsf(traj, traj, frame=0) * 10.0
                x = np.arange(1, len(y) + 1)
            else:
                y = mdtraj.rmsd(traj, traj, 0) * 10.0
                x = np.arange(len(y)) * 0.1
            series.append({"label": label, "x": x, "y": y})
        except Exception as e:
            log.warning("Failed to load trajectory %s: %s", traj_rel, e)
    return series


def _all_trajectory_rels(results_dir):
    """Return relative paths of every ``trajectory.pdb`` under the results dir.

    The MD overlay now shows ALL structures at once and lets the user toggle
    individual traces via the Plotly legend, so we no longer rely on a manual
    selection widget.
    """
    rels = []
    if not results_dir or not os.path.isdir(results_dir):
        return rels
    for root, _, fnames in os.walk(results_dir):
        for f in fnames:
            if f == "trajectory.pdb":
                rels.append(os.path.relpath(os.path.join(root, f), results_dir))
    return sorted(rels)


def overlay_md_callback(kind, state):
    """Build an interactive RMSD or RMSF overlay of ALL MD trajectories.

    Every structure with MD output is plotted as its own Plotly trace; the user
    shows/hides individual structures through the Plotly legend (no server-side
    selection needed).
    """
    rd = state.get("results_dir", "") if state else ""
    if not rd:
        return "<p>No results directory. Run the pipeline first.</p>", "", "No results directory."
    kind = "rmsf" if str(kind).lower().startswith("rmsf") else "rmsd"
    traj_rels = _all_trajectory_rels(rd)
    if not traj_rels:
        return ("<p>No MD trajectories found. Run the 'md' step first.</p>", "",
                "No MD trajectories found.")
    series = _md_series_from_selection(traj_rels, rd, kind)
    if not series:
        return ("<p>No RMSD/RMSF data found.</p>", "",
                "No data (missing CSVs and trajectories).")
    plot_dir = os.path.join(rd, "_plots"
                            )
    os.makedirs(plot_dir, exist_ok=True)
    stem = os.path.join(plot_dir, f"md_overlay_{kind}")
    out = viz.md_overlay(series, stem, kind=kind)
    html_out = _html_iframe(out.get("html"), height=540)
    static = out.get("svg") or out.get("png") or None
    if static is not None and not os.path.isfile(static):
        static = None
    n = out.get("n", 0)
    return (
        html_out,
        static,
        f"Interactive {kind.upper()} overlay of {n} structure(s) — "
        f"toggle structures via the legend. "
        f"Static file: {os.path.basename(static) if static else 'n/a'}"
    )

# ---------------------------------------------------------------------------
# Data-plot builder (docking / pocket / per-subsection summaries)
# ---------------------------------------------------------------------------

def _read_csv_df(path):
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            log.warning("Failed to read %s: %s", path, e)
    return None


def _find_exact_csv(results_dir, filename):
    """Find a CSV by exact basename (avoids substring collisions like
    ``docking_summary`` matching ``docking_summary_boltz2``)."""
    if not results_dir or not os.path.isdir(results_dir):
        return None
    for root, _, fnames in os.walk(results_dir):
        if filename in fnames:
            return os.path.join(root, filename)
    return None


def _find_exact_csv_glob(results_dir, pattern, exclude=None):
    """Find the first CSV whose basename matches a glob ``pattern`` (e.g.
    ``tm_scores_*.csv``), optionally skipping names containing ``exclude``
    (e.g. ``_all.csv`` to avoid the pairwise matrix)."""
    import fnmatch
    if not results_dir or not os.path.isdir(results_dir):
        return None
    for root, _, fnames in os.walk(results_dir):
        for f in sorted(fnames):
            if fnmatch.fnmatch(f, pattern) and (not exclude or exclude not in f):
                return os.path.join(root, f)
    return None


# Component columns that make up the composite impact score. Used to decide
# whether an impact row is "all zero" (no measurable impact) so the GUI can
# drop it, per the requested behaviour.
_IMPACT_COMPONENT_COLS = ("structural", "binding", "dynamics", "network", "sequence")


def _impact_scores_df(rd):
    """Load the proteoform impact *scores* table (not the components sidecar).

    ``_find_csv(rd, "impact")`` would also match ``impact_score_components.csv``;
    this prefers the actual scores file by exact basename first.
    """
    path = (_find_exact_csv(rd, "proteoform_impact_scores.csv")
            or _find_csv(rd, "proteoform_impact_scores")
            or _find_csv(rd, "impact_scores")
            or _find_csv(rd, "impact"))
    # never treat the components sidecar as the scores table
    if path and os.path.basename(path) == "impact_score_components.csv":
        path = (_find_exact_csv(rd, "proteoform_impact_scores.csv")
                or _find_csv(rd, "impact_scores"))
    return _read_csv_df(path)


def _drop_zero_impact_rows(df):
    """Return (filtered_df, n_dropped, all_zero).

    Drops proteoform rows whose impact is entirely zero across every available
    numeric component (and composite). If every row is zero, returns an empty
    frame with all_zero=True so the caller can show an honest note instead of a
    misleading all-zero table/plot.
    """
    if df is None or len(df) == 0:
        return df, 0, False
    numeric_cols = [c for c in list(_IMPACT_COMPONENT_COLS) + ["composite"]
                    if c in df.columns]
    if not numeric_cols:
        return df, 0, False
    vals = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    nonzero_mask = (vals.abs().sum(axis=1) > 0)
    n_dropped = int((~nonzero_mask).sum())
    filtered = df[nonzero_mask].reset_index(drop=True)
    all_zero = (len(filtered) == 0 and len(df) > 0)
    return filtered, n_dropped, all_zero


def _drop_zero_pocket_rows(df):
    """Return (filtered_df, n_dropped, all_zero) for pocket predictions.

    A pocket row carries no real measurement when the detector failed or the
    reported volume is 0 (the geometric fallback emits volume=0,
    detector_failed=True). Those rows are dropped so the plot/table only shows
    structures with an actually detected pocket. If none remain, all_zero=True.
    """
    if df is None or len(df) == 0:
        return df, 0, False
    d = df.copy()
    vol_col = "volume" if "volume" in d.columns else (
        "volume_A3" if "volume_A3" in d.columns else None)
    failed = (d["detector_failed"].astype(str).str.lower().isin(["true", "1"])
              if "detector_failed" in d.columns else pd.Series(False, index=d.index))
    if vol_col is not None:
        vol = pd.to_numeric(d[vol_col], errors="coerce").fillna(0.0)
        keep = (~failed) & (vol.abs() > 0)
    else:
        keep = ~failed
    n_dropped = int((~keep).sum())
    filtered = d[keep].reset_index(drop=True)
    all_zero = (len(filtered) == 0 and len(df) > 0)
    return filtered, n_dropped, all_zero


def _drop_zero_drift_rows(df):
    """Return (filtered_df, n_dropped, all_zero) for pocket drift.

    A drift row is uninformative when both the volume change and the pocket
    centre displacement are zero (nothing moved / no pocket to compare). Those
    rows are dropped. If none remain, all_zero=True.
    """
    if df is None or len(df) == 0:
        return df, 0, False
    d = df.copy()
    cols = [c for c in ("volume_change", "center_displacement_A",
                        "center_displacement") if c in d.columns]
    if not cols:
        return d, 0, False
    vals = d[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    keep = (vals.abs().sum(axis=1) > 0)
    n_dropped = int((~keep).sum())
    filtered = d[keep].reset_index(drop=True)
    all_zero = (len(filtered) == 0 and len(df) > 0)
    return filtered, n_dropped, all_zero


def _build_result_plots(rd):
    """Generate all data-driven result plots into ``<rd>/_plots``.

    Returns a dict mapping a plot key -> HTML iframe string, so the Results tab
    can drop each one straight into a gr.HTML panel. Missing inputs yield a
    friendly placeholder rather than an error.
    """
    placeholder = "<p>No data yet. Run the pipeline and click Refresh.</p>"
    keys = ["dock_bars", "dock_scatter", "pocket_vol", "pocket_drift",
            "impact_summary", "tm_summary", "ab_scatter", "ab_bars"]
    panels = {k: placeholder for k in keys}
    # Track which panels actually received real data. Callers use this to hide
    # empty panels (e.g. docking) instead of showing a "No data yet" box.
    populated = set()
    panels["_populated"] = populated
    if not rd:
        return panels
    plot_dir = os.path.join(rd, "_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ---- Docking: Vina affinity bars ----
    vina_df = _read_csv_df(_find_exact_csv(rd, "docking_summary.csv"))
    if vina_df is not None:
        out = viz.docking_affinity_bars(vina_df, os.path.join(plot_dir, "dock_affinity"))
        if out.get("n"):
            panels["dock_bars"] = _html_iframe(out.get("html"),
                                               height=max(360, 42 * out["n"] + 140))
            populated.add("dock_bars")

    # ---- Docking: Boltz-2 pTM vs ipTM scatter ----
    # Only counts as populated when the CSV carries real confidence values
    # (ptm/iptm/confidence). On a local run Boltz-2 needs a GPU, so the file is
    # often absent or status-only -> the panel stays empty and gets hidden.
    boltz_df = _read_csv_df(_find_exact_csv(rd, "docking_summary_boltz2.csv"))
    if boltz_df is not None:
        conf_cols = [c for c in ("ptm", "iptm", "confidence_score", "complex_plddt")
                     if c in boltz_df.columns]
        has_conf = False
        for c in conf_cols:
            if pd.to_numeric(boltz_df[c], errors="coerce").notna().any():
                has_conf = True
                break
        if has_conf:
            out = viz.docking_boltz_scatter(boltz_df, os.path.join(plot_dir, "dock_boltz"))
            if out.get("n"):
                panels["dock_scatter"] = _html_iframe(out.get("html"), height=660)
                populated.add("dock_scatter")

    # ---- Pocket volume bars ----
    pocket_df = _read_csv_df(_find_exact_csv(rd, "pocket_predictions.csv")
                             or _find_csv(rd, "pocket_pred") or _find_csv(rd, "pocket"))
    if pocket_df is not None:
        # Drop structures with no real pocket (detector failed / volume 0).
        pocket_df, _pn_dropped, _p_all_zero = _drop_zero_pocket_rows(pocket_df)
        if _p_all_zero:
            panels["pocket_vol"] = (
                "<p>No binding pocket was detected for any structure "
                "(all detectors failed or returned an empty pocket), so no real "
                "volume could be measured. Nothing is plotted rather than showing "
                "zeros as data.</p>")
        elif len(pocket_df):
            out = viz.pocket_volume_bars(pocket_df, os.path.join(plot_dir, "pocket_volume"))
            if out.get("n"):
                panels["pocket_vol"] = _html_iframe(out.get("html"), height=520)

    # ---- Pocket drift scatter ----
    drift_df = _read_csv_df(_find_csv(rd, "pocket_drift"))
    if drift_df is not None:
        # Drop proteoforms with zero drift (no volume change and no centre shift).
        drift_df, _dn_dropped, _d_all_zero = _drop_zero_drift_rows(drift_df)
        if _d_all_zero:
            panels["pocket_drift"] = (
                "<p>No pocket drift to show: every proteoform has zero volume "
                "change and zero pocket-centre displacement vs wild-type. Locally, "
                "mutant structures share the WT backbone, so pockets do not move; "
                "run the GPU pipeline for real structural divergence.</p>")
        elif len(drift_df):
            out = viz.pocket_drift_scatter(drift_df, os.path.join(plot_dir, "pocket_drift"))
            if out.get("n"):
                panels["pocket_drift"] = _html_iframe(out.get("html"), height=580)

    # ---- Impact score summary (composite ranking) ----
    impact_df = _impact_scores_df(rd)
    if impact_df is not None:
        # Drop proteoforms with no measurable impact (all-zero across every
        # component). Locally, mutant structures share the WT backbone so the
        # structural/binding/dynamics/network terms are all zero; only rows
        # that actually differ (e.g. ESM2 sequence distance) are informative.
        impact_df, _n_dropped, _all_zero = _drop_zero_impact_rows(impact_df)
        if _all_zero:
            panels["impact_summary"] = (
                "<p>No proteoform shows a measurable impact yet. All impact "
                "components are zero &mdash; in local mode mutant structures "
                "share the wild-type backbone, so structural/binding/dynamics/"
                "network terms are all 0. Run the GPU pipeline (Boltz-2 folding "
                "+ docking) for structural divergence.</p>")
        elif len(impact_df):
            val_col = None
            for c in ("composite", "composite_score", "impact_score", "score"):
                if c in impact_df.columns:
                    val_col = c
                    break
            lab_col = "proteoform" if "proteoform" in impact_df.columns else (
                impact_df.columns[0] if len(impact_df.columns) else None)
            if val_col and lab_col:
                out = viz.summary_bar(impact_df, lab_col, val_col,
                                      os.path.join(plot_dir, "impact_summary"),
                                      title="Proteoform impact ranking",
                                      value_title="Composite impact score")
                if out.get("n"):
                    panels["impact_summary"] = _html_iframe(out.get("html"),
                                                            height=max(340, 34 * out["n"] + 140))

    # ---- TM-score summary ----
    # Prefer the WT-vs-mutant file (has a 'Mutant' column); avoid the *_all.csv
    # pairwise matrix which has no single value column.
    tm_wt_csv = _find_exact_csv_glob(rd, "tm_scores_*.csv", exclude="_all.csv") \
        or _find_csv(rd, "tm_scores")
    tm_df = _read_csv_df(tm_wt_csv)
    if tm_df is not None:
        val_col = None
        for c in ("TM-score", "tm_score", "tmscore", "TMscore"):
            if c in tm_df.columns:
                val_col = c
                break
        lab_col = None
        for c in ("Mutant", "mutant", "structure", "proteoform"):
            if c in tm_df.columns:
                lab_col = c
                break
        if val_col and lab_col:
            # 4-decimal labels: near-identical folds differ only at the 3rd/4th
            # decimal; rounding to 2 dp made distinct structures look identical.
            out = viz.summary_bar(tm_df, lab_col, val_col,
                                  os.path.join(plot_dir, "tm_summary"),
                                  title="TM-score vs wild-type",
                                  value_title="TM-score (1.0 = identical fold)",
                                  text_format=".4f")
            if out.get("n"):
                panels["tm_summary"] = _html_iframe(out.get("html"),
                                                    height=max(340, 34 * out["n"] + 140))

    # ---- Antibody design: RF2 pAE-vs-RMSD scatter + pAE ranking bars ----
    ab_df = _antibody_scores_df(rd)
    if ab_df is not None:
        pae_max, rmsd_max = _antibody_thresholds(rd)
        out = viz.antibody_pae_rmsd_scatter(
            ab_df, os.path.join(plot_dir, "antibody_pae_rmsd"),
            pae_max=pae_max, rmsd_max=rmsd_max)
        if out.get("n"):
            panels["ab_scatter"] = _html_iframe(out.get("html"), height=600)
            populated.add("ab_scatter")
        out = viz.antibody_score_bars(
            ab_df, os.path.join(plot_dir, "antibody_ranking"), pae_max=pae_max)
        if out.get("n"):
            panels["ab_bars"] = _html_iframe(out.get("html"),
                                             height=max(360, 34 * out["n"] + 140))
            populated.add("ab_bars")

    return panels


# ---------------------------------------------------------------------------
# Refresh + view callbacks
# ---------------------------------------------------------------------------

def refresh_all_results(state):
    """Refresh all result subsections. Returns a tuple of all outputs."""
    rd = state.get("results_dir", "") if state else ""

    # Warning banners (visible in the Results tab):
    #  - structure provenance: prominent red banner when structures were built by
    #    ptm-psi grafting (backbone-identical -> TM=1.0), else a short green note.
    #  - pocket warning: shown in BOTH the Pocket Prediction and Pocket Drift
    #    accordions when binding_site_method == 'reference' or all volumes are 0.
    prov_banner = _structure_provenance_banner(rd)
    pocket_banner = _pocket_warning_banner(rd)
    prov_update = gr.update(value=prov_banner, visible=bool(prov_banner))
    pocket_pred_update = gr.update(value=pocket_banner, visible=bool(pocket_banner))
    pocket_drift_update = gr.update(value=pocket_banner, visible=bool(pocket_banner))

    step_csv = os.path.join(rd, "step_status.csv") if rd else None
    step_data = _load_csv(step_csv) if step_csv and os.path.exists(step_csv) else []

    # Mature-chain report (proteolytic processing) + cleaved-mutation warning.
    mature_csv = _find_csv(rd, "mature_chain_report") if rd else None
    mature_data = _load_csv(mature_csv) if mature_csv else []
    mature_warn_update = gr.update(value="", visible=False)
    if mature_csv:
        try:
            _mdf = pd.read_csv(mature_csv)
            _ncleave = int(pd.to_numeric(_mdf.get("n_cleaved_region", 0),
                                         errors="coerce").fillna(0).sum())
            if _ncleave > 0:
                _cms = _mdf.get("cleaved_mutations", pd.Series(dtype=str)).fillna("")
                _cms = "; ".join(sorted({s for s in _cms.astype(str) if s}))
                mature_warn_update = gr.update(
                    value=(f"**{_ncleave} mutation(s) lie in proteolytically "
                           f"cleaved regions** and were excluded from the "
                           f"structural steps (kept, flagged, in the tables): "
                           f"{_cms}"),
                    visible=True)
        except Exception:
            pass

    # Prefer the WT-vs-mutant TM file (Mutant, TM-score) over the pairwise
    # matrix; render TM-scores at 4 decimals so near-identical folds are
    # distinguishable in the table (the CSV keeps full precision).
    tm_csv = (_find_exact_csv_glob(rd, "tm_scores_*.csv", exclude="_all.csv")
              or _find_csv(rd, "tm_scores")) if rd else None
    tm_data = _load_csv_fmt(tm_csv, {"TM-score": 4, "tm_score": 4}) if tm_csv else []

    dock_csv = _find_csv(rd, "docking_summary") if rd else None
    dock_data = _load_csv(dock_csv) if dock_csv else []

    # ΔΔG table reindexed to the v3.7.0 schema (PTM columns included; older
    # 4-column result folders get the new columns filled with "").
    ddg_data = _ddg_table(rd)

    # Impact scores table: drop proteoforms with all-zero impact (see
    # _drop_zero_impact_rows). Falls back to an empty table when nothing is
    # informative rather than showing a misleading all-zero grid.
    impact_df_raw = _impact_scores_df(rd) if rd else None
    impact_df_filt, _, _impact_all_zero = _drop_zero_impact_rows(impact_df_raw)
    if impact_df_filt is not None and len(impact_df_filt):
        impact_data = [list(impact_df_filt.columns)] + impact_df_filt.values.tolist()
    else:
        impact_data = []

    # Pocket predictions table: prefer the exact predictions file (avoid
    # matching pocket_drift.csv) and drop structures with no real pocket.
    pocket_path = (_find_exact_csv(rd, "pocket_predictions.csv")
                   or _find_csv(rd, "pocket_pred")) if rd else None
    pocket_df_raw = _read_csv_df(pocket_path) if pocket_path else None
    pocket_df_filt, _, _ = _drop_zero_pocket_rows(pocket_df_raw)
    if pocket_df_filt is not None and len(pocket_df_filt):
        # Show volume/score at higher precision (small pockets can be < 1 A^3).
        pocket_data = _df_to_table_fmt(
            pocket_df_filt, {"volume": 3, "volume_A3": 3, "score": 4,
                             "center_x": 3, "center_y": 3, "center_z": 3})
    else:
        pocket_data = []

    # Pocket drift table: drop proteoforms with zero drift.
    drift_path = _find_csv(rd, "pocket_drift") if rd else None
    drift_df_raw = _read_csv_df(drift_path) if drift_path else None
    drift_df_filt, _, _ = _drop_zero_drift_rows(drift_df_raw)
    if drift_df_filt is not None and len(drift_df_filt):
        # Drift volume/displacement were previously rounded to 2 dp at the
        # source (erasing small real changes); now full precision on disk and
        # shown at 4 dp so sub-0.01 changes are visible instead of "0.00".
        drift_data = _df_to_table_fmt(
            drift_df_filt, {"volume_change": 4, "center_displacement_A": 4,
                            "center_displacement": 4, "wt_volume": 3,
                            "mut_volume": 3})
    else:
        drift_data = []

    # PDB files for structure viewer. Complexes (docked receptor+ligand and
    # designed antibody) are surfaced with clear labels and listed first so the
    # user can actually see the ligand / antibody (not just the apo receptor).
    pdb_files = _find_files(rd, ".pdb") if rd else []
    pdb_choices = _pdb_viewer_choices(pdb_files)

    # PCN structure dropdown: all non-WT structures (mutants + proteoforms)
    pcn_choices = []
    if rd:
        # Mutants from pdbs/tetramer or pdbs/monomer (exclude WT and reference PDB)
        for subdir in ("tetramer", "monomer"):
            pd = os.path.join(rd, "pdbs", subdir)
            if os.path.isdir(pd):
                for f in sorted(os.listdir(pd)):
                    if not f.endswith(".pdb"):
                        continue
                    name = f.replace(".pdb", "")
                    if not name.lower().startswith("wt") and not re.match(r"^[0-9][a-z0-9]{3}$", name.lower()):
                        pcn_choices.append(name)
                break
        # PTM-modified structures from ptms/ptms/. Note: names based on the
        # WT stem (e.g. wt-<uid>-tetramer_phospho_S123) are still MODIFIED
        # structures, so they are included here even though plain wt-* is
        # excluded above.
        ptm_dir = os.path.join(rd, "ptms", "ptms")
        if os.path.isdir(ptm_dir):
            for f in sorted(os.listdir(ptm_dir)):
                if f.endswith(".pdb"):
                    pcn_choices.append(f.replace(".pdb", ""))
        # Proteoforms from proteoforms/ directory
        pf_dir = os.path.join(rd, "proteoforms")
        if os.path.isdir(pf_dir):
            for f in sorted(os.listdir(pf_dir)):
                if f.endswith(".pdb"):
                    pcn_choices.append(f.replace(".pdb", ""))

    # ESM2/UMAP plots (Fix 2: show plot instead of table)
    esm_plots = []
    if rd:
        emb_dir = os.path.join(rd, "embeddings")
        if os.path.isdir(emb_dir):
            for f in sorted(os.listdir(emb_dir)):
                if "umap" in f.lower() and f.endswith(".png"):
                    esm_plots.append(os.path.join("embeddings", f))

    # ESM2/UMAP per-variant table (Fix 1: the space under the dropdown on the
    # left of the ESM2 tab was empty; now it holds the searchable UMAP
    # coordinates + variant classification read from embeddings/umap.csv).
    esm_table_data = _esm_umap_table(rd) if rd else []

    # Build interactive data plots (docking / pocket / summaries)
    panels = _build_result_plots(rd)
    populated = panels.get("_populated", set())

    # Docking panels are hidden when they hold no real data (issue: the right
    # panel used to show "No data yet"). Locally Boltz-2 needs a GPU, so the
    # confidence scatter is usually empty -> hide it rather than show a stub.
    dock_bars_update = gr.update(value=panels["dock_bars"],
                                 visible="dock_bars" in populated)
    dock_scatter_update = gr.update(value=panels["dock_scatter"],
                                    visible="dock_scatter" in populated)

    # Designed-ligand section (per-ligand druggability table; empty when the
    # ligand-design step did not run).
    ligand_data = _ligand_table(rd)

    # Antibody design section (table + RF2 score plots; hidden when the
    # antibody step did not run).
    ab_data = _antibody_table(rd)
    ab_scatter_update = gr.update(value=panels["ab_scatter"],
                                  visible="ab_scatter" in populated)
    ab_bars_update = gr.update(value=panels["ab_bars"],
                               visible="ab_bars" in populated)

    return (prov_update,                             # structure-provenance banner (top of Results)
            step_data,
            mature_warn_update,                      # cleaved-mutation warning (Mature chain)
            mature_data,                             # mature-chain report table
            tm_data, dock_data,
            gr.update(choices=pcn_choices),          # PCN structure dropdown
            gr.update(choices=esm_plots),            # ESM2 plot dropdown
            esm_table_data,                          # ESM2/UMAP per-variant table (left column)
            ddg_data, impact_data, pocket_data, drift_data,
            gr.update(choices=pdb_choices),          # PDB structure viewer dropdown
            panels["impact_summary"],                # impact plot
            panels["pocket_vol"],                    # pocket volume plot
            panels["pocket_drift"],                  # pocket drift plot
            panels["tm_summary"],                    # tm-score plot
            dock_bars_update,                        # docking affinity bars (hide if empty)
            dock_scatter_update,                     # docking boltz scatter (hide if empty)
            pocket_pred_update,                      # pocket-method warning (Pocket Prediction)
            pocket_drift_update,                     # pocket-method warning (Pocket Drift)
            ligand_data,                             # designed-ligand per-ligand table
            ab_data,                                 # antibody per-design table
            ab_scatter_update,                       # antibody pAE-vs-RMSD scatter
            ab_bars_update,                          # antibody pAE ranking bars
            state)


def view_pdb(pdb_rel, state):
    rd = state.get("results_dir", "") if state else ""
    if not pdb_rel or not rd:
        return "<p>Select a PDB file to view.</p>"
    full = os.path.join(rd, pdb_rel)
    return _pdb_to_html_viewer(full)


def view_esm_plot(plot_rel, state):
    """Fix 2: Show ESM2/UMAP plot instead of table."""
    rd = state.get("results_dir", "") if state else ""
    if not plot_rel or not rd:
        return None
    full = os.path.join(rd, plot_rel)
    if os.path.exists(full):
        return full
    return None


def _esm_umap_table(rd):
    """Return the ESM2/UMAP per-variant table (Fix 1: fill the empty space on
    the left of the ESM2 tab).

    Reads ``embeddings/umap.csv`` (columns x, y, label, subunit,
    pathogenicity). Coordinates are rounded to 3 dp for readability; the CSV on
    disk keeps full precision. Returns the ``[[headers], …]`` table format.
    """
    if not rd:
        return []
    umap_csv = os.path.join(rd, "embeddings", "umap.csv")
    if not os.path.exists(umap_csv):
        umap_csv = _find_exact_csv(rd, "umap.csv")
    if not umap_csv or not os.path.exists(umap_csv):
        return []
    try:
        df = pd.read_csv(umap_csv)
    except Exception:
        return []
    # friendlier column labels
    rename = {"label": "variant", "subunit": "subunit/UniProt",
              "pathogenicity": "classification", "x": "UMAP1", "y": "UMAP2"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # Reorder columns to match the ``esm_table`` gr.Dataframe headers exactly
    # (a gr.Dataframe with fixed headers is positional, so column order must
    # line up or values land under the wrong header). Any extra columns are
    # appended after the known ones; missing ones are simply skipped.
    preferred = ["variant", "subunit/UniProt", "classification", "UMAP1", "UMAP2"]
    ordered = [c for c in preferred if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]
    return _df_to_table_fmt(df, {"UMAP1": 3, "UMAP2": 3})


# ---------------------------------------------------------------------------
# Antibody design results (RFAntibody)
# ---------------------------------------------------------------------------

def _antibody_scores_df(rd):
    """Read ``antibody/scores.tsv`` (RFAntibody qvscorefile output, tab-separated).

    Returns the raw DataFrame (metric columns + ``tag``), or None when the
    antibody step did not run / produced no score table.
    """
    if not rd:
        return None
    path = os.path.join(rd, "antibody", "scores.tsv")
    if not os.path.exists(path):
        path = None
        for root, _, fnames in os.walk(rd):
            if "scores.tsv" in fnames and "antibody" in root.replace("\\", "/"):
                path = os.path.join(root, "scores.tsv")
                break
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as e:
        log.warning("Failed to read antibody scores %s: %s", path, e)
        return None
    return df if len(df) else None


def _antibody_thresholds(rd):
    """``(pae_max, rmsd_max)`` from antibody/design_context.json, else the
    AntibodyConfig defaults (10.0 / 2.0)."""
    pae_max, rmsd_max = 10.0, 2.0
    if rd:
        ctx = os.path.join(rd, "antibody", "design_context.json")
        if os.path.exists(ctx):
            try:
                with open(ctx) as f:
                    d = json.load(f)
                pae_max = float(d.get("rf2_pae_max", pae_max))
                rmsd_max = float(d.get("rmsd_max", rmsd_max))
            except Exception:
                pass
    return pae_max, rmsd_max


def _ligand_metrics_df(rd):
    """Read ``ligand_design/ligand_metrics.csv`` (DiffSBDD ligand design metrics).
    
    Returns the raw DataFrame, or None when the ligand step did not run / 
    produced no metrics table.
    """
    if not rd:
        return None
    path = os.path.join(rd, "ligand_design", "ligand_metrics.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.warning("Failed to read ligand metrics %s: %s", path, e)
        return None
    return df if len(df) else None


def _ligand_safe_name(s):
    """Filesystem-safe ligand name, same rule as the complex builder
    (``core/ligand_metrics.py``): non [A-Za-z0-9_.-] runs become '_'."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def _ligand_metrics_lookup(rd, struct_name):
    """Collect ligand design metrics (QED, SA, logP, MW, Lipinski violations)
    from ligand_design/ligand_metrics.csv for the viewed ligand complex.

    Matching is strictly per-ligand (see the matching code for details).
    Returns a list of ``(label, value)`` strings; missing metrics are omitted.
    """
    out = []
    if not rd or not struct_name:
        return out
    
    name = str(struct_name)
    base = os.path.basename(name).replace(".pdb", "")
    
    # Read ligand metrics
    ldf = _ligand_metrics_df(rd)
    if ldf is None:
        return out
    
    row = None
    # 1) Exact match on the unique complex stem ("<target>__<ligand>_complex").
    if "complex" in ldf.columns:
        s = ldf["complex"].astype(str)
        hit = ldf[(s == base) | (s == name)]
        if len(hit):
            row = hit.iloc[0]
    # 2) Fall back to the ligand name embedded in the complex stem, sanitized
    #    the same way the complex filenames are built.
    if row is None and "__" in base and "ligand" in ldf.columns:
        lig_part = re.sub(r"_complex$", "", base.split("__", 1)[1])
        s = ldf["ligand"].map(_ligand_safe_name)
        hit = ldf[s == lig_part]
        if len(hit):
            row = hit.iloc[0]
    # NOTE: the 'target' column is deliberately NOT used on its own — it is
    # identical for every ligand of a target, so matching it showed the FIRST
    # ligand's metrics for every complex (and for the apo target): the "same
    # QED for every ligand" bug. Viewing the apo target shows no ligand metrics.
    if row is None:
        return out
    for col, label, nd in (
        ("QED", "QED (drug-likeness)", 3),
        ("SA", "Synthetic accessibility", 2),
        ("logP", "logP", 2),
        ("lipinski_violations", "Lipinski violations", 0),
        ("MW", "Molecular weight (Da)", 1),
    ):
        if col in ldf.columns:
            try:
                v = float(row[col])
                out.append((label, f"{v:.0f}" if nd == 0 else f"{v:.{nd}f}"))
            except Exception:
                pass
    
    return out


_LIGAND_TABLE_COLS = ["ligand", "smiles", "QED", "SA", "logP", "MW", "HBD",
                      "HBA", "rotatable_bonds", "TPSA", "lipinski_violations"]


def _ligand_table(rd):
    """Per-ligand druggability table for the Designed Ligands section.

    One row per designed ligand (from ligand_design/ligand_metrics.csv),
    sorted best-first by QED. Returns the ``[[headers], …]`` table format.
    """
    df = _ligand_metrics_df(rd)
    if df is None:
        return []
    cols = [c for c in _LIGAND_TABLE_COLS if c in df.columns]
    df = df[cols]
    if "QED" in df.columns:
        df = df.sort_values("QED", ascending=False)
    return _df_to_table_fmt(df, {"QED": 3, "SA": 2, "logP": 2, "MW": 1,
                                 "TPSA": 1})


_DDG_TABLE_COLS = ["uniprot_id", "mutation", "ptm", "mimetic_mutation",
                   "structure", "ddg_kcal_mol", "method"]


def _ddg_table(rd):
    """ΔΔG table for the Results tab, reindexed to the v3.7.0 schema
    (``_DDG_TABLE_COLS``).

    Older result folders (4-column ddg_summary.csv without the PTM columns)
    get the new columns filled with "" so the fixed-headers table always
    lines up.
    """
    if not rd:
        return []
    df = _read_csv_df(_find_csv(rd, "ddg_summary") or _find_csv(rd, "ddg"))
    if df is None or not len(df):
        return []
    df = df.reindex(columns=_DDG_TABLE_COLS).fillna("")
    return _df_to_table_fmt(df, {"ddg_kcal_mol": 4})


def _antibody_table(rd):
    """Searchable per-design table for the Predicted Antibodies section.

    Columns: design, pAE_interaction, RMSD_CDR (Å), pLDDT, pass (vs the
    configured RF2 thresholds). Sorted best-first by interaction pAE.
    """
    df = _antibody_scores_df(rd)
    if df is None:
        return []
    tag_col, pae_col, rmsd_col, plddt_col = viz._antibody_score_cols(df)
    pae_max, rmsd_max = _antibody_thresholds(rd)
    out = pd.DataFrame()
    out["design"] = df[tag_col].astype(str)
    if pae_col is not None:
        out["pAE_interaction"] = pd.to_numeric(df[pae_col], errors="coerce")
    if rmsd_col is not None:
        out["RMSD_CDR (Å)"] = pd.to_numeric(df[rmsd_col], errors="coerce")
    if plddt_col is not None:
        out["pLDDT"] = pd.to_numeric(df[plddt_col], errors="coerce")
    if pae_col is not None and rmsd_col is not None:
        out["pass"] = [
            "yes" if (pd.notna(p) and pd.notna(r)
                      and p < pae_max and r < rmsd_max) else "no"
            for p, r in zip(out["pAE_interaction"], out["RMSD_CDR (Å)"])]
    if "pAE_interaction" in out.columns:
        out = out.sort_values("pAE_interaction", ascending=True)
    return _df_to_table_fmt(out, {"pAE_interaction": 2, "RMSD_CDR (Å)": 2,
                                  "pLDDT": 1})


def _structure_metrics_lookup(rd, struct_name):
    """Collect this structure's result metrics (TM-score vs WT, ΔΔG, pocket
    volume) from the result CSVs, matched by structure/mutant name. Returns a
    list of ``(label, value)`` strings; missing metrics are simply omitted."""
    out = []
    if not rd or not struct_name:
        return out
    name = str(struct_name)
    base = os.path.basename(name).replace(".pdb", "")

    def _match(df, cols):
        for col in cols:
            if col in df.columns:
                s = df[col].astype(str)
                hit = df[(s == base) | (s == name)]
                if len(hit):
                    return hit.iloc[0]
        return None

    # TM-score vs WT
    tm_csv = _find_exact_csv_glob(rd, "tm_scores_*.csv", exclude="_all.csv")
    tdf = _read_csv_df(tm_csv)
    if tdf is not None:
        row = _match(tdf, ["Mutant", "mutant", "structure"])
        if row is not None:
            for c in ("TM-score", "tm_score", "tmscore"):
                if c in tdf.columns:
                    try:
                        out.append(("TM-score vs WT", f"{float(row[c]):.4f}"))
                    except Exception:
                        pass
                    break

    # ΔΔG (stability)
    ddf = _read_csv_df(_find_csv(rd, "ddg_summary") or _find_csv(rd, "ddg"))
    if ddf is not None:
        row = _match(ddf, ["structure", "mutation", "mutant", "proteoform"])
        if row is None and "ptm" in ddf.columns:
            # PTM/proteoform structures: the FoldX mimetic tier records the
            # scored structure name in "structure" (tried above). PTM-step
            # files on mutant backgrounds
            # ("Mut_<uid>_<mut>-<tag>_<ptm_type>_<residue>") instead join on
            # the (mutation, ptm) pair embedded in the stem.
            ptm_mask = ddf["ptm"].fillna("").astype(str).str.len() > 0
            # A stem on a mutant background ("Mut_<uid>_<mut>-..." or
            # "Proteoform_<uid>_<mut>_...") must join the row carrying that
            # same mutation; a WT-background stem joins the mutation-less
            # PTM row. Without this split, a mutant-background stem would
            # silently pick up the WT-background PTM ΔΔG.
            stem_has_mut = base.lower().startswith(("mut_", "proteoform_"))
            for _, r in ddf[ptm_mask].iterrows():
                ptm = str(r["ptm"])
                mut = str(r.get("mutation", "") or "")
                if mut.lower() in ("", "nan", "none"):
                    mut = ""
                if ptm not in base:
                    continue
                if stem_has_mut:
                    if mut and mut in base:
                        row = r
                        break
                elif not mut:
                    row = r
                    break
        if row is not None:
            for c in ("ddg_kcal_mol", "ddG", "ddg"):
                if c in ddf.columns:
                    try:
                        out.append(("ΔΔG (kcal/mol)", f"{float(row[c]):.3f}"))
                    except Exception:
                        pass
                    break

    # Pocket volume
    pdf = _read_csv_df(_find_exact_csv(rd, "pocket_predictions.csv"))
    if pdf is not None:
        row = _match(pdf, ["structure", "proteoform"])
        if row is not None:
            vcol = "volume" if "volume" in pdf.columns else (
                "volume_A3" if "volume_A3" in pdf.columns else None)
            if vcol is not None:
                try:
                    vol = float(row[vcol])
                    failed = str(row.get("detector_failed", "")).lower() in ("true", "1")
                    out.append(("Pocket volume (Å³)",
                                "n/a (detector failed)" if failed or vol == 0
                                else f"{vol:.3f}"))
                except Exception:
                    pass

    # Designed antibody complex (RFAntibody): RF2 developability metrics from
    # antibody/scores.tsv, matched by design tag.
    adf = _antibody_scores_df(rd)
    if adf is not None:
        tag_col, pae_col, rmsd_col, plddt_col = viz._antibody_score_cols(adf)
        s = adf[tag_col].astype(str)
        hit = adf[(s == base) | (s == base.removesuffix("_best"))]
        if len(hit):
            row = hit.iloc[0]
            if pae_col is not None:
                try:
                    out.append(("RF2 interaction pAE",
                                f"{float(row[pae_col]):.2f}"))
                except Exception:
                    pass
            if rmsd_col is not None:
                try:
                    out.append(("Target-aligned CDR RMSD (Å)",
                                f"{float(row[rmsd_col]):.2f}"))
                except Exception:
                    pass
            if plddt_col is not None:
                try:
                    out.append(("RF2 pLDDT", f"{float(row[plddt_col]):.1f}"))
                except Exception:
                    pass

    # Designed ligand complex (DiffSBDD): druggability metrics from
    # ligand_design/ligand_metrics.csv, matched by complex name.
    out.extend(_ligand_metrics_lookup(rd, struct_name))
    return out


def structure_summary(pdb_rel, state):
    """Build a 'Structure summary' card (Fix 2: fill the empty space to the
    right of the 3D viewer).

    Shows chains + molecule types, ligands/hetero groups, an antibody-complex
    flag, atom/residue counts, and any per-structure result metrics (TM-score,
    ΔΔG, pocket volume) joined from the result CSVs.
    """
    rd = state.get("results_dir", "") if state else ""
    if not pdb_rel or not rd:
        return ("<div style='color:#666;padding:8px;'>Select a structure and click "
                "<b>View</b> to see its summary.</div>")
    full = os.path.join(rd, pdb_rel)
    if not os.path.exists(full):
        return "<div style='color:#b2182b;'>Structure file not found.</div>"
    with open(full) as f:
        pdb_text = f.read()
    info = _classify_pdb_contents(pdb_text, path_hint=pdb_rel)

    # counts
    n_atoms = sum(1 for l in pdb_text.splitlines() if l[:6].strip() in ("ATOM", "HETATM"))
    resids = set()
    for l in pdb_text.splitlines():
        if l[:6].strip() in ("ATOM", "HETATM"):
            resids.add((l[21:22], l[22:27]))
    n_res = len(resids)

    prot_chains = sorted(info["protein_chains"])
    ligs = sorted(info["ligand_resns"])

    # structure type headline
    if info["looks_like_antibody"]:
        ab_chains = [c for c in ("H", "L") if c in info["protein_chains"]]
        kind = "Antibody complex (" + " / ".join(ab_chains) + (
            " + antigen" if len(info["protein_chains"] - _AB_CHAINS) else "") + ")"
    elif info["has_ligand"] and info["has_protein"]:
        kind = "Receptor + ligand complex"
    elif info["has_protein"]:
        kind = "Protein structure"
    elif info["has_ligand"]:
        kind = "Ligand only"
    else:
        kind = "Unknown"

    rows = [
        ("Type", kind),
        ("Protein chains", f"{len(prot_chains)} ({', '.join(prot_chains) or '—'})"),
        ("Ligands / hetero", ", ".join(ligs) if ligs else "none"),
        ("Atoms", str(n_atoms)),
        ("Residues", str(n_res)),
    ]
    # joined result metrics for this structure
    base = os.path.basename(pdb_rel).replace(".pdb", "")
    struct_key = base[: -len("_complex")].split("__", 1)[0] if base.endswith("_complex") else base
    metrics = _structure_metrics_lookup(rd, struct_key)

    def _table(pairs):
        trs = "".join(
            f"<tr><td style='padding:3px 10px 3px 0;white-space:nowrap;'> {k}</td>"
            f"<td style='padding:3px 0;font-weight:600;'> {v}</td></tr>" for k, v in pairs)
        return f"<table style='border-collapse:collapse;font-size:0.92em;'>{trs}</table>"

    html_parts = [
        "<div style='border:1px solid #ccc;border-radius:8px;padding:12px 14px;'>",
        "<div style='font-weight:700;font-size:1.02em;margin-bottom:8px;'>"
        "Structure summary</div>",
        _table(rows),
    ]
    if metrics:
        html_parts.append(
            "<div style='margin-top:10px;padding-top:8px;border-top:1px solid #ddd;'>"
            "<div style='font-weight:700;font-size:0.95em;margin-bottom:6px;'>"
            "Result metrics</div>" + _table(metrics) + "</div>")
    else:
        html_parts.append(
            "<div style='margin-top:10px;padding-top:8px;border-top:1px solid #ddd;"
            "color:#888;font-size:0.85em;'>No per-structure result metrics found "
            "(run the pipeline steps to populate TM-score / ΔΔG / pocket volume).</div>")
    html_parts.append("</div>")
    return "".join(html_parts)

# ---------------------------------------------------------------------------
# GUI layout
# ---------------------------------------------------------------------------

def build_app():
    """Build the Gradio Blocks app and return it (without launching).

    Split out from :func:`launch` so external entry points (e.g. a Hugging
    Face Spaces ``app.py``) can obtain the ``gr.Blocks`` object, attach a
    request queue, and call ``.launch(...)`` with their own server / ZeroGPU
    settings. ``launch()`` remains the standard local entry point.
    """
    step_choices = list(STEP_REGISTRY.keys())

    with gr.Blocks(title="Proteoform Analyzer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Proteoform Analyzer\n"
                    "An AI-powered tool to analyze the effects of mutations, PTMs, and proteoforms on proteins in different oligomeric states. "
        )
        state = gr.State({"done": False, "results": [], "results_dir": ""})

        # ── Tab 1: Setup ──
        with gr.Tab("1. Setup"):
            # Preset selector: choose from 9 presets or custom
            preset_names = list(PRESETS.keys()) + ["Custom (manual)"]
            preset_selector = gr.Dropdown(
                choices=preset_names, value="Hemoglobin — Fast",
                label="Preset configuration",
                info="Select a preset to auto-fill all fields below, or choose Custom for manual entry.")
            with gr.Row():
                uniprot_ids = gr.Textbox(label="UniProt ID(s)", value="P69905,P68871",
                                         info="Comma-separated. 1 for homo, 2+ for hetero.")
                n_subunits = gr.Dropdown([1, 2, 3, 4, 6], value=4, label="Number of subunits")
                stoichiometry = gr.Textbox(value="2,2", label="Stoichiometry (comma-sep copy counts)")
            with gr.Row():
                proteoform_mode = gr.Radio(["single", "pairwise"], value="pairwise",
                                           label="Proteoform mode")
                proteoform_cap = gr.Slider(10, 500, value=100, step=10,
                                           label="Max proteoforms")
            with gr.Row():
                with gr.Column():
                    mutations_text = gr.Textbox(
                        label="Mutations (pipe-separated per subunit)",
                        value="",
                        info="e.g. 'D75A H87R D94H'.")
                    fetch_uniprot = gr.Button("Fetch from UniProt", variant="secondary")
                max_mutations = gr.Slider(0, 50, value=10, step=1, label="Max mutations")
                fetch_uniprot.click(
                    fn=_fetch_mutations_from_uniprot,
                    inputs=[uniprot_ids],
                    outputs=[mutations_text, max_mutations],
                )
            with gr.Row():
                structure_source = gr.Radio(["boltz2"], value="boltz2",
                                            label="Structure source",
                                            info="Boltz-2 folds structures from sequence "
                                                 "(hosted API → local binary → graft fallback).")
                local_pdb_id = gr.Textbox(value="1A3N",
                                          label="Reference PDB ID (binding-site alignment only)",
                                          info="Optional. Used only to align known binding "
                                               "sites; NOT a structure source.")
            with gr.Accordion("Mature chain (proteolytic processing)", open=False):
                gr.Markdown(
                    "Many proteins are synthesised as precursors and **proteolytically "
                    "matured** (signal peptide / propeptide / transit peptide removed): "
                    "the assembled complex contains only the mature chain (e.g. TTR "
                    "loses its 20-aa N-terminal signal peptide). When enabled, the "
                    "mature region is **auto-detected from UniProt features** and the "
                    "mature sequence is used for folding, embeddings, ddG and docking. "
                    "**Mutations and PTMs keep UniProt (precursor) numbering**; sites "
                    "in cleaved regions stay in the tables (flagged) but are excluded "
                    "from structural steps. Use the textbox to override the region "
                    "manually, e.g. `P02766:21-147; P69905:2-142` (1-based, inclusive, "
                    "UniProt numbering).")
                with gr.Row():
                    mature_auto = gr.Checkbox(
                        value=True, label="Auto-detect mature chain from UniProt",
                        info="Uses CHAIN/PEPTIDE features, or trims terminal "
                             "SIGNAL/PROPEP/TRANSIT peptides. Uncheck to always use "
                             "the full-length precursor sequence.")
                    mature_detect_btn = gr.Button("Preview detection from UniProt",
                                                  variant="secondary")
                mature_regions_text = gr.Textbox(
                    value="P69905:2-142; P68871:2-147",
                    label="Manual mature regions (override)",
                    info="Format: 'P02766:21-147; P69905:2-142'. Leave blank to use "
                         "auto-detection (or full-length when detection finds no "
                         "processing). Clear the box AND uncheck auto-detect to "
                         "force full-length precursor sequences.")
                mature_detect_df = gr.Dataframe(
                    value=pd.DataFrame(columns=MATURE_HEADERS),
                    headers=MATURE_HEADERS,
                    label="Detected mature regions (preview)",
                    interactive=False, wrap=True,
                    row_count=(3, "dynamic"), col_count=(5, "fixed"))
                mature_detect_btn.click(_detect_mature_regions,
                                        inputs=[uniprot_ids],
                                        outputs=[mature_detect_df])
            with gr.Accordion("PTM settings", open=False):
                gr.Markdown(
                    "Build PTM sites interactively: click **Load modifiable residues** "
                    "to list only residues that can carry a modelable PTM (from the "
                    "UniProt sequences above), pick a residue, then pick one of its "
                    "chemically compatible PTMs — PTMs observed in UniProt at that "
                    "residue are listed first. **Leave the table empty** to auto-fetch "
                    "observed PTMs from UniProt (capped at the *Max mutations* value); "
                    "if none are known, the PTM step is skipped automatically.")
                ptm_load_btn = gr.Button("Load modifiable residues from UniProt",
                                         variant="secondary")
                ptm_load_status = gr.Markdown(value="")
                with gr.Row():
                    ptm_residue_dd = gr.Dropdown(
                        choices=[], value=None, label="Residue", interactive=True,
                        info="Only residues that can carry a modelable PTM.")
                    ptm_type_dd = gr.Dropdown(
                        choices=[], value=None, label="PTM type", interactive=True,
                        info="Filtered to PTMs compatible with the selected residue.")
                with gr.Row():
                    ptm_add_btn = gr.Button("Add PTM", variant="primary")
                    ptm_clear_btn = gr.Button("Clear all", variant="secondary")
                ptm_pairs = gr.Dataframe(
                    value=pd.DataFrame(columns=PTM_PAIR_HEADERS),
                    headers=PTM_PAIR_HEADERS,
                    label="Selected PTM sites (rows are deletable via the row menu)",
                    interactive=True, wrap=True,
                    row_count=(1, "dynamic"), col_count=(3, "fixed"))
                ptm_load_btn.click(_list_modifiable_residues,
                                   inputs=[uniprot_ids],
                                   outputs=[ptm_residue_dd, ptm_load_status])
                ptm_residue_dd.change(_compatible_ptm_choices,
                                      inputs=[ptm_residue_dd],
                                      outputs=[ptm_type_dd])
                ptm_add_btn.click(_add_ptm_pair,
                                  inputs=[ptm_residue_dd, ptm_type_dd, ptm_pairs],
                                  outputs=[ptm_pairs])
                ptm_clear_btn.click(_clear_ptm_pairs, outputs=[ptm_pairs])
            with gr.Accordion("Engine choices", open=True):
                with gr.Row():
                    md_engine = gr.Radio(["openmm", "gromacs"], value="openmm", label="MD engine")
                    md_steps = gr.Slider(1000, 50000, value=5000, step=1000,
                                         label="MD production steps")
                with gr.Row():
                    docking_engine = gr.Radio(["vina", "boltz2", "diffdock"], value="vina",
                                              label="Docking engine",
                                              info="'boltz2' co-folds receptor+ligand "
                                                   "(confidence score, not kcal/mol).")
                    ligand_design_engine = gr.Radio(
                        ["diffsbdd", "boltzgen"], value="diffsbdd",
                        label="Ligand / binder design engine",
                        info="'diffsbdd' = small-molecule design; 'boltzgen' "
                             "= protein binder design (Boltz API → local install → none).")
                with gr.Row():
                    binding_site_method = gr.Radio(
                        ["auto", "reference", "p2rank", "alphasphere", "user"],
                        value="auto", label="Binding-site prediction method",
                        info="'reference' uses a fixed box from a known ligand site "
                             "(no pocket volume/drift — see warning in Results).")
                    ensemble_docking = gr.Checkbox(value=False, label="Ensemble docking (MD snapshots)")

                gr.Markdown(
                    "**Boltz-2 backend** — structure folding, co-folding docking "
                    "(`boltz2`), and binder design (`boltzgen`) all use "
                    "one backend, chosen in this order: **hosted API key → local "
                    "install → graft fallback** (folding only). Leave everything "
                    "blank to use the offline graft fallback. Local binder design "
                    "runs BoltzGen if installed.")
                with gr.Row():
                    boltz_api_key = gr.Textbox(
                        value="", label="Boltz API key",
                        info="Official api.boltz.bio key. Overrides $BOLTZ_API_KEY. "
                             "Enables folding, docking, and binder design via the "
                             "hosted API (no GPU needed).")
                with gr.Row():
                    boltz_prefer_local = gr.Checkbox(
                        value=False, label="Prefer local Boltz",
                        info="Use a local 'boltz'/'boltzgen' install before the API.")
                    boltz_allow_graft = gr.Checkbox(
                        value=True, label="Allow graft fallback",
                        info="If no folding backend exists, build backbone-identical "
                             "structures by PTM-Psi grafting (TM-score = 1.0; no "
                             "structural/pocket/docking signal). Uncheck to hard-skip.")

                gr.Markdown(
                    "**FoldX (PTM ΔΔG)** — optional, academic-licensed. When a "
                    "binary is provided, PTMs are scored for stability (ΔΔG) as "
                    "mimetic substitutions (phospho SER/THR→GLU, acetyl "
                    "LYS→GLN) on the WT background and on each mutation × PTM "
                    "proteoform. Leave blank to skip PTM stability scoring.")
                with gr.Row():
                    foldx_binary = gr.Textbox(
                        value="", label="FoldX binary path",
                        info="Path to the FoldX executable. Overrides "
                             "$FOLDX_BINARY. rotabase.txt is taken from the "
                             "binary's directory (or --foldx-rotabase).")

            with gr.Accordion("Antibody design (RFAntibody; opt-in)", open=False):
                gr.Markdown(
                    "De novo antibody/nanobody design against a binding site "
                    "(RFdiffusion -> ProteinMPNN -> RF2). Enabling this adds the "
                    "**antibody** step automatically. Requires a **local RFAntibody "
                    "install** (GPU strongly recommended); set the checkout path via "
                    "`config.antibody.local_rfantibody_dir` (or the `--antibody-rfantibody-dir` "
                    "CLI flag). Without it, the step skips cleanly with instructions.")
                with gr.Row():
                    antibody_enabled = gr.Checkbox(value=True,
                                                   label="Enable antibody design")
                    antibody_framework = gr.Radio(["nanobody", "scfv"], value="nanobody",
                                                  label="Framework")
                    antibody_num_designs = gr.Slider(1, 100, value=20, step=1,
                                                     label="Number of designs")
                with gr.Row():
                    antibody_hotspot_source = gr.Radio(
                        ["user", "bcell", "mhc_i", "mhc_ii"], value="bcell",
                        label="Hotspot source",
                        info="'bcell' = predicted B-cell epitopes (default); "
                             "'user' = residues at right; 'mhc_i'/'mhc_ii' "
                             "require an external tool (stubs).")
                    antibody_hotspots = gr.Textbox(
                        value="", label="Target hotspot residues (comma-sep)",
                        info="For source 'user', e.g. '305,456' or 'A305,A456'.")

            selected_steps = gr.CheckboxGroup(step_choices,
                                              value=step_choices,
                                              label="Pipeline steps to run")
                    
            # Wire preset selector to populate all widgets
            preset_selector.change(
                _load_preset,
                inputs=[preset_selector],
                outputs=[uniprot_ids, n_subunits, stoichiometry, proteoform_mode,
                         proteoform_cap, mutations_text, max_mutations, structure_source,
                         local_pdb_id, ptm_pairs, md_engine, md_steps,
                         docking_engine, ligand_design_engine, binding_site_method,
                         ensemble_docking, selected_steps,
                         mature_auto, mature_regions_text, foldx_binary])

        # ── Tab 2: Run ──
        with gr.Tab("2. Run"):
            run_btn = gr.Button("Run pipeline", variant="primary")
            log_out = gr.Textbox(label="Live log", lines=20, max_lines=40,
                                 interactive=False)
            status_table = gr.Dataframe(
                headers=["status", "step", "message", "elapsed_s", "n_outputs"],
                label="Step status", wrap=True)
            run_btn.click(
                run_from_gui,
                inputs=[preset_selector,
                        uniprot_ids, n_subunits, stoichiometry, proteoform_mode, proteoform_cap,
                        mutations_text, max_mutations, structure_source, local_pdb_id,
                        ptm_pairs, md_engine, md_steps,
                        docking_engine, ligand_design_engine, binding_site_method,
                        ensemble_docking, selected_steps,
                        antibody_enabled, antibody_framework, antibody_hotspot_source,
                        antibody_hotspots, antibody_num_designs,
                        boltz_api_key, boltz_prefer_local,
                        boltz_allow_graft,
                        mature_auto, mature_regions_text,
                        foldx_binary,
                        state],
                outputs=[log_out, status_table, state])

        # ── Tab 3: Results ──
        with gr.Tab("3. Results"):
            gr.Markdown("## Results\nBrowse results by type. Use the search box in each "
                        "table to filter. Click 'Refresh' after a run to populate.")
            # Structure-provenance banner: hidden until Refresh detects how the
            # structures were produced. Turns into a prominent red warning when
            # the graft fallback was used (backbone-identical -> TM=1.0).
            provenance_banner = gr.Markdown(value="", visible=False)
            refresh_btn = gr.Button("Refresh all results", variant="secondary")

            with gr.Accordion("Step Status Summary", open=True):
                step_status_df = gr.Dataframe(
                    headers=["status", "step", "message", "elapsed_s", "n_outputs"],
                    label="Pipeline step status", wrap=True,
                    interactive=False, row_count=(10, "dynamic"), show_search=True)

            with gr.Accordion("Mature chain (proteolytic processing)", open=False):
                gr.Markdown("Mature-chain regions resolved for this run (UniProt "
                            "precursor numbering). Mutations/PTMs in cleaved regions "
                            "are kept in the tables but excluded from structural steps.")
                mature_warn = gr.Markdown(value="", visible=False)
                mature_df = gr.Dataframe(
                    headers=["uniprot_id", "precursor_length", "mature_start",
                             "mature_end", "mature_length", "source",
                             "n_mutations", "n_cleaved_region",
                             "cleaved_mutations", "warning"],
                    label="Mature chain report (searchable)", wrap=True,
                    interactive=False, row_count=(5, "dynamic"), show_search=True)

            with gr.Accordion("Proteoform Impact Scores", open=False):
                gr.Markdown("Composite impact score ranking across all proteoforms. "
                            "Higher score = more impactful variant.")
                impact_plot = gr.HTML(
                    value="<p>Run the pipeline and click Refresh.</p>",
                    label="Impact ranking (interactive)")
                impact_df = gr.Dataframe(
                    headers=["proteoform", "structural", "binding", "dynamics", "network", "sequence", "composite"],
                    label="Impact scores (searchable)", wrap=True,
                    interactive=False, row_count=(20, "dynamic"), show_search=True)

            with gr.Accordion("Pocket Prediction", open=False):
                gr.Markdown("Predicted binding pockets (method, center, volume, score). "
                            "Structures where every pocket detector failed are omitted "
                            "from the plot; if any are shown they appear in grey with a "
                            "volume of 0 (no pocket was detected, so no volume could be "
                            "measured).")
                # Warning banner (hidden until Refresh): shown when the
                # binding-site method is 'reference' (fixed box -> volume/drift
                # always 0) or when no detector produced a real volume.
                pocket_pred_warning = gr.Markdown(value="", visible=False)
                pocket_plot = gr.HTML(
                    value="<p>Run the pipeline and click Refresh.</p>",
                    label="Pocket volume (interactive)")
                pocket_df = gr.Dataframe(
                    headers=["structure", "method", "center_x", "center_y", "center_z", "volume", "score"],
                    label="Pocket predictions (searchable)", wrap=True,
                    interactive=False, row_count=(20, "dynamic"), show_search=True)

            with gr.Accordion("Pocket Drift Analysis", open=False):
                gr.Markdown("Pocket property changes vs WT across proteoforms. "
                            "X = volume change (Å³), Y = pocket-centre displacement (Å).")
                # Same warning as Pocket Prediction (reference method / all-zero
                # volumes -> drift is identically 0 and cannot change).
                pocket_drift_warning = gr.Markdown(value="", visible=False)
                drift_plot = gr.HTML(
                    value="<p>Run the pipeline and click Refresh.</p>",
                    label="Pocket drift (interactive)")
                drift_df = gr.Dataframe(
                    headers=["proteoform", "volume_change", "center_displacement", "residue_jaccard"],
                    label="Pocket drift (searchable)", wrap=True,
                    interactive=False, row_count=(20, "dynamic"), show_search=True)

            with gr.Accordion("TM-score (structural comparison)", open=False):
                tm_plot = gr.HTML(
                    value="<p>Run the pipeline and click Refresh.</p>",
                    label="TM-score (interactive)")
                tm_df = gr.Dataframe(
                    headers=["Mutant", "TM-score"],
                    label="TM-scores (searchable)", wrap=True,
                    interactive=False, row_count=(20, "dynamic"), show_search=True)

            with gr.Accordion("Docking (binding affinities)", open=False):
                gr.Markdown("**Vina affinity** (bar chart, lower = stronger) and "
                            "**Boltz-2 confidence** (pTM vs ipTM scatter).")
                with gr.Row():
                    # Hidden until Refresh finds real data (see refresh_all_results).
                    dock_bars_plot = gr.HTML(
                        value="<p>Vina docking: run and Refresh.</p>",
                        label="Vina affinity (interactive)", visible=False)
                    dock_scatter_plot = gr.HTML(
                        value="<p>Boltz-2 docking: run and Refresh.</p>",
                        label="Boltz-2 pTM vs ipTM (interactive)", visible=False)
                dock_df = gr.Dataframe(
                    headers=["structure", "ligand", "affinity_kcal_mol"],
                    label="Docking results (searchable)", wrap=True,
                    interactive=False, row_count=(20, "dynamic"), show_search=True)

            with gr.Accordion("Molecular Dynamics (RMSD / RMSF)", open=False):
                gr.Markdown("**Interactive trajectory overlay (all structures).** "
                            "Choose RMSD or RMSF and click Overlay. Every structure "
                            "with MD output is drawn as its own trace — show/hide "
                            "individual structures directly from the Plotly legend. "
                            "A static SVG/PNG is also saved.")
                with gr.Row():
                    md_overlay_kind = gr.Radio(choices=["RMSD", "RMSF"], value="RMSD",
                                               label="Metric")
                    overlay_btn = gr.Button("Overlay", variant="primary")
                md_overlay_plot = gr.HTML(
                    value="<p>Click Overlay to plot RMSD/RMSF for all structures.</p>",
                    label="Interactive overlay")
                with gr.Row():
                    md_overlay_file = gr.File(label="Static plot (SVG)", interactive=False)
                    md_overlay_status = gr.Textbox(label="Status", interactive=False)
                overlay_btn.click(overlay_md_callback,
                                  inputs=[md_overlay_kind, state],
                                  outputs=[md_overlay_plot, md_overlay_file, md_overlay_status])

            # ── Fix 3: PCN interactive viewers (replaces table) ──
            with gr.Accordion("Protein Contact Networks (interactive)", open=False):
                gr.Markdown(
                    "**Residue centrality** (left) and **community detection** "
                    "(right), each mapped onto the 3D structure. Centrality can be "
                    "shown as a **raw viridis colormap** (works for WT and mutants) "
                    "or as **\u0394 vs WT** (blue = decreased, red = increased). The "
                    "**bar plot** highlights the top-10 residues with the largest "
                    "|\u0394 centrality| (name+id, signed). Communities are coloured "
                    "by community id for the selected structure.")
                with gr.Row():
                    pcn_struct = gr.Dropdown(label="Structure", choices=[], interactive=True,
                                             info="WT or any mutant/proteoform")
                    pcn_measure = gr.Dropdown(
                        choices=[("Betweenness", "betweenness"), ("Closeness", "closeness"),
                                 ("Degree", "degree_c"), ("Eigenvector", "eigenvector_c")],
                        value="betweenness", label="Centrality measure")
                    pcn_algo = gr.Dropdown(
                        choices=[("Louvain", "louvain"), ("Leiden", "leiden"), ("Infomap", "infomap")],
                        value="louvain", label="Community algorithm")
                with gr.Row():
                    pcn_view_mode = gr.Radio(
                        choices=[("Raw centrality (viridis)", "raw"),
                                 ("\u0394 vs WT (diverging)", "delta")],
                        value="raw", label="Centrality view")
                    pcn_btn = gr.Button("Visualize", variant="primary")
                with gr.Row():
                    pcn_centrality_view = gr.HTML(
                        value="<p>Select a structure and click 'Visualize'.</p>",
                        label="Centrality (colormap on structure)")
                    pcn_community_view = gr.HTML(
                        value="<p>Select a structure and click 'Visualize'.</p>",
                        label="Community detection")
                try:
                    pcn_delta_bars = gr.Image(
                        label="Top-10 residues by |\u0394 centrality| vs WT "
                              "(red = increase, blue = decrease)",
                        interactive=False, show_download_button=True)
                    pcn_legend = gr.Textbox(label="Legend", interactive=False, lines=2)
                except Exception as e:
                    pcn_delta_bars = gr.Image(
                        label="Top-10 residues by |\u0394 centrality| vs WT "
                              "(red = increase, blue = decrease)",
                        interactive=False, buttons=["download"])
                    pcn_legend = gr.Textbox(label="Legend", interactive=False, lines=2)
                pcn_btn.click(pcn_visualize,
                              inputs=[pcn_struct, pcn_measure, pcn_algo, state, pcn_view_mode],
                              outputs=[pcn_centrality_view, pcn_community_view,
                                       pcn_legend, pcn_delta_bars])

            # ── Fix 2: ESM2 plot (replaces table) ──
            with gr.Accordion("ESM2 + UMAP (variant classification)", open=False):
                gr.Markdown("ESM2 embeddings projected to 2D via UMAP. Each point is a "
                            "variant; points that cluster together have similar sequences.")
                with gr.Row():
                    # Left column: plot selector + the per-variant UMAP table
                    # (Fix 1: the space under the dropdown was empty; now it
                    # holds the searchable UMAP coordinates + classification).
                    with gr.Column(scale=1):
                        esm_plot_selector = gr.Dropdown(label="Select UMAP plot",
                                                        choices=[], interactive=True)
                        esm_table = gr.Dataframe(
                            headers=["variant", "subunit/UniProt", "classification",
                                     "UMAP1", "UMAP2"],
                            label="UMAP coordinates & classification (searchable)",
                            wrap=True, interactive=False,
                            row_count=(12, "dynamic"), show_search=True)
                    # Right column: the UMAP projection image
                    with gr.Column(scale=1):
                        esm_plot_view = gr.Image(label="UMAP projection", height=450)
                esm_plot_selector.change(view_esm_plot, inputs=[esm_plot_selector, state],
                                         outputs=[esm_plot_view])

            with gr.Accordion("DeltaDeltaG (stability prediction)", open=False):
                gr.Markdown(
                    "Mutations are scored by ThermoMPNN (local, GPU) or ESM2 "
                    "zero-shot (fallback). **PTMs** are scored by FoldX as "
                    "mimetic substitutions (phospho SER/THR→E, acetyl LYS→Q; "
                    "`method = foldx_mimetic`) on the WT background and on each "
                    "mutation × PTM proteoform — requires a FoldX binary "
                    "(Setup → Engine choices).")
                ddg_df = gr.Dataframe(
                    headers=["uniprot_id", "mutation", "ptm", "mimetic_mutation",
                             "structure", "ddg_kcal_mol", "method"],
                    label="ΔΔG results (searchable)", wrap=True,
                    interactive=False, row_count=(20, "dynamic"), show_search=True)

            with gr.Accordion("Designed Ligands (DiffSBDD)", open=False):
                gr.Markdown(
                    "Per-ligand druggability metrics for the designed small "
                    "molecules: **QED** (drug-likeness), **SA** (synthetic "
                    "accessibility), **logP**, **MW**, H-bond donors/acceptors, "
                    "rotatable bonds, **TPSA**, and Lipinski rule-of-5 "
                    "violations. View the target+ligand complexes in the 3D "
                    "Structure Viewer (labelled `[designed ligand · DiffSBDD]`).")
                ligand_df = gr.Dataframe(
                    headers=["ligand", "smiles", "QED", "SA", "logP", "MW",
                             "HBD", "HBA", "rotatable_bonds", "TPSA",
                             "lipinski_violations"],
                    label="Designed ligands (searchable)", wrap=True,
                    interactive=False, row_count=(20, "dynamic"),
                    show_search=True)

            with gr.Accordion("Predicted Antibodies (RFAntibody)", open=False):
                gr.Markdown(
                    "Designed target+antibody complexes scored by RoseTTAFold2. "
                    "**pAE** (predicted aligned error on the target–antibody "
                    "interface) and **RMSD** (target-aligned CDR deviation) are "
                    "filtered against the configured thresholds "
                    "(`rf2_pae_max` / `rmsd_max`); passing designs sit in the "
                    "lower-left of the scatter. View the complexes in the 3D "
                    "Structure Viewer (labelled `[designed antibody]`).")
                antibody_df = gr.Dataframe(
                    headers=["design", "pAE_interaction", "RMSD_CDR (Å)",
                             "pLDDT", "pass"],
                    label="Per-design RF2 scores (searchable)", wrap=True,
                    interactive=False, row_count=(20, "dynamic"),
                    show_search=True)
                with gr.Row():
                    ab_scatter_plot = gr.HTML(
                        value="<p>Run the pipeline with antibody design enabled, "
                              "then click Refresh.</p>")
                    ab_bars_plot = gr.HTML(value="")

            with gr.Accordion("3D Structure Viewer (PDB)", open=False):
                gr.Markdown(
                    "Docked receptor+ligand complexes and designed antibody "
                    "complexes are listed first (labelled `[docked …]` / "
                    "`[designed antibody]`) so you can see the **ligand (licorice)** "
                    "or the **antibody (coloured by chain)**, not just the apo "
                    "receptor.")
                with gr.Row():
                    pdb_selector = gr.Dropdown(label="Select PDB structure", choices=[],
                                               interactive=True, scale=3)
                    view_pdb_btn = gr.Button("View", variant="secondary", scale=1)
                with gr.Row():
                    # Left: the 3D viewer. Right: a structure-summary card
                    # (Fix 2: the space to the right of the viewer was empty).
                    with gr.Column(scale=3):
                        pdb_viewer = gr.HTML(
                            value="<p>Select a PDB file and click 'View' to load "
                                  "the 3D viewer.</p>",
                            label="3D Structure Viewer")
                    with gr.Column(scale=2):
                        pdb_summary = gr.HTML(
                            value="""
                            <div style='
                                color:#0b3d91;
                                background:#eaf2ff;
                                padding:12px;
                                border:1px solid #bcd0f7;
                                border-radius:8px;
                                font-weight:500;
                            '>
                                Structure summary appears here after you click <b>View</b>.
                            </div>
                            """,
                            label="Structure summary"
                        )
                # One click updates both the viewer and the summary card.
                view_pdb_btn.click(view_pdb, inputs=[pdb_selector, state],
                                   outputs=[pdb_viewer])
                view_pdb_btn.click(structure_summary, inputs=[pdb_selector, state],
                                   outputs=[pdb_summary])

            # Fix 4: "All Result Files" section removed

            refresh_btn.click(
                refresh_all_results,
                inputs=[state],
                outputs=[provenance_banner,    # structure-provenance banner (top)
                         step_status_df,
                         mature_warn,          # cleaved-mutation warning (Mature chain)
                         mature_df,            # mature-chain report table
                         tm_df, dock_df,
                         pcn_struct,           # PCN structure dropdown
                         esm_plot_selector,    # ESM2 plot dropdown
                         esm_table,            # ESM2/UMAP per-variant table (left column)
                         ddg_df, impact_df, pocket_df, drift_df,
                         pdb_selector,         # PDB structure viewer dropdown
                         impact_plot,          # impact interactive plot
                         pocket_plot,          # pocket volume interactive plot
                         drift_plot,           # pocket drift interactive plot
                         tm_plot,              # tm-score interactive plot
                         dock_bars_plot,       # docking affinity bars
                         dock_scatter_plot,    # docking boltz scatter
                         pocket_pred_warning,  # pocket-method warning (Pocket Prediction)
                         pocket_drift_warning, # pocket-method warning (Pocket Drift)
                         ligand_df,            # designed-ligand per-ligand table
                         antibody_df,          # antibody per-design table
                         ab_scatter_plot,      # antibody pAE-vs-RMSD scatter
                         ab_bars_plot,         # antibody pAE ranking bars
                         state])

    return app


def launch(**launch_kwargs):
    """Build and launch the Gradio Blocks app (standard local entry point).

    Any keyword arguments are forwarded to ``Blocks.launch`` so callers can
    override the host/port/share/SSR settings. A request queue is always
    enabled (needed for the long-running pipeline and for Hugging Face
    ZeroGPU). Defaults preserve the original local behaviour.
    """
    app = build_app()
    app.queue()
    app.launch(**launch_kwargs)


if __name__ == "__main__":
    launch()