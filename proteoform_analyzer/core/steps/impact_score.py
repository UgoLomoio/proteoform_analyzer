"""Step: composite proteoform impact score.

Reads all prior step outputs and computes a composite impact score that ranks
proteoforms by overall structural/binding/dynamic/network/sequence impact.

Components (each normalized 0-1):
  - structural: |1.0 - TM-score| (0 = no change, 1 = completely different)
  - binding: |affinity_WT - affinity_mut| / max_shift
  - dynamics: RMSD_final / max_RMSD
  - network: |n_coms_WT - n_coms_mut| / max_change
  - sequence: euclidean(emb_WT, emb_mut) / max_dist

Composite = weighted sum (default equal-ish weights, configurable).
"""
from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd

from ..pipeline import StepResult

log = logging.getLogger("proteoform_analyzer.impact_score")


def _safe_load_csv(path):
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def _wt_reference_names(config) -> set[str]:
    """Lower-cased names that denote the canonical WT reference structure.

    Current naming is ``wt-<uid>-<tag>`` (WT folded once from the first
    subunit); legacy plain ``wt`` is also accepted. PTM-on-WT structures
    (``wt-<uid>-<tag>_<ptm_type>_<RESIDUE>``) and proteoforms
    (``Proteoform_*``) are NOT references — they are scored proteoforms.
    """
    tag = "monomer" if getattr(config, "is_monomer", False) else "tetramer"
    names = {"wt"}
    for uid in (getattr(config, "uniprot_ids", None) or []):
        names.add(f"wt-{uid}-{tag}".lower())
        names.add(f"wt-{uid}".lower())
    return names


def _normalize(series, invert=False):
    """Normalize a pandas Series to 0-1 range."""
    if series is None or series.empty or series.max() == series.min():
        return pd.Series(0.0, index=series.index if series is not None else [])
    norm = (series - series.min()) / (series.max() - series.min())
    return 1.0 - norm if invert else norm


def run_impact_score(config, paths: dict) -> StepResult:
    """Compute composite proteoform impact scores."""
    rd = paths["results"]
    weights = config.impact_weights
    # Normalize weights to sum to 1
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}

    # --- Load all available result CSVs ---
    # TM-scores
    tm_csv = None
    for f in os.listdir(paths.get("tmalign", rd)):
        if f.endswith(".csv") and "tm_scores" in f and "all" not in f:
            tm_csv = os.path.join(paths["tmalign"], f)
            break
    tm_df = _safe_load_csv(tm_csv)

    # Docking
    dock_dir = os.path.join(paths.get("docking", rd), config.docking_engine)
    dock_csv = os.path.join(dock_dir, "docking_summary.csv") if os.path.isdir(dock_dir) else None
    dock_df = _safe_load_csv(dock_csv)

    # MD
    md_csv = os.path.join(paths.get("molecular_dynamics", rd), "md_summary.csv")
    md_df = _safe_load_csv(md_csv)

    # PCN
    pcn_csv = os.path.join(paths.get("pcn_outputs", rd), "pcn_summary.csv")
    pcn_df = _safe_load_csv(pcn_csv)

    # ESM2
    esm_npy = os.path.join(paths.get("embeddings", rd), "embeddings.npy")
    esm_labels = os.path.join(paths.get("embeddings", rd), "embedding_labels.csv")
    esm_df = _safe_load_csv(esm_labels)
    esm_emb = None
    if os.path.exists(esm_npy):
        try:
            esm_emb = np.load(esm_npy)
        except Exception:
            pass

    # Pocket drift
    pocket_dir = paths.get("pockets", os.path.join(rd, "pockets"))
    drift_csv = os.path.join(pocket_dir, "pocket_drift.csv")
    drift_df = _safe_load_csv(drift_csv)

    # --- Collect all proteoform names ---
    all_names = set()
    if tm_df is not None and "Mutant" in tm_df.columns:
        all_names.update(tm_df["Mutant"].tolist())
    if dock_df is not None and "structure" in dock_df.columns:
        all_names.update(dock_df["structure"].tolist())
    if md_df is not None and "name" in md_df.columns:
        all_names.update(md_df["name"].tolist())
    if pcn_df is not None and "structure" in pcn_df.columns:
        all_names.update(pcn_df["structure"].tolist())
    if drift_df is not None and "proteoform" in drift_df.columns:
        all_names.update(drift_df["proteoform"].tolist())

    # Remove only the true WT reference (exact match) and the hardcoded
    # reference-PDB controls. PTM-on-WT structures ("wt-..._<ptm>_<residue>")
    # and proteoforms ("Proteoform_*") ARE scored — they are the point of the
    # PTM/proteoform analysis.
    wt_names = _wt_reference_names(config)
    all_names = sorted(n for n in all_names
                       if (n or "").strip().lower() not in wt_names
                       and (n or "").strip().lower() not in ("1a3n", "1f41", "1aie"))
    if not all_names:
        return StepResult("impact_score", "skipped", "No proteoforms to score")

    # --- Build per-proteoform score table ---
    rows = []

    # Get WT reference values
    wt_tm = 1.0
    wt_affinity = None
    wt_rmsd = 0.0
    wt_ncoms = None
    wt_emb = None

    # WT reference rows: exact canonical-WT match only — a PTM-on-WT or
    # proteoform structure must never silently become the baseline.
    if tm_df is not None and "Mutant" in tm_df.columns and "TM-score" in tm_df.columns:
        wt_rows = tm_df[tm_df["Mutant"].str.lower().isin(wt_names)]
        if not wt_rows.empty:
            wt_tm = float(wt_rows.iloc[0]["TM-score"])

    if dock_df is not None and "structure" in dock_df.columns and "affinity_kcal_mol" in dock_df.columns:
        wt_rows = dock_df[dock_df["structure"].str.lower().isin(wt_names)]
        if not wt_rows.empty:
            wt_affinity = float(wt_rows.iloc[0]["affinity_kcal_mol"])

    if md_df is not None and "name" in md_df.columns and "rmsd_final" in md_df.columns:
        wt_rows = md_df[md_df["name"].str.lower().isin(wt_names)]
        if not wt_rows.empty:
            try:
                wt_rmsd = float(wt_rows.iloc[0]["rmsd_final"])
            except (ValueError, TypeError):
                pass

    if pcn_df is not None and "structure" in pcn_df.columns:
        wt_rows = pcn_df[pcn_df["structure"].str.lower().isin(wt_names)]
        if not wt_rows.empty:
            for col in ["n_coms_louvain", "n_coms_leiden"]:
                if col in pcn_df.columns:
                    try:
                        wt_ncoms = float(wt_rows.iloc[0][col])
                        break
                    except (ValueError, TypeError):
                        pass

    if esm_df is not None and esm_emb is not None and "label" in esm_df.columns:
        wt_rows = esm_df[esm_df["label"].str.upper() == "WT"]
        if not wt_rows.empty:
            wt_idx = wt_rows.index[0]
            if wt_idx < len(esm_emb):
                wt_emb = esm_emb[wt_idx]

    # Compute max values for normalization
    max_affinity_shift = 1.0
    if dock_df is not None and wt_affinity is not None and "affinity_kcal_mol" in dock_df.columns:
        affs = pd.to_numeric(dock_df["affinity_kcal_mol"], errors="coerce").dropna()
        if len(affs) > 0:
            max_affinity_shift = max(abs(affs.max() - wt_affinity), abs(affs.min() - wt_affinity), 0.1)

    max_rmsd = 1.0
    if md_df is not None and "rmsd_final" in md_df.columns:
        rmsds = pd.to_numeric(md_df["rmsd_final"], errors="coerce").dropna()
        if len(rmsds) > 0:
            max_rmsd = max(rmsds.max(), 0.1)

    max_ncoms_change = 1.0
    if pcn_df is not None and wt_ncoms is not None:
        for col in ["n_coms_louvain", "n_coms_leiden"]:
            if col in pcn_df.columns:
                vals = pd.to_numeric(pcn_df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    max_ncoms_change = max(abs(vals.max() - wt_ncoms), abs(vals.min() - wt_ncoms), 1.0)
                    break

    max_emb_dist = 1.0
    if esm_emb is not None and wt_emb is not None:
        dists = [np.linalg.norm(emb - wt_emb) for emb in esm_emb]
        max_emb_dist = max(max(dists), 0.1) if dists else 1.0

    for name in all_names:
        scores = {}

        # Structural: |1.0 - TM-score|
        if tm_df is not None and "Mutant" in tm_df.columns and "TM-score" in tm_df.columns:
            row = tm_df[tm_df["Mutant"] == name]
            if not row.empty:
                tm = float(row.iloc[0]["TM-score"])
                scores["structural"] = abs(1.0 - tm)
            else:
                scores["structural"] = 0.0
        else:
            scores["structural"] = 0.0

        # Binding: |affinity_WT - affinity_mut| / max_shift
        if dock_df is not None and "structure" in dock_df.columns and "affinity_kcal_mol" in dock_df.columns and wt_affinity is not None:
            row = dock_df[dock_df["structure"] == name]
            if not row.empty:
                try:
                    aff = float(row.iloc[0]["affinity_kcal_mol"])
                    scores["binding"] = abs(wt_affinity - aff) / max_affinity_shift
                except (ValueError, TypeError):
                    scores["binding"] = 0.0
            else:
                scores["binding"] = 0.0
        else:
            scores["binding"] = 0.0

        # Dynamics: RMSD_final / max_RMSD
        if md_df is not None and "name" in md_df.columns and "rmsd_final" in md_df.columns:
            row = md_df[md_df["name"] == name]
            if not row.empty:
                try:
                    rmsd = float(row.iloc[0]["rmsd_final"])
                    scores["dynamics"] = rmsd / max_rmsd
                except (ValueError, TypeError):
                    scores["dynamics"] = 0.0
            else:
                scores["dynamics"] = 0.0
        else:
            scores["dynamics"] = 0.0

        # Network: |n_coms_WT - n_coms_mut| / max_change
        if pcn_df is not None and "structure" in pcn_df.columns and wt_ncoms is not None:
            row = pcn_df[pcn_df["structure"] == name]
            if not row.empty:
                for col in ["n_coms_louvain", "n_coms_leiden"]:
                    if col in row.columns:
                        try:
                            ncoms = float(row.iloc[0][col])
                            scores["network"] = abs(wt_ncoms - ncoms) / max_ncoms_change
                            break
                        except (ValueError, TypeError):
                            scores["network"] = 0.0
            else:
                scores["network"] = 0.0
        else:
            scores["network"] = 0.0

        # Sequence: euclidean(emb_WT, emb_mut) / max_dist
        if esm_df is not None and esm_emb is not None and wt_emb is not None and "label" in esm_df.columns:
            row = esm_df[esm_df["label"] == name]
            if not row.empty:
                idx = row.index[0]
                if idx < len(esm_emb):
                    dist = np.linalg.norm(esm_emb[idx] - wt_emb)
                    scores["sequence"] = dist / max_emb_dist
                else:
                    scores["sequence"] = 0.0
            else:
                scores["sequence"] = 0.0
        else:
            scores["sequence"] = 0.0

        # Store raw per-component scores; composite is computed after we know
        # which components are informative across the whole proteoform set.
        scores["proteoform"] = name
        rows.append(scores)

    df = pd.DataFrame(rows)

    component_cols = ["structural", "binding", "dynamics", "network", "sequence"]
    present_cols = [c for c in component_cols if c in df.columns]

    # --- Drop degenerate components before building the composite ---
    # A component is "degenerate" locally when it carries no discriminative
    # signal: every proteoform gets the same value (e.g. TM-score is 1.0 for
    # all mutants because local mutant structures share the WT backbone, so
    # |1-TM| == 0 everywhere). Including such a component only dilutes the
    # composite with a constant, so we drop it and renormalize the weights
    # over the surviving informative components. This keeps the ranking driven
    # by the components that actually differ per proteoform. Locally, that is
    # the ESM2 sequence-embedding distance (the "sequence" component); the four
    # backbone-based components (structural/binding/dynamics/network) are all
    # degenerate because the local mutant structures share the WT backbone.
    # NOTE: ddG (the separate `ddg` step) is NOT one of the impact components.
    informative_cols = []
    degenerate_cols = []
    for c in present_cols:
        col = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        # informative if there is spread AND it is not all-zero
        if col.nunique(dropna=True) > 1 and float(col.abs().sum()) > 0.0:
            informative_cols.append(c)
        else:
            degenerate_cols.append(c)

    # Renormalize weights over surviving informative components.
    surv_weight_total = sum(weights.get(c, 0.0) for c in informative_cols)
    if informative_cols and surv_weight_total > 0:
        eff_weights = {c: weights.get(c, 0.0) / surv_weight_total for c in informative_cols}
    elif informative_cols:
        # informative components exist but all had zero configured weight ->
        # fall back to equal weighting so they still rank the proteoforms
        eff_weights = {c: 1.0 / len(informative_cols) for c in informative_cols}
    else:
        eff_weights = {}

    if eff_weights:
        df["composite"] = sum(
            pd.to_numeric(df[c], errors="coerce").fillna(0.0) * w
            for c, w in eff_weights.items()
        )
    else:
        # Nothing informative at all -> honest all-zero composite (GUI drops these rows)
        df["composite"] = 0.0

    # Sort by composite score descending
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)

    # Reorder columns
    cols = ["proteoform", "structural", "binding", "dynamics", "network", "sequence", "composite"]
    df = df[[c for c in cols if c in df.columns]]

    out_dir = paths.get("impact", os.path.join(rd, "impact_scores"))
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, "proteoform_impact_scores.csv")
    df.to_csv(csv, index=False)

    # Sidecar metadata: record which components contributed vs were dropped and
    # the effective (renormalized) weights, so the GUI/report can explain the
    # score composition honestly instead of implying all components were used.
    meta = pd.DataFrame(
        [
            {
                "component": c,
                "used": c in informative_cols,
                "effective_weight": eff_weights.get(c, 0.0),
                "configured_weight": weights.get(c, 0.0),
            }
            for c in present_cols
        ]
    )
    meta_csv = os.path.join(out_dir, "impact_score_components.csv")
    try:
        meta.to_csv(meta_csv, index=False)
    except Exception:
        meta_csv = None

    outputs = [csv] + ([meta_csv] if meta_csv else [])
    if degenerate_cols:
        detail = (
            f"Computed impact scores for {len(df)} proteoforms "
            f"(top: {df.iloc[0]['proteoform'] if len(df) else 'N/A'} = "
            f"{df.iloc[0]['composite']:.3f}); "
            f"used components: {', '.join(informative_cols) or 'none'}; "
            f"dropped degenerate (no local signal): {', '.join(degenerate_cols)}"
        )
    else:
        detail = (
            f"Computed impact scores for {len(df)} proteoforms "
            f"(top: {df.iloc[0]['proteoform'] if len(df) else 'N/A'} = "
            f"{df.iloc[0]['composite']:.3f})"
        )

    return StepResult(
        "impact_score", "ok", detail,
        outputs=outputs, data=df,
    )
