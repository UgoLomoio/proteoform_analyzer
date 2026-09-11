"""Shared visualization helpers (v3.1.0).

Every plotting function here produces BOTH an interactive Plotly ``.html`` and
static ``.svg`` + ``.png`` versions, and returns a dict of the paths that were
successfully written::

    {"html": "...", "svg": "...", "png": "...", "n": <points plotted>}

Design notes
------------
* Static export uses Plotly's ``kaleido`` backend when available; if kaleido is
  missing we still emit the interactive HTML and fall back to a matplotlib
  render for the SVG/PNG so the pipeline never hard-fails on a viz dependency.
* Colours use the Phylo palette / colour-blind-safe hues.
* All text is kept editable in the SVG (``svg.fonttype='none'`` for matplotlib;
  Plotly SVGs keep text as ``<text>`` by default).

These helpers are intentionally free of any Gradio dependency so they can be
unit-tested headless and reused by the CLI.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

log = logging.getLogger("proteoform.viz")

# Phylo palette + colour-blind-safe extensions
PALETTE = ["#0279EE", "#FF9400", "#75A025", "#FD9BED", "#000000",
           "#E9ED4C", "#D4A04A", "#4ECDC4", "#911eb4", "#e6194b"]
FONT_FAMILY = "Liberation Sans, Arimo, DejaVu Sans, sans-serif"

_KALEIDO_OK: Optional[bool] = None


def _kaleido_available() -> bool:
    """Cache whether static Plotly export works in this environment."""
    global _KALEIDO_OK
    if _KALEIDO_OK is None:
        try:
            import plotly.graph_objects as go  # noqa: F401
            import kaleido  # noqa: F401
            _KALEIDO_OK = True
        except Exception:
            _KALEIDO_OK = False
    return _KALEIDO_OK


def _write_plotly(fig, stem: str, static_via_matplotlib=None) -> dict:
    """Write ``fig`` to ``stem.html`` (+ ``.svg``/``.png`` if possible).

    ``static_via_matplotlib`` is an optional zero-arg callable that renders the
    same data with matplotlib and saves ``stem.svg``/``stem.png`` -- used as a
    fallback when kaleido is unavailable.
    """
    out = {}
    try:
        import plotly.io as pio
        html_path = stem + ".html"
        pio.write_html(fig, html_path, include_plotlyjs="cdn", full_html=True)
        out["html"] = html_path
    except Exception as e:  # pragma: no cover - plotly should be present
        log.warning("Plotly HTML export failed for %s: %s", stem, e)

    if _kaleido_available():
        try:
            import plotly.io as pio
            pio.write_image(fig, stem + ".svg", format="svg")
            pio.write_image(fig, stem + ".png", format="png", scale=2)
            out["svg"] = stem + ".svg"
            out["png"] = stem + ".png"
            return out
        except Exception as e:
            log.warning("Plotly static export failed for %s: %s", stem, e)

    # Fallback: matplotlib static render
    if static_via_matplotlib is not None:
        try:
            svg, png = static_via_matplotlib()
            if svg:
                out["svg"] = svg
            if png:
                out["png"] = png
        except Exception as e:
            log.warning("matplotlib static fallback failed for %s: %s", stem, e)
    return out


def _mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"
    # Prefer Liberation Sans / Arimo (Arial-metric) but only request families
    # that are actually installed — naming a missing family makes matplotlib
    # emit "findfont: Font family 'Arimo' not found" warnings on every plot.
    # DejaVu Sans is bundled with matplotlib, so it is always a safe fallback.
    preferred = ["Liberation Sans", "Arimo", "DejaVu Sans"]
    try:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        families = [f for f in preferred if f in installed]
    except Exception:
        families = []
    if not families:
        families = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = families
    return plt


# ---------------------------------------------------------------------------
# 1. Molecular dynamics: RMSD & RMSF overlays
# ---------------------------------------------------------------------------

def md_overlay(series, stem: str, kind: str = "rmsd") -> dict:
    """Overlay per-structure RMSD or RMSF curves.

    Parameters
    ----------
    series : list of dict
        Each item: ``{"label": str, "x": array, "y": array}``.
    stem : str
        Output path stem (no extension).
    kind : {"rmsd", "rmsf"}
        Controls axis labels/title.
    """
    series = [s for s in series if s and len(s.get("x", [])) and len(s.get("y", []))]
    if not series:
        return {"n": 0}

    if kind == "rmsf":
        xlab, ylab, title = "Residue", "Cα RMSF (Å)", "MD RMSF overlay"
    else:
        xlab, ylab, title = "Time (ps)", "Cα RMSD (Å)", "MD RMSD overlay"

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for i, s in enumerate(series):
            fig.add_trace(go.Scatter(
                x=list(s["x"]), y=list(s["y"]), mode="lines", name=str(s["label"]),
                line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
        fig.update_layout(
            template="simple_white", title=title,
            xaxis_title=xlab, yaxis_title=ylab,
            font=dict(family=FONT_FAMILY, size=13),
            legend=dict(title="Structure"), width=900, height=520)
    except Exception as e:
        log.warning("md_overlay plotly build failed: %s", e)
        fig = None

    def _mpl():
        plt = _mpl_setup()
        fig2, ax = plt.subplots(figsize=(9, 5.2))
        for i, s in enumerate(series):
            ax.plot(s["x"], s["y"], label=str(s["label"]),
                    color=PALETTE[i % len(PALETTE)], linewidth=1.8)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
        ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)
        fig2.tight_layout()
        svg, png = stem + ".svg", stem + ".png"
        fig2.savefig(svg, format="svg"); fig2.savefig(png, dpi=150)
        plt.close(fig2)
        return svg, png

    out = _write_plotly(fig, stem, _mpl) if fig is not None else {}
    if not out:  # plotly missing entirely
        svg, png = _mpl()
        out = {"svg": svg, "png": png}
    out["n"] = len(series)
    return out


# ---------------------------------------------------------------------------
# 3. Docking plots
# ---------------------------------------------------------------------------

def docking_affinity_bars(df, stem: str) -> dict:
    """Horizontal bar chart of Vina binding affinity per structure/ligand.

    Expects columns ``structure``, ``ligand``, ``affinity_kcal_mol`` and
    optionally ``affinity_std``. Lower (more negative) = stronger binding.
    """
    import pandas as pd
    if df is None or len(df) == 0 or "affinity_kcal_mol" not in df.columns:
        return {"n": 0}
    d = df.copy()
    d = d[pd.to_numeric(d["affinity_kcal_mol"], errors="coerce").notna()]
    if len(d) == 0:
        return {"n": 0}
    d["affinity_kcal_mol"] = d["affinity_kcal_mol"].astype(float)
    lig = d["ligand"] if "ligand" in d.columns else ""
    d["label"] = d["structure"].astype(str) + " · " + pd.Series(lig, index=d.index).astype(str).str.replace(".sdf", "", regex=False)
    # strongest (most negative) at top
    d = d.sort_values("affinity_kcal_mol", ascending=False)
    err = d["affinity_std"].astype(float).tolist() if "affinity_std" in d.columns else None

    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=d["affinity_kcal_mol"].tolist(), y=d["label"].tolist(),
            orientation="h",
            error_x=dict(type="data", array=err, visible=True) if err else None,
            marker=dict(color=d["affinity_kcal_mol"].tolist(),
                        colorscale="Blues_r", showscale=True,
                        colorbar=dict(title="kcal/mol"))))
        fig.update_layout(
            template="simple_white", title="Docking binding affinity",
            xaxis_title="Vina affinity (kcal/mol, lower = stronger)",
            yaxis_title="", font=dict(family=FONT_FAMILY, size=13),
            width=900, height=max(320, 40 * len(d) + 120))
    except Exception as e:
        log.warning("docking_affinity_bars plotly failed: %s", e)
        fig = None

    def _mpl():
        plt = _mpl_setup()
        fig2, ax = plt.subplots(figsize=(9, max(3.2, 0.4 * len(d) + 1.2)))
        ax.barh(d["label"].tolist(), d["affinity_kcal_mol"].tolist(),
                xerr=err, color=PALETTE[0])
        ax.set_xlabel("Vina affinity (kcal/mol, lower = stronger)")
        ax.set_title("Docking binding affinity")
        fig2.tight_layout()
        svg, png = stem + ".svg", stem + ".png"
        fig2.savefig(svg, format="svg"); fig2.savefig(png, dpi=150)
        plt.close(fig2)
        return svg, png

    out = _write_plotly(fig, stem, _mpl) if fig is not None else {}
    if not out:
        svg, png = _mpl(); out = {"svg": svg, "png": png}
    out["n"] = len(d)
    return out


def docking_boltz_scatter(df, stem: str) -> dict:
    """Scatter of Boltz-2 pTM (x) vs ipTM (y), coloured by confidence_score.

    Note: the original spec called for colouring by RFAntibody pAE, but pAE is
    produced by the separate antibody step and does not join to the Boltz-2
    docking table (different structures). We colour by ``confidence_score``, the
    Boltz-2-native global confidence, which IS present in this table.
    """
    import pandas as pd
    if df is None or len(df) == 0:
        return {"n": 0}
    if "ptm" not in df.columns or "iptm" not in df.columns:
        return {"n": 0}
    d = df.copy()
    d = d[pd.to_numeric(d["ptm"], errors="coerce").notna()
          & pd.to_numeric(d["iptm"], errors="coerce").notna()]
    if len(d) == 0:
        return {"n": 0}
    d["ptm"] = d["ptm"].astype(float)
    d["iptm"] = d["iptm"].astype(float)
    color = None
    if "confidence_score" in d.columns and pd.to_numeric(d["confidence_score"], errors="coerce").notna().any():
        color = pd.to_numeric(d["confidence_score"], errors="coerce").tolist()
    lig = d["ligand"] if "ligand" in d.columns else ""
    labels = (d["structure"].astype(str) + " · "
              + pd.Series(lig, index=d.index).astype(str)).tolist()

    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(
            x=d["ptm"].tolist(), y=d["iptm"].tolist(), mode="markers+text",
            text=labels, textposition="top center", textfont=dict(size=9),
            marker=dict(size=13,
                        color=color if color is not None else PALETTE[0],
                        colorscale="Viridis" if color is not None else None,
                        showscale=color is not None,
                        colorbar=dict(title="confidence") if color is not None else None,
                        line=dict(width=1, color="#333")),
            hovertext=labels))
        fig.update_layout(
            template="simple_white", title="Boltz-2 docking confidence: pTM vs ipTM",
            xaxis_title="pTM", yaxis_title="ipTM",
            font=dict(family=FONT_FAMILY, size=13), width=760, height=640)
        fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1])
    except Exception as e:
        log.warning("docking_boltz_scatter plotly failed: %s", e)
        fig = None

    def _mpl():
        plt = _mpl_setup()
        fig2, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(d["ptm"], d["iptm"],
                        c=color if color is not None else PALETTE[0],
                        cmap="viridis" if color is not None else None,
                        s=90, edgecolor="#333")
        if color is not None:
            fig2.colorbar(sc, ax=ax, label="confidence")
        for x, y, t in zip(d["ptm"], d["iptm"], labels):
            ax.annotate(t, (x, y), fontsize=8, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("pTM"); ax.set_ylabel("ipTM")
        ax.set_title("Boltz-2 docking confidence: pTM vs ipTM")
        fig2.tight_layout()
        svg, png = stem + ".svg", stem + ".png"
        fig2.savefig(svg, format="svg"); fig2.savefig(png, dpi=150)
        plt.close(fig2)
        return svg, png

    out = _write_plotly(fig, stem, _mpl) if fig is not None else {}
    if not out:
        svg, png = _mpl(); out = {"svg": svg, "png": png}
    out["n"] = len(d)
    return out


# ---------------------------------------------------------------------------
# 4. Pocket volume + drift
# ---------------------------------------------------------------------------

def pocket_volume_bars(df, stem: str) -> dict:
    """Grouped bars of predicted pocket volume per structure.

    Expects ``structure`` and ``volume`` (Å³). If a ``detector_failed`` column
    is present, failed rows are annotated so zeros are never shown as real data.
    """
    import pandas as pd
    if df is None or len(df) == 0:
        return {"n": 0}
    vol_col = "volume" if "volume" in df.columns else (
        "volume_A3" if "volume_A3" in df.columns else None)
    if vol_col is None or "structure" not in df.columns:
        return {"n": 0}
    d = df.copy()
    d[vol_col] = pd.to_numeric(d[vol_col], errors="coerce").fillna(0.0)
    failed = (d["detector_failed"].astype(str).str.lower().isin(["true", "1"])
              if "detector_failed" in d.columns else pd.Series(False, index=d.index))
    colors = ["#bbbbbb" if f else PALETTE[0] for f in failed]
    labels = d["structure"].astype(str).tolist()
    annot = ["detector failed" if f else "" for f in failed]

    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=labels, y=d[vol_col].tolist(), marker_color=colors, text=annot,
            textposition="outside"))
        fig.update_layout(
            template="simple_white", title="Predicted binding-pocket volume",
            xaxis_title="Structure", yaxis_title="Pocket volume (Å³)",
            font=dict(family=FONT_FAMILY, size=13),
            width=max(600, 60 * len(d) + 200), height=500)
    except Exception as e:
        log.warning("pocket_volume_bars plotly failed: %s", e)
        fig = None

    def _mpl():
        plt = _mpl_setup()
        fig2, ax = plt.subplots(figsize=(max(6, 0.6 * len(d) + 2), 5))
        ax.bar(labels, d[vol_col].tolist(), color=colors)
        ax.set_xlabel("Structure"); ax.set_ylabel("Pocket volume (Å³)")
        ax.set_title("Predicted binding-pocket volume")
        plt.xticks(rotation=45, ha="right")
        fig2.tight_layout()
        svg, png = stem + ".svg", stem + ".png"
        fig2.savefig(svg, format="svg"); fig2.savefig(png, dpi=150)
        plt.close(fig2)
        return svg, png

    out = _write_plotly(fig, stem, _mpl) if fig is not None else {}
    if not out:
        svg, png = _mpl(); out = {"svg": svg, "png": png}
    out["n"] = len(d)
    return out


def pocket_drift_scatter(df, stem: str) -> dict:
    """Scatter of pocket volume change (x) vs centre displacement (y).

    Expects columns ``proteoform``, ``volume_change``, ``center_displacement_A``.
    """
    import pandas as pd
    if df is None or len(df) == 0:
        return {"n": 0}
    need = {"volume_change", "center_displacement_A"}
    if not need.issubset(set(df.columns)):
        return {"n": 0}
    d = df.copy()
    d["volume_change"] = pd.to_numeric(d["volume_change"], errors="coerce")
    d["center_displacement_A"] = pd.to_numeric(d["center_displacement_A"], errors="coerce")
    d = d[d["volume_change"].notna() & d["center_displacement_A"].notna()]
    if len(d) == 0:
        return {"n": 0}
    labels = (d["proteoform"].astype(str).tolist()
              if "proteoform" in d.columns else [str(i) for i in range(len(d))])

    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(
            x=d["volume_change"].tolist(), y=d["center_displacement_A"].tolist(),
            mode="markers+text", text=labels, textposition="top center",
            textfont=dict(size=9),
            hovertemplate=("%{text}<br>Volume change: %{x:.4f} Å³"
                           "<br>Centre shift: %{y:.4f} Å<extra></extra>"),
            marker=dict(size=13, color=PALETTE[1], line=dict(width=1, color="#333"))))
        fig.add_vline(x=0, line_dash="dash", line_color="#888")
        fig.update_layout(
            template="simple_white",
            title="Pocket drift vs wild-type (volume change vs centre shift)",
            xaxis_title="Volume change (Å³)  [− shrink / + grow]",
            yaxis_title="Pocket-centre displacement (Å)",
            font=dict(family=FONT_FAMILY, size=13), width=780, height=560)
    except Exception as e:
        log.warning("pocket_drift_scatter plotly failed: %s", e)
        fig = None

    def _mpl():
        plt = _mpl_setup()
        fig2, ax = plt.subplots(figsize=(7.5, 5.4))
        ax.scatter(d["volume_change"], d["center_displacement_A"],
                   s=90, color=PALETTE[1], edgecolor="#333")
        for x, y, t in zip(d["volume_change"], d["center_displacement_A"], labels):
            ax.annotate(t, (x, y), fontsize=8, xytext=(3, 3),
                        textcoords="offset points")
        ax.axvline(0, linestyle="--", color="#888")
        ax.set_xlabel("Volume change (Å³)  [− shrink / + grow]")
        ax.set_ylabel("Pocket-centre displacement (Å)")
        ax.set_title("Pocket drift vs wild-type")
        fig2.tight_layout()
        svg, png = stem + ".svg", stem + ".png"
        fig2.savefig(svg, format="svg"); fig2.savefig(png, dpi=150)
        plt.close(fig2)
        return svg, png

    out = _write_plotly(fig, stem, _mpl) if fig is not None else {}
    if not out:
        svg, png = _mpl(); out = {"svg": svg, "png": png}
    out["n"] = len(d)
    return out


# ---------------------------------------------------------------------------
# 5. Generic single-column summary bar (used per Results subsection)
# ---------------------------------------------------------------------------

def summary_bar(df, label_col: str, value_col: str, stem: str,
                title: str = "", value_title: str = "",
                horizontal: bool = True, sort: bool = True,
                text_format: str | None = None) -> dict:
    """Generic bar chart from a dataframe (one categorical + one numeric col).

    ``text_format`` (e.g. ``".4f"``) prints the exact value on each bar and in
    the hover box. Use it for metrics whose meaningful differences are small
    (TM-scores near 1.0), so bars that look identical are still readable.
    """
    import pandas as pd
    if df is None or len(df) == 0:
        return {"n": 0}
    if label_col not in df.columns or value_col not in df.columns:
        return {"n": 0}
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d[d[value_col].notna()]
    if len(d) == 0:
        return {"n": 0}
    if sort:
        d = d.sort_values(value_col, ascending=horizontal)
    labels = d[label_col].astype(str).tolist()
    vals = d[value_col].astype(float).tolist()
    value_title = value_title or value_col
    title = title or value_col
    bar_text = ([format(v, text_format) for v in vals] if text_format else None)

    try:
        import plotly.graph_objects as go
        _tt = (f"%{{y}}<br>{value_title}: %{{x:{text_format}}}<extra></extra>"
               if (text_format and horizontal) else
               (f"%{{x}}<br>{value_title}: %{{y:{text_format}}}<extra></extra>"
                if text_format else None))
        if horizontal:
            fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                                   marker_color=PALETTE[2],
                                   text=bar_text, textposition="outside",
                                   hovertemplate=_tt))
            fig.update_layout(xaxis_title=value_title, yaxis_title="")
        else:
            fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=PALETTE[2],
                                   text=bar_text, textposition="outside",
                                   hovertemplate=_tt))
            fig.update_layout(yaxis_title=value_title, xaxis_title="")
        fig.update_layout(template="simple_white", title=title,
                          font=dict(family=FONT_FAMILY, size=13),
                          width=860, height=max(320, 34 * len(d) + 140))
        if text_format:
            fig.update_traces(cliponaxis=False)
    except Exception as e:
        log.warning("summary_bar plotly failed: %s", e)
        fig = None

    def _mpl():
        plt = _mpl_setup()
        if horizontal:
            fig2, ax = plt.subplots(figsize=(8.6, max(3, 0.36 * len(d) + 1.4)))
            bars = ax.barh(labels, vals, color=PALETTE[2]); ax.set_xlabel(value_title)
            if text_format:
                ax.bar_label(bars, labels=[format(v, text_format) for v in vals],
                             fontsize=7, padding=2)
        else:
            fig2, ax = plt.subplots(figsize=(max(6, 0.6 * len(d) + 2), 5))
            bars = ax.bar(labels, vals, color=PALETTE[2]); ax.set_ylabel(value_title)
            plt.xticks(rotation=45, ha="right")
            if text_format:
                ax.bar_label(bars, labels=[format(v, text_format) for v in vals],
                             fontsize=7, padding=2)
        ax.set_title(title)
        fig2.tight_layout()
        svg, png = stem + ".svg", stem + ".png"
        fig2.savefig(svg, format="svg"); fig2.savefig(png, dpi=150)
        plt.close(fig2)
        return svg, png

    out = _write_plotly(fig, stem, _mpl) if fig is not None else {}
    if not out:
        svg, png = _mpl(); out = {"svg": svg, "png": png}
    out["n"] = len(d)
    return out


def centrality_delta_bars(wt_dict, mut_dict, measure: str, stem: str,
                          structure: str = "", top_n: int = 10) -> dict:
    """Top-N residues by |Δ centrality| (mutant - WT) as signed horizontal bars.

    Parameters
    ----------
    wt_dict, mut_dict : {residue_label: centrality} dicts (e.g. 'VAL1 A' -> 0.12),
        as parsed from pcn_miner centrality output.
    measure : centrality name (for the axis/title label).
    stem : output path stem; writes ``stem.svg`` and ``stem.png``.
    structure : structure name (title).
    top_n : number of residues to show (default 10).

    Bars are coloured red for a positive Δ (centrality increased vs WT) and blue
    for a negative Δ (decreased); each is labelled ``NAME###(chain)``. A small
    +/- legend is drawn. Returns ``{svg, png, n}`` (``{"n": 0}`` if no overlap).
    """
    import re as _re
    if not wt_dict or not mut_dict:
        return {"n": 0}

    def _parse(label):
        m = _re.match(r"([A-Z]+)(\d+)\s+([A-Za-z0-9])", str(label).strip())
        if m:
            return m.group(1), m.group(2), m.group(3)
        return None

    deltas = []
    for label, mval in mut_dict.items():
        if label in wt_dict:
            try:
                d = float(mval) - float(wt_dict[label])
            except (TypeError, ValueError):
                continue
            parsed = _parse(label)
            if parsed is None:
                continue
            name, resi, chain = parsed
            deltas.append((f"{name}{resi}({chain})", d))

    if not deltas:
        return {"n": 0}

    # Drop residues whose centrality did not actually change: padding a
    # "top-N by |Δ|" plot with zero-Δ residues (e.g. in backbone-identical /
    # graft mode where most centralities are unchanged) is misleading. Use a
    # small tolerance to also discard floating-point noise. If nothing changed,
    # return n=0 so the GUI can say "no centrality change vs WT".
    _EPS = 1e-9
    nonzero = [d for d in deltas if abs(d[1]) > _EPS]
    if not nonzero:
        return {"n": 0}

    # Rank by absolute delta, keep top_n, then order for a clean horizontal bar
    # (largest |Δ| at the top).
    nonzero.sort(key=lambda x: abs(x[1]), reverse=True)
    top = nonzero[:top_n]
    top.sort(key=lambda x: x[1])  # ascending so barh puts biggest +Δ at top
    labels = [t[0] for t in top]
    vals = [t[1] for t in top]

    RED = "#b2182b"    # increase
    BLUE = "#2166ac"   # decrease
    colors = [RED if v >= 0 else BLUE for v in vals]

    measure_label = {
        "betweenness": "Betweenness", "closeness": "Closeness",
        "degree_c": "Degree", "eigenvector_c": "Eigenvector",
    }.get(measure, measure)
    title = (f"Top {len(top)} residues by |\u0394 {measure_label} centrality|"
             + (f"  \u2014  {structure} vs WT" if structure else ""))

    plt = _mpl_setup()
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(8.4, max(3.0, 0.42 * len(top) + 1.6)))
    ax.barh(labels, vals, color=colors, edgecolor="#333", linewidth=0.4)
    ax.axvline(0, color="#666", linewidth=0.8)
    ax.set_xlabel(f"\u0394 {measure_label} centrality (mutant \u2212 WT)")
    ax.set_ylabel("Residue (name+id, chain)")
    ax.set_title(title, fontsize=11)
    legend_handles = [
        mpatches.Patch(color=RED, label="Positive \u0394 (increase vs WT)"),
        mpatches.Patch(color=BLUE, label="Negative \u0394 (decrease vs WT)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    svg, png = stem + ".svg", stem + ".png"
    fig.savefig(svg, format="svg")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return {"svg": svg, "png": png, "n": len(top)}


# ---------------------------------------------------------------------------
# 6. Antibody design (RFAntibody / RF2 scores)
# ---------------------------------------------------------------------------

def _antibody_score_cols(df):
    """Map the RF2 score-table columns to canonical roles.

    The RFAntibody ``scores.tsv`` (qvscorefile output) carries whichever metrics
    the RF2 run recorded; typical columns are ``interaction_pae``, ``pae``,
    ``pred_lddt``, ``target_aligned_cdr_rmsd``, ``target_aligned_antibody_rmsd``,
    ``framework_aligned_*_rmsd`` and ``tag``. Returns (tag_col, pae_col,
    rmsd_col, plddt_col); any except tag_col may be None when absent.
    """
    cols = {c.lower(): c for c in df.columns}
    tag_col = cols.get("tag") or cols.get("design") or df.columns[0]
    pae_col = cols.get("interaction_pae") or cols.get("pae_interaction") \
        or cols.get("pae")
    rmsd_col = cols.get("target_aligned_cdr_rmsd") \
        or cols.get("target_aligned_antibody_rmsd") \
        or cols.get("framework_aligned_cdr_rmsd") or cols.get("rmsd")
    plddt_col = cols.get("pred_lddt") or cols.get("mean_plddt") \
        or cols.get("plddt")
    return tag_col, pae_col, rmsd_col, plddt_col


def antibody_pae_rmsd_scatter(df, stem: str, pae_max: float = 10.0,
                              rmsd_max: float = 2.0) -> dict:
    """Scatter of RF2 interaction pAE (y) vs target-aligned CDR RMSD (x).

    One point per designed complex; vertical/horizontal threshold lines mark the
    configured pass cutoffs (``rmsd_max`` / ``pae_max``) so passing designs pop
    out in the lower-left quadrant. Points are coloured by pass/fail.
    """
    import pandas as pd
    if df is None or len(df) == 0:
        return {"n": 0}
    tag_col, pae_col, rmsd_col, plddt_col = _antibody_score_cols(df)
    if pae_col is None or rmsd_col is None:
        return {"n": 0}
    d = df.copy()
    d["_pae"] = pd.to_numeric(d[pae_col], errors="coerce")
    d["_rmsd"] = pd.to_numeric(d[rmsd_col], errors="coerce")
    d = d.dropna(subset=["_pae", "_rmsd"])
    if len(d) == 0:
        return {"n": 0}
    d["_pass"] = (d["_pae"] < float(pae_max)) & (d["_rmsd"] < float(rmsd_max))
    d["_tag"] = d[tag_col].astype(str)
    hover = d["_tag"].tolist()
    if plddt_col is not None:
        d["_plddt"] = pd.to_numeric(d[plddt_col], errors="coerce")
        hover = [f"{t}<br>pLDDT={p:.1f}" if pd.notna(p) else t
                 for t, p in zip(d["_tag"], d["_plddt"])]

    PASS, FAIL = "#0279EE", "#FF9400"
    title = (f"Antibody designs: RF2 pAE vs RMSD "
             f"(pass: pAE&lt;{pae_max:g}, RMSD&lt;{rmsd_max:g} Å)")

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for flag, label, color in ((True, "pass", PASS), (False, "fail", FAIL)):
            sub = d[d["_pass"] == flag]
            if len(sub) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=sub["_rmsd"].tolist(), y=sub["_pae"].tolist(),
                mode="markers", name=label,
                text=[h for h, f in zip(hover, d["_pass"]) if f == flag],
                hovertemplate="%{text}<br>RMSD=%{x:.2f} Å<br>pAE=%{y:.2f}<extra>%{fullData.name}</extra>",
                marker=dict(size=10, color=color, opacity=0.85,
                            line=dict(color="#333", width=0.6))))
        fig.add_vline(x=float(rmsd_max), line_dash="dash", line_color="#888",
                      annotation_text=f"RMSD {rmsd_max:g} Å")
        fig.add_hline(y=float(pae_max), line_dash="dash", line_color="#888",
                      annotation_text=f"pAE {pae_max:g}")
        fig.update_layout(
            template="simple_white", title=title,
            xaxis_title="Target-aligned CDR RMSD (Å, lower = better)",
            yaxis_title="RF2 interaction pAE (lower = better)",
            font=dict(family=FONT_FAMILY, size=13),
            legend=dict(title="Thresholds"), width=900, height=560)
    except Exception as e:
        log.warning("antibody_pae_rmsd_scatter plotly failed: %s", e)
        fig = None

    def _mpl():
        plt = _mpl_setup()
        fig2, ax = plt.subplots(figsize=(8.6, 5.4))
        for flag, label, color in ((True, "pass", PASS), (False, "fail", FAIL)):
            sub = d[d["_pass"] == flag]
            if len(sub):
                ax.scatter(sub["_rmsd"], sub["_pae"], s=42, c=color,
                           label=label, edgecolors="#333", linewidths=0.5,
                           zorder=3)
        ax.axvline(float(rmsd_max), color="#888", linestyle="--", linewidth=1)
        ax.axhline(float(pae_max), color="#888", linestyle="--", linewidth=1)
        ax.set_xlabel("Target-aligned CDR RMSD (Å, lower = better)")
        ax.set_ylabel("RF2 interaction pAE (lower = better)")
        ax.set_title(title.replace("&lt;", "<"), fontsize=11)
        ax.legend(title="Thresholds"); ax.grid(True, alpha=0.3)
        fig2.tight_layout()
        svg, png = stem + ".svg", stem + ".png"
        fig2.savefig(svg, format="svg"); fig2.savefig(png, dpi=150)
        plt.close(fig2)
        return svg, png

    out = _write_plotly(fig, stem, _mpl) if fig is not None else {}
    if not out:
        svg, png = _mpl(); out = {"svg": svg, "png": png}
    out["n"] = len(d)
    return out


def antibody_score_bars(df, stem: str, pae_max: float = 10.0) -> dict:
    """Horizontal bar chart ranking designed antibodies by RF2 interaction pAE.

    Best (lowest pAE) at top; bars passing the ``pae_max`` threshold are drawn
    in the accent colour, failing bars in grey.
    """
    import pandas as pd
    if df is None or len(df) == 0:
        return {"n": 0}
    tag_col, pae_col, _rmsd_col, _plddt_col = _antibody_score_cols(df)
    if pae_col is None:
        return {"n": 0}
    d = df.copy()
    d["_pae"] = pd.to_numeric(d[pae_col], errors="coerce")
    d = d.dropna(subset=["_pae"])
    if len(d) == 0:
        return {"n": 0}
    d["_tag"] = d[tag_col].astype(str)
    d = d.sort_values("_pae", ascending=False)  # best (lowest) at top of barh
    d["_pass"] = d["_pae"] < float(pae_max)
    PASS, FAIL = "#0279EE", "#b8b8b8"
    colors = [PASS if p else FAIL for p in d["_pass"]]
    title = f"Antibody design ranking by RF2 interaction pAE (pass: pAE&lt;{pae_max:g})"

    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=d["_pae"].tolist(), y=d["_tag"].tolist(), orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}<br>pAE=%{x:.2f}<extra></extra>"))
        fig.add_vline(x=float(pae_max), line_dash="dash", line_color="#888",
                      annotation_text=f"pAE {pae_max:g}")
        fig.update_layout(
            template="simple_white", title=title,
            xaxis_title="RF2 interaction pAE (lower = better)",
            yaxis_title="", font=dict(family=FONT_FAMILY, size=13),
            width=900, height=max(320, 34 * len(d) + 140))
    except Exception as e:
        log.warning("antibody_score_bars plotly failed: %s", e)
        fig = None

    def _mpl():
        plt = _mpl_setup()
        fig2, ax = plt.subplots(figsize=(9, max(3.2, 0.34 * len(d) + 1.4)))
        ax.barh(d["_tag"].tolist(), d["_pae"].tolist(), color=colors,
                edgecolor="#333", linewidth=0.4)
        ax.axvline(float(pae_max), color="#888", linestyle="--", linewidth=1)
        ax.set_xlabel("RF2 interaction pAE (lower = better)")
        ax.set_title(title.replace("&lt;", "<"), fontsize=11)
        ax.tick_params(axis="y", labelsize=8)
        fig2.tight_layout()
        svg, png = stem + ".svg", stem + ".png"
        fig2.savefig(svg, format="svg"); fig2.savefig(png, dpi=150)
        plt.close(fig2)
        return svg, png

    out = _write_plotly(fig, stem, _mpl) if fig is not None else {}
    if not out:
        svg, png = _mpl(); out = {"svg": svg, "png": png}
    out["n"] = len(d)
    return out
