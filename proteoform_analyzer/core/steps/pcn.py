"""Step: Protein Contact Network analysis (PCN-Miner).

Builds residue contact networks (4-8 A), computes centrality measures
(betweenness, closeness, degree, eigenvector) and community detection
(louvain, leiden, infomap) for WT and mutant structures.
"""
from __future__ import annotations

import os
import sys
import logging
import numpy as np
import pandas as pd

from ..pipeline import StepResult

log = logging.getLogger("proteoform_analyzer.pcn")


def _vendored_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "_vendored")


def _import_pcn():
    v = _vendored_path()
    pcn_dir = os.path.join(v, "pcn_miner")
    if pcn_dir not in sys.path:
        sys.path.insert(0, pcn_dir)
    import pcn_miner_core
    return pcn_miner_core


def _compute_pcn_graph(pdb_path, pcn_miner, adj_dir, min_t=4.0, max_t=8.0):
    p_name = os.path.basename(pdb_path).replace(".pdb", "")
    out_path = os.path.join(adj_dir, f"{p_name}_adjacency_matrix.txt")
    atoms = pcn_miner.readPDBFile(pdb_path)
    residues = pcn_miner.getResidueCoordinates(atoms)
    dict_res = pcn_miner.associateResidueName(residues)
    residue_names = np.array(list(dict_res.items()))
    if os.path.exists(out_path):
        adj = pcn_miner.read_adj_matrix(out_path)
    else:
        adj = pcn_miner.adjacent_matrix_nonparallel(out_path, residues, p_name, min_t, max_t)
    from networkx import from_numpy_array
    G = from_numpy_array(adj)
    return G, residue_names, p_name


def _centrality(pdb_path, pcn_miner, paths, min_t=4.0, max_t=8.0):
    adj_dir = os.path.join(paths["pcn_outputs"], "Adj")
    os.makedirs(adj_dir, exist_ok=True)
    G, residue_names, p_name = _compute_pcn_graph(pdb_path, pcn_miner, adj_dir, min_t, max_t)
    res_names = np.array(residue_names[:, 1], dtype=str)
    measures = ["betweenness", "closeness", "degree_c", "eigenvector_c"]
    out = {}
    for m in measures:
        outdir = os.path.join(paths["pcn_outputs"], "Centralities", m)
        os.makedirs(outdir, exist_ok=True)
        try:
            fn = getattr(pcn_miner, m)
            vals = fn(G, res_names)
            pcn_miner.save_centralities(outdir, vals, p_name, m)
            out[m] = vals
        except Exception as e:
            log.warning("centrality %s on %s failed: %s", m, p_name, e)
    return out, p_name


def _communities(pdb_path, pcn_miner, paths, min_t=4.0, max_t=8.0):
    adj_dir = os.path.join(paths["pcn_outputs"], "Adj")
    G, residue_names, p_name = _compute_pcn_graph(pdb_path, pcn_miner, adj_dir, min_t, max_t)
    res_names = np.array(residue_names[:, 1], dtype=str)
    algos = ["louvain", "leiden", "infomap"]
    out = {}
    for algo in algos:
        outdir = os.path.join(paths["pcn_outputs"], "Communities", algo)
        os.makedirs(outdir, exist_ok=True)
        try:
            fn = getattr(pcn_miner, algo)
            labels = fn(G)
            # Robust community count. Some algorithms (louvain on small/degenerate
            # PCNs) can return a Python scalar or a 0-d numpy array, which breaks
            # ``max(labels)`` with "only 0-dimensional arrays can be converted to
            # Python scalars". Normalize to a 1-d int array first.
            labels_arr = np.atleast_1d(np.asarray(labels)).ravel()
            if labels_arr.size == 0:
                log.warning("community %s on %s: no labels returned; skipping.",
                            algo, p_name)
                continue
            n_coms = int(labels_arr.max()) + 1
            pcn_miner.save_labels(outdir, labels, residue_names, p_name, method=algo)
            out[algo] = {"n_communities": n_coms, "labels": labels}
            log.info("%s: %s -> %d communities", p_name, algo, n_coms)
        except ImportError as e:
            # e.g. infomap needs the optional 'wurlitzer' package.
            log.warning("community %s on %s unavailable (%s). Install the "
                        "optional dependency to enable it.", algo, p_name, e)
        except Exception as e:
            log.warning("community %s on %s failed: %s", algo, p_name, e)
    return out, p_name


def run_pcn(config, paths: dict) -> StepResult:
    """Run PCN centrality + community detection on WT, mutants, PTM structures,
    and proteoforms."""
    pcn_miner = _import_pcn()
    pdb_dir = paths["pdbs_monomer"] if config.is_monomer else paths["pdbs"]
    # All current-job structures: canonical WT+mutants (resolved through
    # boltz-experiments first), PTM-modified (ptms/ptms/) and proteoform
    # (proteoforms/) structures — see _structure_source.
    from ._structure_source import iter_all_structure_pdbs
    pdbs = [p for _stem, p in iter_all_structure_pdbs(
        config, paths, "pcn", include_ptms=True, include_proteoforms=True)]

    if not pdbs:
        return StepResult("pcn", "skipped",
                          f"No PDBs in {pdb_dir} (or ptms/ptms, proteoforms)")

    outputs = []
    summary = []
    for pdb in pdbs:
        name = os.path.basename(pdb).replace(".pdb", "")
        log.info("PCN analysis: %s", name)
        centrality, _ = _centrality(pdb, pcn_miner, paths)
        comm, _ = _communities(pdb, pcn_miner, paths)
        row = {"structure": name}
        for algo, info in comm.items():
            row[f"n_coms_{algo}"] = info["n_communities"]
        summary.append(row)

    df = pd.DataFrame(summary)
    csv = os.path.join(paths["pcn_outputs"], "pcn_summary.csv")
    df.to_csv(csv, index=False)
    outputs.append(csv)

    # plot centrality for WT if consequences available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cons = getattr(config, "_consequences", None)
        if cons and config.uniprot_ids:
            cons_df = cons[0]
            wt_name = next((n for n in df["structure"] if n.lower().startswith("wt")), None)
            if wt_name and "betweenness" in centrality:
                vals = centrality["betweenness"]
                # split by chain
                plt.figure(figsize=(10, 5))
                plt.plot(list(vals.values()) if isinstance(vals, dict) else vals, c="purple")
                plt.title(f"Betweenness centrality - {wt_name}")
                plt.xlabel("Residue"); plt.ylabel("Value")
                plt.tight_layout()
                plot_path = os.path.join(paths["pcn_outputs"], "betweenness_wt.svg")
                plt.savefig(plot_path, format="svg")
                plt.savefig(plot_path.replace(".svg", ".png"), dpi=150)
                plt.close()
                outputs.append(plot_path)
    except Exception as e:
        log.warning("centrality plot failed: %s", e)

    return StepResult(
        "pcn", "ok",
        f"PCN analysis on {len(pdbs)} structures "
        f"(centrality + communities)",
        outputs=outputs, data=df,
    )
