"""Step: ESM2 embeddings + UMAP variant classification.

Computes ESM2 embeddings for WT and each mutant sequence, projects to 2D via UMAP,
and clusters variants.  Requires the `esm` (fair-esm) package and a model download
(~1.3 GB for ESM2-650M).  Gracefully skips if unavailable.
"""
from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd

from ..pipeline import StepResult

log = logging.getLogger("proteoform_analyzer.esm2")


def _apply_mutation(seq: str, mut: str) -> str:
    if mut.upper() == "WT":
        return seq
    pos = int(mut[1:-1])
    new = mut[-1]
    return seq[:pos - 1] + new + seq[pos:]


def _embed_sequences(sequences: list[str], model_name: str = "esm2_t33_650M_UR50D") -> np.ndarray:
    """Embed a list of sequences with ESM2, returning mean-pooled embeddings."""
    import torch
    import esm
    from ._device import torch_device
    # CPU by default; uses CUDA on a ZeroGPU Space (inside the @spaces.GPU wrap
    # installed by the Spaces app.py) when a GPU is actually allocated.
    device = torch_device()
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            data = [("p", seq)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_repr = results["representations"][33][0, 1:-1]  # drop BOS/EOS
            emb = token_repr.mean(0).cpu().numpy()
            embeddings.append(emb)
    return np.array(embeddings)


def run_esm2(config, paths: dict) -> StepResult:
    """Run ESM2 + UMAP on WT and mutant sequences."""
    sequences = getattr(config, "_sequences", None)
    mutation_lists = getattr(config, "_mutation_lists", None)
    if not sequences or not mutation_lists:
        # try loading from cached files
        from .sequence import read_fasta
        sequences = []
        mutation_lists = []
        for uid in config.uniprot_ids:
            fp = os.path.join(paths["input"], f"{uid}.fasta")
            if not os.path.exists(fp):
                return StepResult("esm2", "skipped", "No sequences available (run 'sequence' step first)")
            sequences.append(read_fasta(fp))
            mf = os.path.join(paths["results"], f"{uid}_mutations.txt")
            if os.path.exists(mf):
                with open(mf) as f:
                    mutation_lists.append([l.strip() for l in f if l.strip()])
            else:
                mutation_lists.append(["WT"])
        config._sequences = sequences
        config._mutation_lists = mutation_lists
    if not sequences:
        return StepResult("esm2", "skipped", "No sequences available")

    emb_dir = paths["embeddings"]
    os.makedirs(emb_dir, exist_ok=True)

    # build the full sequence list (WT + mutants per subunit)
    # Use mature (post-cleavage) sequences when a mature region is resolved;
    # mutations are remapped to mature numbering and cleaved-region mutations
    # are skipped (they are absent from the mature protein).
    from .sequence import (get_mature_sequences, resolve_mature_regions,
                           to_mature_pos, parse_mutation_pos)
    mature_regions = resolve_mature_regions(config, paths)
    sequences = get_mature_sequences(config, paths) or sequences
    all_seqs = []
    labels = []
    subunit = []
    n_cleaved = 0
    for idx, (uid, seq) in enumerate(zip(config.uniprot_ids, sequences)):
        region = mature_regions.get(uid)
        muts = mutation_lists[idx] if idx < len(mutation_lists) else []
        for mut in muts:
            applied_mut = mut
            pos = parse_mutation_pos(mut)
            if pos is not None and region is not None:
                mpos = to_mature_pos(region, pos)
                if mpos is None:
                    n_cleaved += 1
                    log.warning(
                        "esm2: mutation %s (%s) lies in the proteolytically "
                        "cleaved region (mature chain %d-%d); skipped",
                        mut, uid, region[0], region[1])
                    continue
                applied_mut = f"{mut[0]}{mpos}{mut[-1]}"
            all_seqs.append(_apply_mutation(seq, applied_mut))
            labels.append(mut)  # keep UniProt-numbered label for downstream joins
            subunit.append(uid)
    if n_cleaved:
        log.info("esm2: %d mutation(s) in cleaved regions excluded from embedding", n_cleaved)

    # try to load ESM2
    try:
        import esm  # noqa: F401
    except Exception as e:
        return StepResult("esm2", "skipped", f"fair-esm not available: {e}")

    model_name = "esm2_t33_650M_UR50D"
    try:
        log.info("Loading ESM2 model %s (may download ~1.3 GB on first use)...", model_name)
        emb = _embed_sequences(all_seqs, model_name)
    except Exception as e:
        return StepResult("esm2", "skipped", f"ESM2 embedding failed: {type(e).__name__}: {e}")

    np.save(os.path.join(emb_dir, "embeddings.npy"), emb)
    df_emb = pd.DataFrame({"label": labels, "subunit": subunit})
    df_emb.to_csv(os.path.join(emb_dir, "embedding_labels.csv"), index=False)

    # UMAP
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["svg.fonttype"] = "none"
        n = len(all_seqs)
        # UMAP fails for very small N (spectral embedding). Use PCA fallback then.
        if n < 4:
            from sklearn.decomposition import PCA
            log.info("Too few sequences (%d) for UMAP; using PCA instead", n)
            proj = PCA(n_components=2).fit_transform(emb)
        else:
            reducer = umap.UMAP(n_neighbors=min(4, n - 1),
                                min_dist=0.4, n_components=2, metric="euclidean",
                                random_state=42)
            proj = reducer.fit_transform(emb)
        # Build pathogenicity classification for each variant
        pathogenicities = []
        cons_list = getattr(config, "_consequences", None) or []
        for idx, uid in enumerate(config.uniprot_ids):
            cons_dict = {}
            if idx < len(cons_list):
                cons_df = cons_list[idx]
                if cons_df is not None and not cons_df.empty:
                    for mut_str, row in cons_df.iterrows():
                        cons_dict[mut_str] = row.get("Consequence", "Unknown")
            muts = mutation_lists[idx] if idx < len(mutation_lists) else []
            for mut in muts:
                if mut.upper() == "WT":
                    pathogenicities.append("WT")
                else:
                    pathogenicities.append(cons_dict.get(mut, "Unknown"))

        df_proj = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1],
                                "label": labels, "subunit": subunit,
                                "pathogenicity": pathogenicities})
        df_proj.to_csv(os.path.join(emb_dir, "umap.csv"), index=False)

        # Color UMAP by ClinVar pathogenicity classification
        _PATHOGENICITY_COLORS = {
            "Pathogenic": "#d62728",         # red
            "Likely pathogenic": "#ff7f0e",   # orange
            "VUS": "#7f7f7f",                 # gray
            "Likely benign": "#2ca02c",       # green
            "Benign": "#1f77b4",              # blue
            "Unknown": "#c0c0c0",             # light gray
            "WT": "#000000",                  # black
        }
        _PATHOGENICITY_ORDER = [
            "Pathogenic", "Likely pathogenic", "VUS",
            "Likely benign", "Benign", "Unknown", "WT",
        ]

        plt.figure(figsize=(10, 7))
        plotted = set()
        for i, p in enumerate(pathogenicities):
            color = _PATHOGENICITY_COLORS.get(p, "#c0c0c0")
            label = p if p not in plotted else None
            plotted.add(p)
            plt.scatter(proj[i, 0], proj[i, 1], c=color, label=label, s=70,
                        edgecolors="black", linewidths=0.5, zorder=3)
        # Annotate each point with mutation label
        for i in range(len(labels)):
            plt.annotate(labels[i], (proj[i, 0], proj[i, 1]),
                         fontsize=6, xytext=(4, 4), textcoords="offset points",
                         zorder=4)
        plt.title(f"ESM2 + UMAP variant projection ({config.name})\n"
                  f"Colored by ClinVar pathogenicity", fontsize=12)
        # Build legend in fixed order (only classes that appear)
        handles = []
        from matplotlib.lines import Line2D
        for p in _PATHOGENICITY_ORDER:
            if p in plotted:
                handles.append(Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor=_PATHOGENICITY_COLORS[p],
                                      markeredgecolor="black", markersize=8, label=p))
        plt.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plot = os.path.join(emb_dir, "umap.svg")
        plt.savefig(plot, format="svg")
        plt.savefig(plot.replace(".svg", ".png"), dpi=150)
        plt.close()
        outputs = [os.path.join(emb_dir, "embeddings.npy"),
                   os.path.join(emb_dir, "embedding_labels.csv"),
                   os.path.join(emb_dir, "umap.csv"), plot]
        return StepResult("esm2", "ok",
                          f"Embedded {len(all_seqs)} sequences with ESM2-650M + UMAP",
                          outputs=outputs, data=df_proj)
    except Exception as e:
        return StepResult("esm2", "ok",
                          f"ESM2 embeddings done ({len(all_seqs)} seqs) but UMAP failed: {e}",
                          outputs=[os.path.join(emb_dir, "embeddings.npy")])
