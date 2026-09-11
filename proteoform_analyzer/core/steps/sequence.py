"""Step: sequence + mutation retrieval from UniProt (cached locally).

Ports src/utils.py (download_from_uniprot, get_protein_mutationlist,
filter_mutation_list, get_mutation_consequences) into a prompt-free function.
"""
from __future__ import annotations

import os
import json
import logging
import requests
import pandas as pd
import gradio as gr
from ..pipeline import StepResult

log = logging.getLogger("proteoform_analyzer.sequence")


def _download_fasta(uniprot_id: str, input_dir: str) -> str:
    """Download a FASTA file from UniProt (cached)."""
    path = os.path.join(input_dir, f"{uniprot_id}.fasta")
    if os.path.exists(path):
        log.info("FASTA for %s cached at %s", uniprot_id, path)
        return path
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    os.makedirs(input_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(resp.text)
    log.info("Downloaded FASTA for %s -> %s", uniprot_id, path)
    return path


def read_fasta(fasta_path: str) -> str:
    """Read a FASTA file and return the sequence (header skipped)."""
    lines = []
    with open(fasta_path) as f:
        next(f)  # skip header
        for line in f:
            lines.append(line.strip())
    return "".join(lines)


# ---------------------------------------------------------------------------
# Mature chain (proteolytic processing)
# ---------------------------------------------------------------------------
# UniProt feature types that are cleaved off during maturation. Only features
# touching a sequence terminus are used for terminal trimming.
_CLEAVED_FEATURE_TYPES = {"Signal", "Transit peptide", "Propeptide"}


def _feature_range(feature: dict) -> tuple[int | None, int | None]:
    """(start, end) of a UniProt JSON feature, or (None, None) if absent."""
    location = feature.get("location") or {}
    start = (location.get("start") or {}).get("value")
    end = (location.get("end") or {}).get("value")
    try:
        start = int(start) if start is not None else None
        end = int(end) if end is not None else None
    except (TypeError, ValueError):
        return None, None
    return start, end


def fetch_mature_region(uniprot_id: str, sequence: str | None = None,
                        cache_dir: str | None = None) -> dict:
    """Auto-detect the mature-chain region of a UniProt entry.

    Parses the UniProt JSON feature table. Detection rule:

    * exactly one ``Chain`` feature -> that range (authoritative, covers both
      N- and C-terminal processing, e.g. TTR CHAIN 21-147). Extra ``Peptide``
      features are ignored: they annotate small derived bioactive peptides
      (e.g. hemorphins on hemoglobin), not the main mature product;
    * several ``Chain`` features -> internal excision (e.g. insulin) is NOT
      supported: a warning is recorded and only terminal trimming below is
      applied;
    * no ``Chain`` but exactly one ``Peptide`` feature -> that range;
    * otherwise, terminal trimming: ``Signal`` / ``Transit peptide`` /
      ``Propeptide`` features touching position 1 or the last residue are
      stripped;
    * no usable annotation -> ``region`` is None (full-length; the pipeline
      behaves exactly as before this feature existed).

    Returns a dict with keys ``region`` ((start, end) 1-based inclusive in
    UniProt numbering, or None), ``source`` (human-readable provenance) and
    ``warning`` (str or None). Results are cached as JSON next to the FASTA
    cache when ``cache_dir`` is given.
    """
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{uniprot_id}_mature_region.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                reg = cached.get("region")
                cached["region"] = tuple(reg) if reg else None
                return cached
            except Exception:
                pass  # corrupt cache -> refetch

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    seq_len = len(sequence) if sequence else None
    if seq_len is None:
        seq_len = ((data.get("sequence") or {}).get("length")
                   or len((data.get("sequence") or {}).get("value") or "")) or None

    features = data.get("features", []) or []
    region = None
    source = "full-length"
    warning = None

    chains = [f for f in features if f.get("type") == "Chain"]
    peptides = [f for f in features if f.get("type") == "Peptide"]
    if len(chains) == 1:
        # The single CHAIN is the authoritative mature product; extra PEPTIDE
        # features (derived bioactive peptides, e.g. hemorphins) are ignored.
        s, e = _feature_range(chains[0])
        if s and e:
            region = (s, e)
            source = f"auto: UniProt Chain {s}-{e}"
    elif len(chains) > 1:
        warning = (
            f"{uniprot_id}: {len(chains)} Chain features found — internal "
            "excision (insulin-like processing) is not supported; only terminal "
            "signal/pro-peptide trimming is applied. Set a manual region if needed."
        )
        log.warning(warning)
    elif len(peptides) == 1:
        s, e = _feature_range(peptides[0])
        if s and e:
            region = (s, e)
            source = f"auto: UniProt Peptide {s}-{e}"
    elif len(peptides) > 1:
        warning = (
            f"{uniprot_id}: no Chain feature but {len(peptides)} Peptide features "
            "found — the mature product is ambiguous; only terminal "
            "signal/pro-peptide trimming is applied. Set a manual region if needed."
        )
        log.warning(warning)

    if region is None:
        # Terminal trimming from Signal / Transit peptide / Propeptide features.
        start, end = 1, seq_len
        used = []
        if seq_len:
            for f in features:
                if f.get("type") not in _CLEAVED_FEATURE_TYPES:
                    continue
                s, e = _feature_range(f)
                if not s or not e:
                    continue
                if s <= 1 and e + 1 > start:
                    start = e + 1
                    used.append(f"{f.get('type')} {s}-{e}")
                elif e >= seq_len and s - 1 < end:
                    end = s - 1
                    used.append(f"{f.get('type')} {s}-{e}")
        if used and start <= end:
            region = (start, end)
            source = "auto: terminal trimming (" + "; ".join(used) + ")"

    # Validate against the sequence length when known.
    if region is not None and seq_len:
        s, e = region
        if s < 1 or e > seq_len or s > e:
            warning = (f"{uniprot_id}: detected region {s}-{e} is out of range for "
                       f"sequence length {seq_len}; using full-length")
            log.warning(warning)
            region, source = None, "full-length"
    if region is not None and seq_len and region == (1, seq_len):
        # Annotation covers the whole sequence: equivalent to full-length.
        source += " (full-length)"
        region = None

    out = {"region": region, "source": source, "warning": warning}
    if cache_path:
        try:
            with open(cache_path, "w") as f:
                json.dump(out, f, indent=1)
        except Exception:
            pass
    return out


def mature_sequence(seq: str, region: tuple[int, int] | None) -> str:
    """Return the mature subsequence for a (start, end) region (None = full)."""
    if not region:
        return seq
    s, e = region
    return seq[s - 1:e]


def to_mature_pos(region: tuple[int, int] | None, pos: int) -> int | None:
    """Map a UniProt (precursor) position to mature-chain numbering.

    Returns the 1-based position inside the mature sequence, or None when the
    site lies in a cleaved region (not present in the mature protein).
    """
    if not region:
        return pos
    s, e = region
    if s <= pos <= e:
        return pos - s + 1
    return None


def is_cleaved_site(region: tuple[int, int] | None, pos: int) -> bool:
    """True when ``pos`` (UniProt numbering) falls in a cleaved region."""
    return region is not None and not (region[0] <= pos <= region[1])


def parse_mutation_pos(mut: str) -> int | None:
    """'V50M' -> 50; None for WT/malformed."""
    if not mut or str(mut).upper() == "WT":
        return None
    try:
        return int(str(mut)[1:-1])
    except (ValueError, IndexError):
        return None


def resolve_mature_regions(config, paths: dict | None = None) -> dict:
    """Resolved ``{uniprot_id: (start, end) | None}`` for the current job.

    Precedence (lowest to highest): persisted ``mature_chain_report.csv`` in
    the results dir (lets individual steps be re-run standalone on an existing
    job) < runtime stash ``config._mature_regions`` written by the sequence
    step < manual ``config.mature_regions`` entries (always win).
    """
    uids = list(getattr(config, "uniprot_ids", []) or [])
    resolved: dict[str, tuple[int, int] | None] = {uid: None for uid in uids}

    if paths:
        rep = os.path.join(paths.get("results", "") or "", "mature_chain_report.csv")
        if rep and os.path.exists(rep):
            try:
                df = pd.read_csv(rep)
                for _, row in df.iterrows():
                    uid = str(row.get("uniprot_id", "")).strip()
                    if uid not in resolved:
                        continue
                    s, e = row.get("mature_start"), row.get("mature_end")
                    if pd.notna(s) and pd.notna(e) and int(s) >= 1 and int(e) >= int(s):
                        resolved[uid] = (int(s), int(e))
            except Exception as e:
                log.debug("could not read mature_chain_report.csv: %s", e)

    stash = getattr(config, "_mature_regions", None) or {}
    for uid, reg in stash.items():
        if uid in resolved and reg:
            resolved[uid] = (int(reg[0]), int(reg[1]))

    for uid, reg in (getattr(config, "mature_regions", None) or {}).items():
        if uid not in resolved or not reg:
            continue
        try:
            s, e = int(reg[0]), int(reg[1])
            resolved[uid] = (s, e) if 1 <= s <= e else None
        except Exception:
            log.warning("Unparseable manual mature region for %s: %r; ignored", uid, reg)

    # Last resort: UniProt auto-detection (cached as JSON next to the FASTA
    # cache). This makes every step self-sufficient when it is re-run
    # standalone on an existing job — without the sequence step's runtime
    # stash and without a mature_chain_report.csv in the results dir, regions
    # would otherwise silently degrade to full-length and downstream
    # structure-mapping would use the wrong numbering. Skipped for IDs with a
    # manual entry (explicit user choice) and when auto-detect is disabled.
    if getattr(config, "mature_auto_detect", True):
        manual = getattr(config, "mature_regions", None) or {}
        cache_dir = (paths or {}).get("input") if paths else None
        for uid in uids:
            if resolved.get(uid) is not None or uid in manual:
                continue
            try:
                det = fetch_mature_region(uid, cache_dir=cache_dir)
            except Exception as e:  # offline etc. -> stay full-length
                log.debug("mature-region auto-detect fallback failed for %s: %s",
                          uid, e)
                continue
            reg = det.get("region")
            if reg:
                resolved[uid] = (int(reg[0]), int(reg[1]))
                log.info("%s: mature chain %d-%d resolved by auto-detection (%s)",
                         uid, resolved[uid][0], resolved[uid][1],
                         det.get("source", "auto"))

    # Refresh the runtime stash so later steps in the same process reuse these
    # regions even if the sequence step itself did not run.
    try:
        config._mature_regions = dict(resolved)
    except Exception:  # pragma: no cover - defensive
        pass
    return resolved


def get_mature_sequences(config, paths: dict | None = None) -> list[str] | None:
    """Mature sequences aligned with ``config.uniprot_ids`` (None if unknown).

    Uses the sequence step's stash when available, else the precursor
    sequences on ``config._sequences`` (or the FASTA cache under
    ``paths['input']``) trimmed by the resolved regions.
    """
    stashed = getattr(config, "_mature_sequences", None)
    if stashed:
        return list(stashed)
    seqs = getattr(config, "_sequences", None)
    if not seqs and paths:
        seqs = []
        for uid in getattr(config, "uniprot_ids", []) or []:
            fp = os.path.join(paths.get("input", "") or "", f"{uid}.fasta")
            if not os.path.exists(fp):
                return None
            seqs.append(read_fasta(fp))
    if not seqs:
        return None
    regions = resolve_mature_regions(config, paths)
    uids = list(getattr(config, "uniprot_ids", []) or [])
    return [mature_sequence(s, regions.get(uid)) for uid, s in zip(uids, seqs)]


def structural_mutation_lists(config, paths: dict | None = None):
    """Mutation lists restricted to sites present in the mature protein.

    Returns ``(filtered_lists, dropped)`` where ``filtered_lists`` mirrors
    ``config._mutation_lists`` (WT kept) minus mutations in cleaved regions,
    and ``dropped`` is a list of ``"MUT (uid)"`` strings that were excluded.
    """
    mut_lists = getattr(config, "_mutation_lists", None) or []
    regions = resolve_mature_regions(config, paths)
    uids = list(getattr(config, "uniprot_ids", []) or [])
    filtered, dropped = [], []
    for idx, uid in enumerate(uids):
        region = regions.get(uid)
        muts = mut_lists[idx] if idx < len(mut_lists) else []
        keep = []
        for m in muts:
            pos = parse_mutation_pos(m)
            if pos is not None and is_cleaved_site(region, pos):
                dropped.append(f"{m} ({uid})")
                continue
            keep.append(m)
        filtered.append(keep)
    return filtered, dropped


def get_protein_mutationlist(uniprot_id: str) -> list[str]:
    """Fetch natural single-point variants from UniProt JSON API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    mutations = []
    for feature in data.get("features", []):
        if feature.get("type") != "Natural variant":
            continue
        location = feature.get("location", {})
        start = location.get("start", {}).get("value")
        end = location.get("end", {}).get("value")
        if start != end:
            continue  # skip non-point mutations
        alt = feature.get("alternativeSequence", {})
        original = alt.get("originalSequence")
        variants = alt.get("alternativeSequences") or []
        if not original:
            continue
        for v in variants:
            v = v.strip()
            if start and original and v:
                mutations.append(f"{original}{start}{v}")
    return mutations


def filter_mutation_list(mutation_list: list[str], sequence: str) -> list[str]:
    """Keep only single-point mutations whose WT aa matches the sequence."""
    filtered = []
    for mut in mutation_list:
        try:
            parts = mut.split()
            mut_str = parts[-1] if len(parts) >= 1 else mut
            wt_aa = mut_str[0]
            position = int(mut_str[1:-1])
            new_aa = mut_str[-1]
            if 1 <= position <= len(sequence) and sequence[position - 1] == wt_aa:
                filtered.append(f"{wt_aa}{position}{new_aa}")
        except (ValueError, IndexError):
            continue
    return filtered


# 5-tier ClinVar pathogenicity classification
_PATHOGENICITY_MAP = {
    "Pathogenic": "Pathogenic",
    "Likely pathogenic": "Likely pathogenic",
    "Benign": "Benign",
    "Likely benign": "Likely benign",
    "Variant of uncertain significance": "VUS",
}


def _normalize_clinical(ctype: str) -> str:
    """Normalize a clinical significance string to one of the 5 tiers or 'Unknown'."""
    if not ctype:
        return "Unknown"
    # Try exact match first
    if ctype in _PATHOGENICITY_MAP:
        return _PATHOGENICITY_MAP[ctype]
    # Try case-insensitive match
    for key, val in _PATHOGENICITY_MAP.items():
        if key.lower() == ctype.lower():
            return val
    # Try substring match (handles variations like "Pathogenic/Likely pathogenic")
    lower = ctype.lower()
    if "pathogenic" in lower and "likely" not in lower:
        return "Pathogenic"
    if "likely pathogenic" in lower:
        return "Likely pathogenic"
    if "benign" in lower and "likely" not in lower:
        return "Benign"
    if "likely benign" in lower:
        return "Likely benign"
    if "uncertain" in lower or "vus" in lower:
        return "VUS"
    return "Unknown"


def get_mutation_consequences(uniprot_id: str, sequence: str) -> pd.DataFrame:
    """Fetch ClinVar clinical significance from the EBI Proteins API.

    Uses a single API call (instead of per-position) to fetch all variants
    for the protein. Returns a DataFrame indexed by mutation string with a
    'Consequence' column containing one of the 5-tier classifications:
    Pathogenic, Likely pathogenic, VUS, Likely benign, Benign, or Unknown.
    """
    url = f"https://www.ebi.ac.uk/proteins/api/variation?offset=0&size=500&accession={uniprot_id}"
    consequences = {}
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            log.warning("EBI variation API returned %d for %s", resp.status_code, uniprot_id)
            df = pd.DataFrame(columns=["Consequence"])
            df.index.name = "Mutation"
            return df
        data = resp.json()
        if not data:
            df = pd.DataFrame(columns=["Consequence"])
            df.index.name = "Mutation"
            return df

        features = data[0].get("features") or []
        for feature in features:
            wt_aa = feature.get("wildType")
            mut_aa = feature.get("mutatedType")
            pos = feature.get("begin")
            if not (wt_aa and mut_aa and pos):
                continue
            if len(mut_aa) != 1 or len(wt_aa) != 1:
                continue
            try:
                pos_int = int(pos)
            except (ValueError, TypeError):
                continue
            # Validate against sequence
            if pos_int < 1 or pos_int > len(sequence):
                continue
            if sequence[pos_int - 1] != wt_aa:
                continue
            mutation = f"{wt_aa}{pos_int}{mut_aa}"
            clinical = feature.get("clinicalSignificances") or []
            if clinical:
                ctype = clinical[0].get("type", "Unknown")
            else:
                ctype = "Unknown"
            consequences[mutation] = {"Consequence": _normalize_clinical(ctype)}
    except Exception as e:
        log.warning("EBI variation API fetch failed for %s: %s", uniprot_id, e)

    df = pd.DataFrame.from_dict(consequences, orient="index")
    if df.empty:
        df = pd.DataFrame(columns=["Consequence"])
    df.index.name = "Mutation"
    return df

def _fetch_mutations_from_uniprot(uniprot_ids_text: str):
    """
    Fetch all UniProt natural variants for every comma-separated UniProt ID.

    Returns a pipe-separated string with exactly one mutation group per input
    UID. Does not cap, duplicate, redistribute, or append WT: that belongs to
    fetch_sequences_and_mutations().
    """
    if not uniprot_ids_text or not uniprot_ids_text.strip():
        raise gr.Error("Please enter at least one UniProt accession.")

    uids = [
        uid.strip().upper()
        for uid in uniprot_ids_text.split(",")
        if uid.strip()
    ]

    if not uids:
        raise gr.Error("Please enter at least one valid UniProt accession.")

    mutation_groups = []
    total_found = 0

    for uid in uids:
        try:
            muts = get_protein_mutationlist(uid)
        except requests.RequestException as exc:
            raise gr.Error(f"Could not fetch mutations for {uid}: {exc}") from exc

        # get_protein_mutationlist() must return list[str].
        if muts is None:
            muts = []
        elif isinstance(muts, str):
            # Defensive only; normal behaviour should be list[str].
            muts = muts.split()
        else:
            muts = list(muts)

        muts = [
            str(mut).strip()
            for mut in muts
            if str(mut).strip() and str(mut).strip().upper() != "WT"
        ]

        mutation_groups.append(muts)
        total_found += len(muts)

    # Exactly one group per input UID. This is what downstream config parsing
    # expects for monomer, homo-oligomer, and hetero-oligomer use cases.
    mutations_text = " | ".join(
        " ".join(group) for group in mutation_groups
    )

    # Slider range needs to include every available mutation.
    # Value is left at 10 when possible, otherwise clipped to total found.
    slider_max = max(10, total_found)
    slider_value = min(10, slider_max)

    return (
        mutations_text,
        gr.update(
            minimum=-1,
            maximum=slider_max,
            value=slider_value,
            label=f"Max mutations (found: {total_found})",
        ),
    )
def fetch_sequences_and_mutations(config, paths: dict) -> StepResult:
    """Fetch sequences for all uniprot_ids, validate mutations, get consequences."""
    outputs = []
    sequences = []
    all_mutation_lists = []
    consequences_dfs = []
    for uid in config.uniprot_ids:
        fasta_path = _download_fasta(uid, paths["input"])
        seq = read_fasta(fasta_path)
        sequences.append(seq)
        log.info("%s: sequence length %d", uid, len(seq))

        # mutation list: use config if provided, else fetch from UniProt
        idx = config.uniprot_ids.index(uid)
        if config.mutations and idx < len(config.mutations) and config.mutations[idx]:
            muts = config.mutations[idx]
            muts = filter_mutation_list(muts, seq)
            log.info("%s: using %d user-provided mutations (after validation)", uid, len(muts))
        else:
            muts = get_protein_mutationlist(uid)
            muts = filter_mutation_list(muts, seq)
            log.info("%s: fetched %d mutations from UniProt", uid, len(muts))

        # cap to max_mutations (per subunit, split evenly for hetero)
        if config.max_mutations is not None:
            if config.is_hetero and len(config.uniprot_ids) > 1:
                per = config.max_mutations // len(config.uniprot_ids)
                muts = muts[:per]
            else:
                muts = muts[:config.max_mutations]
            log.info("%s: capped to %d mutations", uid, len(muts))

        muts.append("WT")
        all_mutation_lists.append(muts)

        # save mutation list
        mut_file = os.path.join(paths["results"], f"{uid}_mutations.txt")
        with open(mut_file, "w") as f:
            for m in muts:
                f.write(m + "\n")
        outputs.append(mut_file)

        # consequences
        cons_file = os.path.join(paths["results"], f"{uid}_mutation_consequences.csv")
        if os.path.exists(cons_file):
            df_cons = pd.read_csv(cons_file, index_col=0)
        else:
            df_cons = get_mutation_consequences(uid, seq)
            df_cons.to_csv(cons_file)
        consequences_dfs.append(df_cons)
        outputs.append(cons_file)

    # --- Mature chain (proteolytic processing) ---------------------------
    # Resolve the mature region per subunit (manual config entry wins over
    # UniProt auto-detection), build the mature sequences actually used for
    # folding / embedding / docking, and flag mutations that fall in cleaved
    # regions (they stay in the mutation tables but are excluded from all
    # structural steps).
    mature_regions: dict = {}
    report_rows = []
    for idx, uid in enumerate(config.uniprot_ids):
        seq = sequences[idx]
        region, source, warning = None, "full-length", None
        manual = (getattr(config, "mature_regions", None) or {}).get(uid)
        if manual:
            try:
                s, e = int(manual[0]), int(manual[1])
                if 1 <= s <= e <= len(seq):
                    region, source = (s, e), "manual"
                    if (s, e) == (1, len(seq)):
                        region, source = None, "manual (full-length)"
                else:
                    warning = (f"manual mature region {s}-{e} out of range for "
                               f"{uid} (length {len(seq)}); using full-length")
                    log.warning(warning)
            except Exception:
                warning = (f"unparseable manual mature region for {uid}: "
                           f"{manual!r}; using full-length")
                log.warning(warning)
        elif getattr(config, "mature_auto_detect", True):
            try:
                det = fetch_mature_region(uid, sequence=seq,
                                          cache_dir=paths.get("input"))
                region, source = det.get("region"), det.get("source", "full-length")
                warning = det.get("warning")
            except Exception as e:
                log.warning("Mature-region auto-detection failed for %s: %s; "
                            "using full-length", uid, e)
        mature_regions[uid] = region

        muts = all_mutation_lists[idx] if idx < len(all_mutation_lists) else []
        cleaved = [m for m in muts
                   if parse_mutation_pos(m) is not None
                   and is_cleaved_site(region, parse_mutation_pos(m))]
        if cleaved:
            log.warning("%s: %d mutation(s) in the cleaved region (%s) are "
                        "excluded from structural steps: %s",
                        uid, len(cleaved),
                        f"outside {region[0]}-{region[1]}", " ".join(cleaved))
        report_rows.append({
            "uniprot_id": uid,
            "precursor_length": len(seq),
            "mature_start": region[0] if region else "",
            "mature_end": region[1] if region else "",
            "mature_length": (region[1] - region[0] + 1) if region else len(seq),
            "source": source,
            "n_mutations": len([m for m in muts if str(m).upper() != "WT"]),
            "n_cleaved_region": len(cleaved),
            "cleaved_mutations": " ".join(cleaved),
            "warning": warning or "",
        })

    config._mature_regions = mature_regions
    config._mature_sequences = [
        mature_sequence(seq, mature_regions.get(uid))
        for uid, seq in zip(config.uniprot_ids, sequences)
    ]
    report_path = os.path.join(paths["results"], "mature_chain_report.csv")
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    outputs.append(report_path)

    # stash on config for downstream steps
    config._sequences = sequences
    config._mutation_lists = all_mutation_lists
    config._consequences = consequences_dfs

    n_muts = sum(len(m) - 1 for m in all_mutation_lists)  # exclude WT
    trimmed = {uid: reg for uid, reg in mature_regions.items() if reg}
    n_cleaved = sum(r["n_cleaved_region"] for r in report_rows)
    msg = (f"Retrieved {len(sequences)} sequences, {n_muts} mutations, "
           f"{len(consequences_dfs)} consequence tables")
    if trimmed:
        regs = "; ".join(f"{uid} {s}-{e}" for uid, (s, e) in trimmed.items())
        msg += f"; mature chain: {regs}"
        if n_cleaved:
            msg += (f"; {n_cleaved} mutation(s) in cleaved regions flagged and "
                    f"excluded from structural steps")
    return StepResult(
        "sequence", "ok", msg,
        outputs=outputs,
        data={"sequences": sequences, "mutations": all_mutation_lists,
              "consequences": consequences_dfs,
              "mature_regions": mature_regions,
              "mature_sequences": config._mature_sequences},
    )
