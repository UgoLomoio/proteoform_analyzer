"""Step: PTM generation via ptmpsi.

Applies post-translational modifications to specified residues on WT and mutant
structures, on the configured chains.

PTM sites are resolved with the following precedence (see ``PTMConfig``):

1. ``config.ptm.pairs`` — explicit (residue, ptm_type, uniprot_id|None) triples
   from the GUI PTM builder. Each pair is validated against the ptmpsi
   compatibility rules (:mod:`proteoform_analyzer.core.ptm_rules`); impossible
   combinations are skipped with a warning instead of failing at runtime.
2. ``config.ptm.residues`` x ``config.ptm.ptm_types`` — legacy cross-product,
   now also validated against the compatibility rules.
3. Both empty — *auto-fetch*: experimentally observed PTMs are retrieved from
   UniProt feature annotations ("Modified residue", "Cross-link", "Lipidation",
   "Glycosylation"), filtered to the ones ptmpsi can model, and capped at
   ``config.max_mutations`` sites (no cap when max_mutations is None). If no
   observed/modelable PTM exists, the step is skipped.
"""
from __future__ import annotations

import os
import sys
import json
import logging

from ..pipeline import StepResult
from ..ptm_rules import (
    compatible_ptms, ptm_allowed, map_uniprot_ptm, parse_residue_spec,
    UNIPROT_PTM_FEATURE_TYPES,
)

log = logging.getLogger("proteoform_analyzer.ptm")


def _vendored_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "_vendored")


def _import_ptmpsi():
    v = _vendored_path()
    if v not in sys.path:
        sys.path.insert(0, v)
    from ptmpsi.protein import Protein
    return Protein


# ---------------------------------------------------------------------------
# UniProt observed-PTM fetching
# ---------------------------------------------------------------------------

def fetch_observed_ptms(uniprot_id: str, sequence: str | None = None,
                        cache_dir: str | None = None) -> list[tuple[str, str]]:
    """Fetch experimentally observed PTMs for a UniProt entry.

    Parses the UniProt JSON feature table, keeps only features whose
    description maps onto a ptmpsi-modelable PTM *and* whose residue is
    compatible with that PTM (verified against the sequence when provided).
    Results are cached as JSON next to the FASTA cache when ``cache_dir`` is
    given.

    Returns a list of ``(residue_spec, ptm_type)`` e.g. ``("SER15",
    "phosphorylation")``, sorted by position.
    """
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{uniprot_id}_observed_ptms.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    return [tuple(x) for x in json.load(f)]
            except Exception:
                pass  # corrupt cache -> refetch

    import requests
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # The UniProt JSON embeds the canonical sequence; use it to validate
    # residue identity when the caller did not supply a sequence.
    if not sequence:
        sequence = (data.get("sequence") or {}).get("value")

    pairs: list[tuple[str, str]] = []
    seen = set()
    for feature in data.get("features", []):
        if feature.get("type") not in UNIPROT_PTM_FEATURE_TYPES:
            continue
        ptm = map_uniprot_ptm(feature.get("description", ""))
        if ptm is None:
            continue
        location = feature.get("location", {})
        start = (location.get("start") or {}).get("value")
        end = (location.get("end") or {}).get("value")
        if not start or start != end:
            continue  # only single-residue PTMs are modelable
        pos = int(start)
        aa3 = None
        if sequence and 1 <= pos <= len(sequence):
            from ..ptm_rules import one_to_three
            aa3 = one_to_three(sequence[pos - 1])
        if aa3 is None:
            continue
        if not ptm_allowed(aa3, ptm, n_terminal=(pos == 1)):
            log.info("UniProt PTM '%s' at %s%d of %s is not modelable by "
                     "ptmpsi on that residue; skipping",
                     feature.get("description"), aa3, pos, uniprot_id)
            continue
        key = (f"{aa3}{pos}", ptm)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    pairs.sort(key=lambda x: (int(x[0][3:]), x[1]))
    if cache_path:
        try:
            with open(cache_path, "w") as f:
                json.dump(pairs, f, indent=1)
        except Exception:
            pass
    return pairs


# ---------------------------------------------------------------------------
# Site resolution (precedence: pairs -> cross-product -> auto-fetch)
# ---------------------------------------------------------------------------

def _sequences_by_uid(config, paths: dict) -> dict:
    """Precursor sequences per UniProt ID: runtime stash, else cached FASTA."""
    seqs = getattr(config, "_sequences", None) or []
    out = {}
    for i, uid in enumerate(getattr(config, "uniprot_ids", []) or []):
        if i < len(seqs) and seqs[i]:
            out[uid] = seqs[i]
            continue
        try:
            from .sequence import _download_fasta, read_fasta
            out[uid] = read_fasta(_download_fasta(uid, paths.get("input") or "."))
        except Exception:
            pass
    return out


def _normalize_site_to_uniprot(aa3: str, pos: int, uid: str, region, seq):
    """Re-number a PTM site given in MATURE-chain numbering to UniProt numbering.

    PTM sites are documented and validated in UniProt (precursor) numbering,
    but the literature on proteolytically matured proteins often uses
    mature-chain numbering (e.g. TTR "Cys10" S-nitrosylation = UniProt Cys30).
    When the expected residue does NOT match the precursor sequence at ``pos``
    but DOES match the mature sequence there, the site is interpreted as
    mature-numbered and converted (``pos + region[0] - 1``). Identity is
    verified against both sequences, so a UniProt-numbered site is never
    touched. Returns ``(new_pos, note_or_None)``.
    """
    if not (region and seq and 1 <= pos <= len(seq)):
        return pos, None
    from ..ptm_rules import one_to_three
    if one_to_three(seq[pos - 1]) == aa3:
        return pos, None  # already UniProt (precursor) numbered
    mseq = seq[region[0] - 1:region[1]]
    if 1 <= pos <= len(mseq) and one_to_three(mseq[pos - 1]) == aa3:
        new_pos = pos + region[0] - 1
        return new_pos, (f"site {aa3}{pos} does not match the {uid} precursor "
                         f"sequence but matches the mature chain at that "
                         f"position; interpreted as mature-chain numbering -> "
                         f"UniProt {aa3}{new_pos}")
    return pos, None


def _resolve_sites(config, paths: dict) -> list[tuple[str, str, str | None]]:
    """Return validated (residue_spec, ptm_type, uniprot_id|None) triples."""
    ptm_cfg = config.ptm

    # Mature (post-cleavage) regions: sites in proteolytically removed segments
    # are absent from the mature protein and can never be modified.
    try:
        from .sequence import resolve_mature_regions, is_cleaved_site
        _mature_regions = resolve_mature_regions(config, paths)
    except Exception:  # pragma: no cover - defensive
        _mature_regions = {}
        is_cleaved_site = lambda region, pos: False  # noqa: E731

    # 1) Explicit pairs from the GUI builder / CLI
    if ptm_cfg.pairs:
        out = []
        _seqs = None  # lazily resolved only when a site needs renumbering
        for triple in ptm_cfg.pairs:
            residue, ptm_type = triple[0], triple[1]
            uid = triple[2] if len(triple) > 2 else None
            parsed = parse_residue_spec(str(residue))
            if parsed is None:
                log.warning("PTM pair %s: unparseable residue spec; skipped", triple)
                continue
            aa3, _pos = parsed
            if not ptm_allowed(aa3, str(ptm_type)):
                log.warning("PTM pair %s: '%s' is not possible on %s (allowed: %s); "
                            "skipped", triple, ptm_type, aa3,
                            ", ".join(compatible_ptms(aa3)) or "none")
                continue
            # Sites are UniProt-numbered, but literature on matured proteins
            # often uses mature-chain numbering (e.g. TTR Cys10 = UniProt
            # Cys30): detect that case by residue identity and renumber.
            _region = _mature_regions.get(uid) if uid else None
            if uid and _region:
                if _seqs is None:
                    _seqs = _sequences_by_uid(config, paths)
                _pos, _note = _normalize_site_to_uniprot(
                    aa3, _pos, uid, _region, _seqs.get(uid))
                if _note:
                    log.info("PTM pair %s: %s", triple, _note)
            if uid and is_cleaved_site(_mature_regions.get(uid), _pos):
                reg = _mature_regions[uid]
                log.warning(
                    "PTM pair %s: site %s%d lies in the proteolytically cleaved "
                    "region of %s (mature chain %d-%d); skipped",
                    triple, aa3, _pos, uid, reg[0], reg[1])
                continue
            out.append((f"{aa3}{_pos}", str(ptm_type).lower().strip(), uid))
        return out

    # 2) Legacy cross-product residues x ptm_types
    if ptm_cfg.residues and ptm_cfg.ptm_types:
        out = []
        for residue in ptm_cfg.residues:
            parsed = parse_residue_spec(str(residue))
            if parsed is None:
                log.warning("PTM: could not parse residue spec '%s' "
                            "(expected e.g. 'CYS10'); skipping", residue)
                continue
            aa3, pos = parsed
            for ptm_type in ptm_cfg.ptm_types:
                ptm_type = str(ptm_type).lower().strip()
                if not ptm_allowed(aa3, ptm_type):
                    log.warning("PTM '%s' is not possible on %s (allowed: %s); "
                                "skipping this combination", ptm_type, residue,
                                ", ".join(compatible_ptms(aa3)) or "none")
                    continue
                out.append((f"{aa3}{pos}", ptm_type, None))
        return out

    # 3) Auto-fetch observed PTMs from UniProt
    sequences = getattr(config, "_sequences", None) or []
    fetched: list[tuple[str, str, str | None]] = []
    for i, uid in enumerate(config.uniprot_ids):
        seq = sequences[i] if i < len(sequences) else None
        try:
            obs = fetch_observed_ptms(uid, sequence=seq, cache_dir=paths.get("input"))
        except Exception as e:
            log.warning("Could not fetch observed PTMs for %s: %s", uid, e)
            continue
        region = _mature_regions.get(uid)
        if region is not None:
            before = len(obs)
            obs = [(res, ptm) for res, ptm in obs
                   if not is_cleaved_site(region, parse_residue_spec(res)[1]
                                          if parse_residue_spec(res) else -1)]
            n_dropped = before - len(obs)
            if n_dropped:
                log.info("%s: %d observed PTM site(s) lie in the cleaved region "
                         "(mature chain %d-%d) and were excluded",
                         uid, n_dropped, region[0], region[1])
        log.info("%s: %d observed, ptmpsi-modelable PTM site(s) from UniProt",
                 uid, len(obs))
        fetched.extend((res, ptm, uid) for res, ptm in obs)

    # Cap at the same number as the mutations (max_mutations), if set
    if config.max_mutations is not None and len(fetched) > config.max_mutations:
        log.info("Capping auto-fetched PTM sites from %d to max_mutations=%d",
                 len(fetched), config.max_mutations)
        fetched = fetched[: config.max_mutations]
    return fetched


def _chains_for_uid(config, uid: str | None) -> list[str]:
    """Chains a PTM pair applies to: the uid's chains when known, else all."""
    if uid and config.chain_map.get(uid):
        return list(config.chain_map[uid])
    return list(config.ptm.chains)


# ---------------------------------------------------------------------------
# ptmpsi application
# ---------------------------------------------------------------------------

class _HisDefault:
    """Guard for ptmpsi's interactive histidine prompt.

    ptmpsi's ``doptm`` calls ``input()`` to ask which histidine nitrogen
    (pros/tele) reacts; in a GUI/CLI pipeline that would block forever (or
    raise EOFError). This context manager feeds a deterministic default
    ("tele", the more common tautomer) for the duration of one modify() call.
    """

    def __enter__(self):
        import builtins
        self._orig_input = builtins.input
        builtins.input = lambda *a, **k: "tele"
        return self

    def __exit__(self, *exc):
        import builtins
        builtins.input = self._orig_input
        return False


def generate_ptms(config, paths: dict) -> StepResult:
    """Generate PTM-modified structures for WT and mutants."""
    if not config.run_ptm:
        return StepResult("ptm", "skipped", "run_ptm=False")

    sites = _resolve_sites(config, paths)
    if not sites:
        return StepResult(
            "ptm", "skipped",
            "No PTM sites configured and no observed, ptmpsi-modelable PTMs "
            "found on UniProt for the configured protein(s); PTM analysis skipped.")

    Protein = _import_ptmpsi()
    ptms_out = os.path.join(paths["ptms"], "ptms")
    os.makedirs(ptms_out, exist_ok=True)

    # gather all input PDBs (WT + mutants) from the pdbs dir.
    # Structures resolve through boltz-experiments first (see _structure_source).
    pdb_dir = paths["pdbs_monomer"] if config.is_monomer else paths["pdbs"]
    from ._structure_source import iter_structure_pdbs
    pdbs = dict(iter_structure_pdbs(config, paths, "ptm"))

    if not pdbs:
        return StepResult("ptm", "skipped", f"No PDBs found in {pdb_dir}")

    from ._resnum import (map_site_to_structure_candidates, find_chain_offset,
                          parse_pdb_residues)

    # Mature (post-cleavage) regions: structures folded from the mature
    # sequence (Boltz v3.6+) — and most reference PDBs of matured proteins —
    # are numbered in MATURE coordinates, while PTM sites are UniProt
    # (precursor) numbered. For each site we try the mature position first,
    # then the UniProt one; every candidate is name-verified by
    # map_site_to_structure, so a candidate only wins on a residue-identity
    # match. Cleaved-region sites were already filtered in _resolve_sites.
    try:
        from .sequence import resolve_mature_regions, to_mature_pos
        _mature_regions = resolve_mature_regions(config, paths) or {}
    except Exception:  # pragma: no cover - defensive
        _mature_regions = {}
        to_mature_pos = lambda region, pos: pos  # noqa: E731

    # Pre-compute per-chain numbering offsets for every structure, as the
    # proteoform/graft steps do: reference PDBs are author-numbered and often
    # shifted from the sequence numbering the sites use. This is a safety net
    # on top of the mature/UniProt candidates (all name-verified).
    chain_site_pairs: dict = {}
    for _residue, _ptm_type, _uid in sites:
        _parsed = parse_residue_spec(_residue)
        if _parsed is None:
            continue
        for _ch in _chains_for_uid(config, _uid):
            chain_site_pairs.setdefault(_ch, []).append(_parsed)
    offsets_by_pdb: dict = {}
    try:
        for _name, _pdb in pdbs.items():
            _parsed_pdb = parse_pdb_residues(_pdb)
            offsets_by_pdb[_pdb] = {
                ch: find_chain_offset(_parsed_pdb.get(ch), sp)
                for ch, sp in chain_site_pairs.items()
            }
    except Exception as e:  # pragma: no cover - defensive
        log.debug("ptm: chain offset pre-computation skipped: %s", e)
        offsets_by_pdb = {}

    # PTM sites are written as "<AA3><seqnum>" (e.g. "CYS10") in UniProt sequence
    # numbering, but the reference/mutant PDBs are author-numbered. Each site is
    # resolved to the ordinal-based selector PTM-Psi expects, verifying the
    # residue identity so we never modify the wrong residue. Sites that can't be
    # verified are skipped with a warning.
    outputs = []
    n_ptms = 0
    n_skipped = 0
    for residue, ptm_type, uid in sites:
        parsed = parse_residue_spec(residue)
        if parsed is None:
            continue
        expected_aa3, seq_pos = parsed
        chains = _chains_for_uid(config, uid)
        for mutant_name, pdb_path in pdbs.items():
            out_file = os.path.join(ptms_out, f"{mutant_name}_{ptm_type}_{residue}.pdb")
            if os.path.exists(out_file):
                outputs.append(out_file)
                n_ptms += 1
                continue
            try:
                p = Protein(filename=pdb_path)
                applied = 0
                mature_pos = to_mature_pos(_mature_regions.get(uid), seq_pos)
                _offsets = offsets_by_pdb.get(pdb_path) or {}
                for chain in chains:
                    mapped = map_site_to_structure_candidates(
                        expected_aa3, [mature_pos, seq_pos], chain,
                        protein=p, pdb_path=pdb_path,
                        offset=_offsets.get(chain),
                    )
                    if mapped is None:
                        log.warning(
                            "PTM %s: site %s%d not found on chain %s of %s "
                            "(residue identity/number could not be verified in "
                            "mature or UniProt numbering); skipping this chain",
                            ptm_type, expected_aa3, seq_pos, chain, mutant_name)
                        continue
                    selector, _ordinal = mapped
                    try:
                        with _HisDefault():
                            p.modify(selector, ptm_type)
                        applied += 1
                    except Exception as e:
                        log.warning("PTM %s on %s chain %s (selector %s) failed: %s",
                                    ptm_type, residue, chain, selector, e)
                if applied == 0:
                    n_skipped += 1
                    log.warning("PTM %s @ %s on %s: site not verifiable on any chain; "
                                "no PTM written", ptm_type, residue, mutant_name)
                    continue
                p.write_pdb(out_file)
                outputs.append(out_file)
                n_ptms += 1
                log.info("PTM %s @ %s on %s -> %s", ptm_type, residue, mutant_name, out_file)
            except Exception as e:
                log.error("PTM %s @ %s on %s failed: %s", ptm_type, residue, mutant_name, e)

    if n_ptms == 0:
        return StepResult(
            "ptm", "skipped",
            f"None of the {len(sites)} configured/observed PTM site(s) could be "
            f"verified on the available structures; PTM analysis skipped.",
            outputs=outputs)

    return StepResult(
        "ptm", "ok",
        f"Generated {n_ptms} PTM structures ({len(sites)} site(s) x "
        f"{len(pdbs)} structures; {n_skipped} site/structure combos unverifiable)",
        outputs=outputs,
    )
