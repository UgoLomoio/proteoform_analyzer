"""Graft-fallback structure seeding (no GPU, no API, no HPC).

This is the honest last-resort backend for the *folding* step when neither a
Boltz API key, a local ``boltz`` binary is available. 
Instead of *folding* structures from sequence, it:

  1. fetches a reference experimental structure for the assembly
     (``config.local_pdb_id``, e.g. ``1A3N`` for hemoglobin) and writes it to
     the canonical **WT** PDB path, and
  2. produces each **mutant** PDB by side-chain grafting the point mutation onto
     that same WT backbone with PTM-Psi (``Protein(...).mutate().write_pdb()``).

Because every mutant shares the WT backbone atom-for-atom, this path is
**backbone-identical by construction**: TM-score == 1.0, no pocket drift, no
docking/structural signal. It exists so the rest of the pipeline (PTM-Psi
proteoforms, ESM2 sequence divergence, ddG, PCN topology on the grafted side
chains) can still run end-to-end offline. Callers MUST surface
``_boltz_backend.GRAFT_WARNING`` to the user and record provenance via
``_boltz_backend.write_structure_provenance(..., method="graft",
backbone_identical=True)``.

Kept separate from ``boltz2_fold`` so it can be unit-tested in isolation and so
the (heavy) PTM-Psi import stays lazy.
"""
from __future__ import annotations

import os
import shutil
import logging

log = logging.getLogger("proteoform_analyzer.graft_seed")

# 1-letter -> 3-letter, shared with proteoform grafting.
_AA3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


def _import_ptmpsi_protein():
    """Lazy import of ptmpsi Protein (raises ImportError if unavailable).

    Reuses ``proteoform._import_ptmpsi`` so the vendored ptmpsi copy under
    ``_vendored/`` is placed on ``sys.path`` exactly the same way the proteoform
    grafting step does.
    """
    from .proteoform import _import_ptmpsi
    return _import_ptmpsi()


def graft_available(config) -> bool:
    """True iff a graft seed *could* be produced: a reference PDB id is set and
    PTM-Psi is importable. Side-effect free; never raises."""
    if not getattr(config, "local_pdb_id", None):
        return False
    try:
        _import_ptmpsi_protein()
    except Exception:
        return False
    return True


def _parse_point_mutation(mut: str):
    """'V1M' -> ('V', '1', 'M'); returns None for WT/malformed."""
    if not mut or mut.upper() == "WT" or len(mut) < 3:
        return None
    orig = mut[0].upper()
    new = mut[-1].upper()
    pos = mut[1:-1]
    if orig not in _AA3 or new not in _AA3 or not pos.isdigit():
        return None
    return orig, pos, new


def build_graft_structures(config, paths: dict, worklist, canonical_pdb_path) -> dict:
    """Create canonical WT + mutant PDBs by reference fetch + PTM-Psi grafting.

    Parameters
    ----------
    config : pipeline config (needs ``local_pdb_id``; chain map derived below).
    paths : path dict (uses ``boltz_structures`` for the reference cache dir).
    worklist : iterable of ``(uid, mut)`` or ``(uid, mut, idx)`` tuples, exactly
        as assembled by ``boltz2_fold.build_structures``. WT is the item whose
        ``mut == 'WT'``.
    canonical_pdb_path : callable ``(config, paths, uid, mut) -> path`` — pass in
        ``boltz2_fold._canonical_pdb_path`` so both paths stay identical.

    Returns
    -------
    dict with keys: ``wt_pdb`` (str|None), ``outputs`` (list[str]),
    ``n_done`` (int), ``errors`` (list[str]).
    """
    from .structure import _get_chain_map
    from ._pdb_utils import fetch_reference_pdb

    result = {"wt_pdb": None, "outputs": [], "n_done": 0, "errors": []}

    pdb_id = getattr(config, "local_pdb_id", None)
    if not pdb_id:
        result["errors"].append("graft fallback: config.local_pdb_id is not set")
        return result

    # --- 1. Fetch reference structure and place it at the canonical WT path ---
    ref_dir = os.path.join(paths["boltz_structures"], "graft_reference")
    try:
        ref_pdb = fetch_reference_pdb(pdb_id, ref_dir)
    except Exception as e:
        result["errors"].append(f"graft fallback: could not fetch reference {pdb_id}: {e}")
        return result

    wt_uid = None
    for item in worklist:
        if item[1].upper() == "WT":
            wt_uid = item[0]
            break
    if wt_uid is None:
        wt_uid = config.uniprot_ids[0]

    wt_out = canonical_pdb_path(config, paths, wt_uid, "WT")
    if not os.path.exists(wt_out):
        try:
            shutil.copyfile(ref_pdb, wt_out)
        except Exception as e:
            result["errors"].append(f"graft fallback: could not write WT PDB: {e}")
            return result
    result["wt_pdb"] = wt_out
    result["outputs"].append(wt_out)
    result["n_done"] += 1
    log.info("Graft fallback: WT seed %s -> %s", pdb_id, os.path.basename(wt_out))

    # --- 2. Graft each point mutation onto the WT backbone via PTM-Psi ---
    try:
        Protein = _import_ptmpsi_protein()
    except Exception as e:
        result["errors"].append(f"graft fallback: PTM-Psi unavailable ({e}); "
                                "only WT seed produced")
        return result

    from ._resnum import (map_site_to_structure, map_site_to_structure_candidates,
                          find_chain_offset, parse_pdb_residues)

    chain_map = _get_chain_map(config)

    # Resolve mature (post-cleavage) regions once: mutations whose site lies in
    # a proteolytically removed segment are skipped with a specific warning
    # instead of a generic "site not found" error.
    try:
        from .sequence import resolve_mature_regions, is_cleaved_site, to_mature_pos
        mature_regions = resolve_mature_regions(config, paths) or {}
    except Exception:  # pragma: no cover - defensive
        mature_regions = {}
        is_cleaved_site = lambda region, pos: False  # noqa: E731
        to_mature_pos = lambda region, pos: pos  # noqa: E731

    # Pre-compute a single consistent sequence->structure numbering offset per
    # chain from ALL parsed point mutations. Reference PDBs (config.local_pdb_id)
    # are author-numbered and usually offset from the UniProt sequence numbering
    # the mutations use; discovering the offset once (and reusing it) lets even
    # sites whose author number is absent still resolve, while identity is still
    # verified per site inside map_site_to_structure.
    parsed_muts = []  # (expected_aa3, seq_pos)
    for item in worklist:
        if item[1].upper() == "WT":
            continue
        pm = _parse_point_mutation(item[1])
        if pm is not None:
            # Cleaved-region sites are absent from the mature reference
            # structure; do not let them poison the offset discovery.
            if is_cleaved_site(mature_regions.get(item[0]), int(pm[1])):
                continue
            parsed_muts.append((_AA3[pm[0]], int(pm[1])))
    chain_offsets: dict = {}
    try:
        parsed_pdb = parse_pdb_residues(wt_out)
        for _uid, _chains in chain_map.items():
            for _ch in _chains:
                res = parsed_pdb.get(_ch)
                if res and parsed_muts:
                    chain_offsets[_ch] = find_chain_offset(res, parsed_muts)
    except Exception as e:  # pragma: no cover - defensive
        log.debug("graft: chain offset pre-computation skipped: %s", e)

    for item in worklist:
        uid, mut = item[0], item[1]
        if mut.upper() == "WT":
            continue
        mut_out = canonical_pdb_path(config, paths, uid, mut)
        if os.path.exists(mut_out):
            result["outputs"].append(mut_out)
            result["n_done"] += 1
            continue
        parsed = _parse_point_mutation(mut)
        if parsed is None:
            result["errors"].append(f"graft fallback: unparsable mutation '{mut}' ({uid})")
            continue
        orig, pos, new = parsed
        expected_aa3 = _AA3[orig]
        seq_pos = int(pos)
        if is_cleaved_site(mature_regions.get(uid), seq_pos):
            log.warning(
                "graft mutate %s: site %s%d lies in the proteolytically cleaved "
                "region of %s (mature chain %s); the residue is absent from the "
                "mature structure - skipping graft",
                mut, expected_aa3, seq_pos, uid, mature_regions.get(uid))
            continue
        uid_chains = chain_map.get(uid, ["A"])
        try:
            p = Protein(filename=wt_out)
            applied = 0
            graft_mature_pos = to_mature_pos(mature_regions.get(uid), seq_pos)
            for chain in uid_chains:
                # Resolve the sequence-numbered site to the ordinal-based
                # selector PTM-Psi expects, verifying the residue identity so we
                # never graft onto the wrong position. The site is
                # UniProt-numbered; mature-folded structures (and most PDBs of
                # matured proteins) use mature numbering, so the mature
                # position is tried first, then the UniProt one — every
                # candidate is name-verified. Skip (with a warning) any chain
                # where the site can't be located with a matching residue.
                mapped = map_site_to_structure_candidates(
                    expected_aa3, [graft_mature_pos, seq_pos], chain,
                    protein=p, pdb_path=wt_out,
                    offset=chain_offsets.get(chain),
                )
                if mapped is None:
                    log.warning(
                        "graft mutate %s: site %s%d not found on chain %s of %s "
                        "(residue identity/number could not be verified in "
                        "mature or UniProt numbering); skipping this chain",
                        mut, expected_aa3, seq_pos, chain, uid)
                    continue
                selector, _ordinal = mapped
                try:
                    p.mutate(selector, _AA3[new])
                    applied += 1
                except Exception as e:
                    log.warning("graft mutate %s chain %s (selector %s) failed: %s",
                                mut, chain, selector, e)
            if applied == 0:
                result["errors"].append(
                    f"graft fallback: mutation {mut} could not be applied to any chain of {uid} "
                    f"(site {expected_aa3}{seq_pos} not verifiable on any chain)")
                continue
            p.write_pdb(mut_out)
            result["outputs"].append(mut_out)
            result["n_done"] += 1
            log.info("Graft fallback: grafted %s -> %s", mut, os.path.basename(mut_out))
        except Exception as e:
            result["errors"].append(f"graft fallback: failed to graft {mut} ({uid}): {e}")

    return result
