"""Name-verified residue-number mapping for PTM-Psi grafting / PTM steps.

Why this exists
---------------
Mutation and PTM sites in this pipeline are numbered against the **UniProt
canonical sequence** (the sequence step validates ``sequence[pos-1] == wt_aa``).
But the reference PDB structures used by the graft / PTM steps
(``config.local_pdb_id``, e.g. ``1A3N`` for hemoglobin) use **author / PDB
residue numbering**, which is almost always offset from the sequence numbering
(initiator Met, construct tags, disordered N-termini, or structures that only
resolve a sub-range such as p53 ``1PES`` spanning author residues 325-356).

The vendored PTM-Psi ``get_residue`` (``_vendored/ptmpsi/protein/tools.py``)
resolves a selector like ``"A:LEU87"`` by **ordinal position**
(``chain.residues[87-1]``) and then *validates the name* -- so when author-87 is
not the 87th residue in the chain, the name check fails with::

    Residue 'LEU87' could not be found

which is exactly the two bugs reported by the user (``L87R`` graft on hemoglobin,
``glutathionylation`` on ``CYS10``).

What this module does
---------------------
``map_site_to_structure`` takes the *sequence-numbered* site
(``expected_aa3`` + ``seq_pos``) and returns an **ordinal-based** selector that
PTM-Psi's ``get_residue`` will accept, i.e. ``"{chain}:{AA3}{ordinal}"`` where
``ordinal`` is the 1-based index of the residue inside the chain. The mapping is
resolved with three strategies, and **the amino-acid identity is verified in
every strategy** so we never silently mutate/modify the wrong residue:

  1. **Author-number match** -- residue whose PDB author number == ``seq_pos``
     *and* whose name == ``expected_aa3``. (Handles the offset case.)
  2. **Ordinal match** -- ``residues[seq_pos-1]`` when its name == ``expected_aa3``.
     (Handles structures that happen to be 1-based / offset-free.)
  3. **Single consistent per-chain offset** -- a unique integer ``k`` such that
     the residue at author number ``seq_pos + k`` (equivalently ordinal) has the
     expected name; the offset is discovered once and can be reused by callers.

If none of the strategies produces an identity-verified hit (site genuinely
absent, ambiguous, or a name mismatch), the function returns ``None`` and the
caller is expected to **skip that site with a warning** -- never guess.

This module is import-light (pure stdlib + regex) and never raises for
operational problems; it only parses text and compares strings.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, List, Tuple, Dict

log = logging.getLogger("proteoform_analyzer.resnum")

# Residue names PTM-Psi treats as water / non-polymer and drops during parsing;
# we mirror that so ordinals line up with the parsed Protein object.
_WATER = {"HOH", "WAT"}


# --------------------------------------------------------------------------- #
# PDB parsing (author number + ordinal), mirroring ptmpsi.io.digestpdb
# --------------------------------------------------------------------------- #
def parse_pdb_residues(pdb_path: str) -> Dict[str, List[Tuple[int, str, int]]]:
    """Parse a PDB file into ``{chain_id: [(author_num, resname3, ordinal), ...]}``.

    ``ordinal`` is the 1-based position of the residue within its chain, exactly
    matching how PTM-Psi indexes ``chain.residues`` after parsing the same file.
    HETATM and water are skipped (PTM-Psi drops them by default), and residues
    are de-duplicated on ``(chain, author_num, insertion_code)`` so that
    multi-atom records collapse to one residue entry.

    Never raises for malformed lines; unreadable files yield ``{}``.
    """
    chains: Dict[str, List[Tuple[int, str, int]]] = {}
    ordinals: Dict[str, int] = {}
    seen: Dict[str, set] = {}
    try:
        with open(pdb_path, "r") as fh:
            lines = fh.readlines()
    except Exception as e:  # pragma: no cover - defensive
        log.debug("resnum: could not read %s: %s", pdb_path, e)
        return {}

    for line in lines:
        rec = line[:6]
        if rec.startswith("ENDMDL"):
            # Only take the first model of an NMR / multi-model ensemble, which
            # is what ptmpsi effectively does when it stops at END/ENDMDL.
            break
        if not (rec.startswith("ATOM") or rec.startswith("HETATM")):
            continue
        # HETATM are dropped by ptmpsi's default digestpdb(delhet=True).
        if rec.startswith("HETATM"):
            continue
        try:
            resname = line[17:20].strip().upper()
            chain_id = line[21:22]
            author_num = int(line[22:26])
            icode = line[26:27]
        except (ValueError, IndexError):
            continue
        if resname in _WATER:
            continue
        key = f"{chain_id}|{author_num}|{icode}"
        chain_seen = seen.setdefault(chain_id, set())
        if key in chain_seen:
            continue
        chain_seen.add(key)
        ordn = ordinals.get(chain_id, 0) + 1
        ordinals[chain_id] = ordn
        chains.setdefault(chain_id, []).append((author_num, resname, ordn))
    return chains


# --------------------------------------------------------------------------- #
# Residue index abstraction: works from a PDB path OR a parsed Protein object
# --------------------------------------------------------------------------- #
def _residues_from_protein(protein, chain_id: str) -> Optional[List[Tuple[Optional[int], str, int]]]:
    """Return ``[(author_num_or_None, resname3, ordinal), ...]`` for one chain of a
    parsed PTM-Psi ``Protein`` object.

    PTM-Psi overwrites ``residue.resid`` with the ordinal during ``update()`` and
    does **not** retain the original author number, so ``author_num`` is ``None``
    here (strategy 1 is unavailable and we fall back to ordinal / offset, which
    only need names + positions). Returns ``None`` if the chain is absent.
    """
    chains = getattr(protein, "chains", None)
    if not chains:
        return None
    for ch in chains:
        if getattr(ch, "name", None) == chain_id:
            residues = getattr(ch, "residues", None) or []
            out: List[Tuple[Optional[int], str, int]] = []
            for i, r in enumerate(residues):
                name = str(getattr(r, "name", "")).strip().upper()
                out.append((None, name, i + 1))
            return out
    return None


def _get_chain_residues(
    chain_id: str,
    protein=None,
    pdb_path: Optional[str] = None,
) -> Optional[List[Tuple[Optional[int], str, int]]]:
    """Resolve the per-chain residue list, preferring the source PDB (which
    preserves author numbers) and falling back to the parsed Protein object."""
    if pdb_path and os.path.exists(pdb_path):
        parsed = parse_pdb_residues(pdb_path)
        if chain_id in parsed:
            return list(parsed[chain_id])
    if protein is not None:
        return _residues_from_protein(protein, chain_id)
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def find_chain_offset(
    residues: List[Tuple[Optional[int], str, int]],
    site_pairs: List[Tuple[str, int]],
) -> Optional[int]:
    """Find a single consistent offset ``k`` (author = seq_pos + k) that makes the
    largest number of ``(expected_aa3, seq_pos)`` sites resolve by *name*.

    Returns the unique best offset, or ``None`` if no offset validates any site
    or the best offset is ambiguous (tied with another offset on the max count).
    Used so the graft / PTM steps discover the numbering shift **once per chain**
    and reuse it for every site instead of re-searching per site.
    """
    if not residues or not site_pairs:
        return None
    # author-number -> name (only meaningful when author numbers are present)
    by_author: Dict[int, str] = {}
    have_author = any(a is not None for a, _, _ in residues)
    if have_author:
        for author, name, _ordn in residues:
            if author is not None:
                by_author[author] = name
    # ordinal -> name (always available)
    by_ordinal: Dict[int, str] = {ordn: name for _a, name, ordn in residues}

    counts: Dict[int, int] = {}
    for aa3, seq_pos in site_pairs:
        aa3 = aa3.upper()
        lookup = by_author if have_author else by_ordinal
        for pos, name in lookup.items():
            if name == aa3:
                k = pos - seq_pos
                counts[k] = counts.get(k, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    winners = [k for k, c in counts.items() if c == best]
    if len(winners) != 1:
        return None
    return winners[0]


def map_site_to_structure(
    expected_aa3: str,
    seq_pos: int,
    chain_id: str,
    protein=None,
    pdb_path: Optional[str] = None,
    offset: Optional[int] = None,
) -> Optional[Tuple[str, int]]:
    """Map a sequence-numbered site to an **ordinal-based** PTM-Psi selector.

    Parameters
    ----------
    expected_aa3 : 3-letter code of the wild-type residue expected at the site
        (e.g. ``"LEU"`` for ``L87R``, ``"CYS"`` for a glutathionylation on Cys).
        Identity is verified against this in every strategy.
    seq_pos : 1-based residue number in the *sequence* numbering (UniProt).
    chain_id : chain to search (e.g. ``"A"``).
    protein : optional parsed PTM-Psi ``Protein`` (provides ordinal + names).
    pdb_path : optional path to the source PDB (provides author numbers too;
        preferred, because it enables the author-number strategy).
    offset : optional pre-computed per-chain offset from :func:`find_chain_offset`;
        when given it is tried *first* (author = seq_pos + offset).

    Returns
    -------
    ``(selector, ordinal)`` where ``selector == f"{chain_id}:{expected_aa3}{ordinal}"``
    and ``ordinal`` is the 1-based chain index PTM-Psi will resolve, or ``None``
    if the site cannot be located with a verified amino-acid identity.
    """
    expected_aa3 = (expected_aa3 or "").upper()
    if not expected_aa3 or seq_pos is None or seq_pos < 1:
        return None
    residues = _get_chain_residues(chain_id, protein=protein, pdb_path=pdb_path)
    if not residues:
        return None

    by_author: Dict[int, Tuple[str, int]] = {}
    for author, name, ordn in residues:
        if author is not None:
            by_author[author] = (name, ordn)
    by_ordinal: Dict[int, Tuple[str, int]] = {ordn: (name, ordn) for _a, name, ordn in residues}

    def _verified(ordn: int, name: str) -> Optional[Tuple[str, int]]:
        if name == expected_aa3:
            return f"{chain_id}:{expected_aa3}{ordn}", ordn
        return None

    # Strategy 0: caller-provided offset (author = seq_pos + offset). Only when
    # author numbers exist; otherwise treat offset against ordinals.
    if offset is not None:
        target = seq_pos + offset
        if by_author:
            hit = by_author.get(target)
            if hit is not None:
                res = _verified(hit[1], hit[0])
                if res:
                    return res
        else:
            hit = by_ordinal.get(target)
            if hit is not None:
                res = _verified(hit[1], hit[0])
                if res:
                    return res

    # Strategy 1: author-number match (name-verified).
    if by_author:
        hit = by_author.get(seq_pos)
        if hit is not None:
            res = _verified(hit[1], hit[0])
            if res:
                return res

    # Strategy 2: ordinal match (name-verified).
    hit = by_ordinal.get(seq_pos)
    if hit is not None:
        res = _verified(hit[1], hit[0])
        if res:
            return res

    # NO lone-match fallback. Deliberately: if a single site does not resolve by
    # its author number, its ordinal, or a *corroborated* per-chain offset
    # (Strategy 0, discovered from the full site list via find_chain_offset),
    # we refuse to guess. Grabbing "the only CYS in the chain" from a bare
    # position/name mismatch is exactly how the wrong residue gets modified, so
    # unresolved sites return None and the caller skips them with a warning.
    return None


def map_site_to_structure_candidates(
    expected_aa3: str,
    seq_positions,
    chain_id: str,
    protein=None,
    pdb_path: Optional[str] = None,
    offset: Optional[int] = None,
) -> Optional[Tuple[str, int]]:
    """Try several candidate sequence positions and return the first verified hit.

    Mutation/PTM sites are UniProt (precursor) numbered, but structures folded
    from the **mature** sequence (Boltz v3.6+) — and most reference PDBs of
    proteolytically matured proteins (e.g. TTR) — use mature numbering. Callers
    pass ``[mature_pos, uniprot_pos]`` (from ``to_mature_pos``); each candidate
    goes through :func:`map_site_to_structure`, whose every strategy verifies
    the amino-acid identity, so a candidate only wins when the residue in the
    structure actually matches ``expected_aa3``. ``None``/duplicate positions
    are skipped. Returns ``None`` when no candidate verifies (never guesses).
    """
    seen = set()
    for pos in seq_positions or []:
        if pos is None or pos in seen:
            continue
        seen.add(pos)
        hit = map_site_to_structure(
            expected_aa3, pos, chain_id,
            protein=protein, pdb_path=pdb_path, offset=offset,
        )
        if hit is not None:
            return hit
    return None
