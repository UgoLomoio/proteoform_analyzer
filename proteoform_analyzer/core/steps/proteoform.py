"""Step: proteoform generation (pairwise mutation + PTM combinations).

Generates structures where each single-point mutation is combined with each PTM
type, producing proteoforms that capture mutation-PTM interactions.  Uses ptmpsi
.mutate() then .modify() on the WT structure.

As of v3.3.1, both the mutation and PTM sites are mapped to the correct
structural residue via ``_resnum.map_site_to_structure`` (name-verified,
ordinal-based selector), fixing the same offset-numbering bug that v3.3.0 fixed
in the PTM step. Sites that cannot be verified are skipped with a warning rather
than being applied to the wrong residue.

PTM site resolution uses the SAME 3-level precedence as the PTM step
(``ptm._resolve_sites``): explicit ``ptm.pairs`` from the GUI builder, then the
legacy ``ptm.residues`` x ``ptm.ptm_types`` cross-product, then UniProt
auto-fetch. Previously this step only inspected the legacy lists, so it skipped
with "No PTM residues or types configured" even when PTM pairs (or auto-fetchable
PTMs) and mutations were both configured.
"""
from __future__ import annotations

import os
import sys
import logging

from ..pipeline import StepResult

log = logging.getLogger("proteoform_analyzer.proteoform")


def _vendored_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "_vendored")


def _import_ptmpsi():
    v = _vendored_path()
    if v not in sys.path:
        sys.path.insert(0, v)
    from ptmpsi.protein import Protein
    return Protein


_AA_MAP = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


def _parse_mutation(mut: str):
    """'D58G' -> ('D', 58, 'G'); returns None for WT/malformed."""
    if not mut or mut.upper() == "WT" or len(mut) < 3:
        return None
    orig = mut[0].upper()
    new = mut[-1].upper()
    pos = mut[1:-1]
    if orig not in _AA_MAP or new not in _AA_MAP or not pos.isdigit():
        return None
    return orig, int(pos), new


def _parse_ptm_residue(residue: str):
    """'CYS10' -> ('CYS', 10); returns None if unparseable."""
    aa3 = residue[:3].upper()
    num_part = residue[3:]
    if not num_part.isdigit():
        return None
    return aa3, int(num_part)


def generate_proteoforms(config, paths: dict) -> StepResult:
    """Generate pairwise mutation+PTM combination structures."""
    from .structure import _get_chain_map
    from ._resnum import (map_site_to_structure, map_site_to_structure_candidates,
                          find_chain_offset, parse_pdb_residues)
    from .ptm import _resolve_sites, _chains_for_uid

    if config.proteoform_mode == "single":
        return StepResult("proteoform", "skipped",
                          "proteoform_mode='single' — no combinatorial proteoforms")

    Protein = _import_ptmpsi()

    # Resolve PTM sites with the same precedence as the PTM step:
    # explicit pairs -> legacy residues x ptm_types -> UniProt auto-fetch.
    sites = _resolve_sites(config, paths)
    if not sites:
        return StepResult(
            "proteoform", "skipped",
            "No PTM sites configured and no observed, ptmpsi-modelable PTMs "
            "found on UniProt for the configured protein(s); proteoform "
            "generation skipped.")

    out_dir = os.path.join(paths.get("proteoforms", os.path.join(paths["results"], "proteoforms")))
    os.makedirs(out_dir, exist_ok=True)

    # Get WT structure (resolves through boltz-experiments first).
    from ._structure_source import resolve_structure_pdb
    pdb_dir = paths["pdbs_monomer"] if config.is_monomer else paths["pdbs"]
    wt_pdb = None
    for f in sorted(os.listdir(pdb_dir)):
        if f.endswith(".pdb") and f.lower().startswith("wt"):
            wt_pdb = (resolve_structure_pdb(paths, f[:-4], "proteoform")
                      or os.path.join(pdb_dir, f))
            break
    if wt_pdb is None:
        return StepResult("proteoform", "skipped", "No WT PDB found")

    chain_map = _get_chain_map(config)
    mutation_lists = getattr(config, "_mutation_lists", None) or []

    # Mature (post-cleavage) regions: mutations in proteolytically removed
    # segments are absent from the mature structure.
    try:
        from .sequence import resolve_mature_regions, is_cleaved_site, to_mature_pos
        mature_regions = resolve_mature_regions(config, paths) or {}
    except Exception:  # pragma: no cover - defensive
        mature_regions = {}
        is_cleaved_site = lambda region, pos: False  # noqa: E731
        to_mature_pos = lambda region, pos: pos  # noqa: E731

    # Pre-compute per-chain numbering offsets from ALL parsed mutations and PTM
    # sites, so even sites whose author number is absent can resolve. Reference
    # PDBs are author-numbered and usually offset from the UniProt sequence
    # numbering the mutations/PTMs use.
    parsed_muts = []
    for ml_idx, ml in enumerate(mutation_lists):
        ml_uid = (config.uniprot_ids[ml_idx]
                  if ml_idx < len(config.uniprot_ids) else None)
        for m in ml:
            pm = _parse_mutation(m)
            if pm is not None:
                # Cleaved-region sites must not poison the offset discovery.
                if ml_uid and is_cleaved_site(mature_regions.get(ml_uid), pm[1]):
                    continue
                parsed_muts.append((_AA_MAP[pm[0]], pm[1]))
    parsed_ptm_sites = []
    for r, _ptm_type, _site_uid in sites:
        pr = _parse_ptm_residue(r)
        if pr is not None:
            parsed_ptm_sites.append(pr)
    all_sites = parsed_muts + parsed_ptm_sites

    chain_offsets: dict = {}
    try:
        parsed_pdb = parse_pdb_residues(wt_pdb)
        for _uid, _chains in chain_map.items():
            for _ch in _chains:
                res = parsed_pdb.get(_ch)
                if res and all_sites:
                    chain_offsets[_ch] = find_chain_offset(res, all_sites)
    except Exception as e:
        log.debug("proteoform: chain offset pre-computation skipped: %s", e)

    outputs = []
    n_proteoforms = 0
    cap = config.proteoform_cap

    for idx, uid in enumerate(config.uniprot_ids):
        uid_chains = chain_map.get(uid, ["A"])
        muts = mutation_lists[idx] if idx < len(mutation_lists) else []
        muts = [m for m in muts if m.upper() != "WT"]

        for mut in muts:
            if n_proteoforms >= cap:
                break
            parsed_mut = _parse_mutation(mut)
            if parsed_mut is None:
                log.warning("proteoform: unparsable mutation '%s'; skipping", mut)
                continue
            orig_aa, seq_pos, new_aa = parsed_mut
            if is_cleaved_site(mature_regions.get(uid), seq_pos):
                reg = mature_regions[uid]
                log.warning(
                    "proteoform: mutation %s (%s) lies in the proteolytically "
                    "cleaved region (mature chain %d-%d); skipped",
                    mut, uid, reg[0], reg[1])
                continue
            expected_mut_aa3 = _AA_MAP[orig_aa]
            new_aa3 = _AA_MAP[new_aa]

            for residue, ptm_type, site_uid in sites:
                if n_proteoforms >= cap:
                    break
                parsed_ptm = _parse_ptm_residue(residue)
                if parsed_ptm is None:
                    log.warning("proteoform: could not parse PTM residue spec "
                                "'%s' (expected e.g. 'CYS10'); skipping", residue)
                    continue
                ptm_aa3, ptm_seq_pos = parsed_ptm
                # Chains this PTM site applies to: the site's own subunit when
                # known (explicit pairs carry a uniprot_id), else the configured
                # PTM chains.
                ptm_chains = _chains_for_uid(config, site_uid)

                name = f"Proteoform_{uid}_{mut}_{ptm_type}_{residue}"
                out_path = os.path.join(out_dir, f"{name}.pdb")
                if os.path.exists(out_path):
                    outputs.append(out_path)
                    n_proteoforms += 1
                    continue
                try:
                    p = Protein(filename=wt_pdb)
                    mut_applied = False
                    # Apply mutation to the subunit's chains (name-verified
                    # ordinal selector via map_site_to_structure). The site is
                    # UniProt-numbered; mature-folded structures (and most PDBs
                    # of matured proteins) use mature numbering, so the mature
                    # position is tried first, then the UniProt one — every
                    # candidate is name-verified before use.
                    mut_mature_pos = to_mature_pos(mature_regions.get(uid), seq_pos)
                    for chain in uid_chains:
                        mapped = map_site_to_structure_candidates(
                            expected_mut_aa3, [mut_mature_pos, seq_pos], chain,
                            protein=p, pdb_path=wt_pdb,
                            offset=chain_offsets.get(chain),
                        )
                        if mapped is None:
                            log.warning(
                                "proteoform mutate %s: site %s%d not found on "
                                "chain %s of %s (residue identity/number could "
                                "not be verified in mature or UniProt "
                                "numbering); skipping this chain",
                                mut, expected_mut_aa3, seq_pos, chain, uid)
                            continue
                        selector, _ordinal = mapped
                        try:
                            p.mutate(selector, new_aa3)
                            mut_applied = True
                        except Exception as e:
                            log.warning("proteoform mutate %s chain %s "
                                        "(selector %s) failed: %s",
                                        mut, chain, selector, e)
                    if not mut_applied:
                        log.warning("proteoform %s: mutation %s not verifiable "
                                    "on any chain; skipping proteoform",
                                    name, mut)
                        continue
                    # Apply PTM to the site's chains (name-verified ordinal
                    # selector via map_site_to_structure; mature position tried
                    # before the UniProt one, as for the mutation above).
                    ptm_applied = False
                    ptm_mature_pos = to_mature_pos(mature_regions.get(site_uid),
                                                   ptm_seq_pos)
                    for chain in ptm_chains:
                        mapped = map_site_to_structure_candidates(
                            ptm_aa3, [ptm_mature_pos, ptm_seq_pos], chain,
                            protein=p, pdb_path=wt_pdb,
                            offset=chain_offsets.get(chain),
                        )
                        if mapped is None:
                            log.warning(
                                "proteoform PTM %s: site %s%d not found on "
                                "chain %s of %s (residue identity/number could "
                                "not be verified in mature or UniProt "
                                "numbering); skipping this chain",
                                ptm_type, ptm_aa3, ptm_seq_pos, chain, name)
                            continue
                        selector, _ordinal = mapped
                        try:
                            p.modify(selector, ptm_type)
                            ptm_applied = True
                        except Exception as e:
                            log.warning("proteoform PTM %s on %s chain %s "
                                        "(selector %s) failed: %s",
                                        ptm_type, residue, chain, selector, e)
                    if not ptm_applied:
                        log.warning("proteoform %s: PTM %s @ %s not verifiable "
                                    "on any chain; skipping proteoform",
                                    name, ptm_type, residue)
                        continue
                    p.write_pdb(out_path)
                    outputs.append(out_path)
                    n_proteoforms += 1
                    log.info("Generated proteoform %s", name)
                except Exception as e:
                    log.error("Failed to generate proteoform %s: %s", name, e)

    n_muts = sum(len([m for m in ml if m.upper() != "WT"]) for ml in mutation_lists)
    return StepResult(
        "proteoform", "ok" if n_proteoforms else "skipped",
        f"Generated {n_proteoforms} proteoform structures "
        f"(pairwise: {n_muts} muts x {len(sites)} PTM sites, cap={cap})",
        outputs=outputs,
    )
