"""Helpers to build viewable receptor+ligand (and receptor+pose) *complex* PDBs.

Motivation
----------
The docking step historically produced only:
  * Vina  -> a ligand-only docked pose in **PDBQT** (``*_vina_out.pdbqt``), and
  * Boltz -> a receptor+ligand **mmCIF** that was never converted/registered,
so the 3D viewer (which lists ``pdbs/…`` receptor-only files) always showed the
apo receptor: protein chains only, no ligand. This module writes a single
``*_complex.pdb`` that contains the receptor **and** the docked ligand (as
``HETATM`` records on a dedicated chain/resname), so the existing
molecule-type-aware viewer renders the ligand as licorice automatically.

All functions are pure/offline and unit-testable (no network, no GPU).
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger("proteoform_analyzer.complex_pdb")

# Chain id reserved for a docked ligand appended to a receptor, and a fallback
# 3-letter residue name when the pose does not carry one.
LIG_CHAIN = "Z"
LIG_RESNAME = "LIG"


# ---------------------------------------------------------------------------
# PDBQT pose -> PDB HETATM ligand block
# ---------------------------------------------------------------------------

def _pdbqt_first_model_atoms(pdbqt_path: str) -> list[str]:
    """Return the ATOM/HETATM lines of the FIRST model in a Vina output PDBQT.

    Vina writes ``MODEL 1 … ENDMDL`` blocks (best pose first). We keep only the
    first model's atom lines. PDBQT atom lines are PDB-compatible in columns
    1-66; the trailing charge/atom-type columns (67+) are simply dropped by
    truncating to 66 chars when we re-emit as PDB.
    """
    if not pdbqt_path or not os.path.exists(pdbqt_path):
        return []
    atoms: list[str] = []
    in_first_model = True
    seen_model = False
    with open(pdbqt_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "MODEL":
                if seen_model:
                    # second model starts -> stop
                    break
                seen_model = True
                in_first_model = True
                continue
            if rec == "ENDMDL":
                if seen_model:
                    break
                continue
            if rec in ("ATOM", "HETATM") and in_first_model:
                atoms.append(line.rstrip("\n"))
    return atoms


def pose_lines_to_hetatm(atom_lines: list[str],
                         chain: str = LIG_CHAIN,
                         resname: str = LIG_RESNAME,
                         resseq: int = 1,
                         start_serial: int = 9000) -> list[str]:
    """Rewrite docked-pose atom lines as clean PDB ``HETATM`` records.

    Forces record type = HETATM, a dedicated ``chain``/``resname``/``resseq`` so
    the ligand is unambiguous in the viewer, and renumbers serials from
    ``start_serial`` to avoid collisions with the receptor. Element (cols 77-78)
    is preserved when present, otherwise guessed from the atom name.
    """
    out: list[str] = []
    serial = start_serial
    for ln in atom_lines:
        # pad to at least 78 columns so slicing is safe
        s = ln.ljust(80)
        atom_name = s[12:16]
        # element: prefer existing cols 77-78, else derive from the atom name
        element = s[76:78].strip()
        if not element:
            nm = atom_name.strip()
            # strip leading digits, take first 1-2 alpha chars
            alpha = "".join(c for c in nm if c.isalpha())
            element = (alpha[:1] or "C").upper()
        x = s[30:38]
        y = s[38:46]
        z = s[46:54]
        occ = s[54:60] if s[54:60].strip() else "  1.00"
        bfac = s[60:66] if s[60:66].strip() else "  0.00"
        rec = (
            f"HETATM{serial:>5} {atom_name}{'':1}{resname:>3} {chain:1}"
            f"{resseq:>4}{'':1}   {x:>8}{y:>8}{z:>8}{occ:>6}{bfac:>6}"
            f"{'':10}{element:>2}"
        )
        out.append(rec[:80])
        serial += 1
    return out


# ---------------------------------------------------------------------------
# Receptor PDB reading (protein-only)
# ---------------------------------------------------------------------------

def _receptor_lines(receptor_pdb: str) -> list[str]:
    """Return receptor ATOM/HETATM/TER lines (drop END/CONECT/MODEL wrappers)."""
    keep: list[str] = []
    with open(receptor_pdb) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM", "TER"):
                keep.append(line.rstrip("\n"))
    return keep


# ---------------------------------------------------------------------------
# Public: build a receptor+ligand complex PDB
# ---------------------------------------------------------------------------

def write_vina_complex(receptor_pdb: str, vina_out_pdbqt: str,
                       out_pdb: str,
                       ligand_resname: str = LIG_RESNAME) -> str | None:
    """Merge a receptor PDB with the best Vina pose into a complex PDB.

    Returns ``out_pdb`` on success (file contains receptor ATOM records + the
    docked ligand as HETATM on chain ``Z``), or ``None`` if inputs are missing
    or the pose could not be parsed.
    """
    if not (receptor_pdb and os.path.exists(receptor_pdb)):
        log.warning("Vina complex: receptor PDB missing (%s)", receptor_pdb)
        return None
    pose = _pdbqt_first_model_atoms(vina_out_pdbqt)
    if not pose:
        log.warning("Vina complex: no pose atoms parsed from %s", vina_out_pdbqt)
        return None
    rec = _receptor_lines(receptor_pdb)
    if not rec:
        log.warning("Vina complex: receptor has no atom records (%s)", receptor_pdb)
        return None
    het = pose_lines_to_hetatm(pose, resname=(ligand_resname or LIG_RESNAME)[:3])
    os.makedirs(os.path.dirname(os.path.abspath(out_pdb)), exist_ok=True)
    with open(out_pdb, "w") as fh:
        fh.write("REMARK   Proteoform Analyzer docked complex (receptor + Vina pose)\n")
        for ln in rec:
            # drop any trailing END the receptor might carry (handled by _receptor_lines)
            fh.write(ln + "\n")
        # ensure a TER between receptor and ligand
        if not rec[-1].startswith("TER"):
            fh.write("TER\n")
        for ln in het:
            fh.write(ln + "\n")
        fh.write("END\n")
    if os.path.getsize(out_pdb) > 0:
        log.info("Wrote Vina complex PDB: %s", out_pdb)
        return out_pdb
    return None


def write_boltz_complex(cif_path: str, out_pdb: str) -> str | None:
    """Convert a Boltz-2 co-folded receptor+ligand mmCIF to a complex PDB.

    The Boltz output already contains both the protein and the ligand, so a
    direct CIF->PDB conversion yields a viewable complex. Returns ``out_pdb`` or
    ``None`` on failure.
    """
    if not (cif_path and os.path.exists(cif_path)):
        return None
    try:
        from ._pdb_utils import cif_to_pdb
        cif_to_pdb(cif_path, out_pdb)
        if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0:
            log.info("Wrote Boltz complex PDB: %s", out_pdb)
            return out_pdb
    except Exception as e:  # pragma: no cover - depends on gemmi/biopython
        log.warning("Boltz complex CIF->PDB failed (%s): %s", cif_path, e)
    return None
