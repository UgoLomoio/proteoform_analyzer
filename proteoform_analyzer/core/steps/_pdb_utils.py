"""Shared structure/PDB utilities.

Houses helpers used by multiple steps:
  - ``fetch_reference_pdb``: download a reference/co-crystal PDB (used ONLY for
    known-binding-site alignment and optional validation, NOT as a structure
    source). This is the internal PDB-fetch retained after removing the
    ``local_pdb`` structure source.
  - ``cif_to_pdb``: convert a Boltz-2 (mmCIF) prediction to PDB so the rest of
    the pipeline (PTM-Psi, TM-align, PCN, docking, pocket) keeps working on the
    canonical ``.pdb`` filenames.
  - sequence helpers for applying single-point mutations.
"""
from __future__ import annotations

import os
import shutil
import logging

log = logging.getLogger("proteoform_analyzer.pdb_utils")


# ---------------------------------------------------------------------------
# Three-letter -> one-letter residue code (Biopython-version robust)
# ---------------------------------------------------------------------------
# Bio.PDB.Polypeptide.three_to_one was removed in newer Biopython. Prefer the
# data table, fall back to the older function, then to a builtin map.
try:  # Biopython >= 1.80
    from Bio.Data.PDBData import protein_letters_3to1 as _AA3TO1
except Exception:  # pragma: no cover
    try:
        from Bio.Data.SCOPData import protein_letters_3to1 as _AA3TO1  # type: ignore
    except Exception:
        _AA3TO1 = {}

_AA3TO1_FALLBACK = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def three_to_one(resname: str, default: str = "X") -> str:
    """Convert a 3-letter residue name to its 1-letter code.

    Robust across Biopython versions; unknown residues return ``default``.
    """
    key = (resname or "").strip().upper()
    aa = _AA3TO1.get(key) if _AA3TO1 else None
    if aa:
        return aa
    return _AA3TO1_FALLBACK.get(key, default)


# ---------------------------------------------------------------------------
# Reference PDB fetch (internal; not a user-facing structure source)
# ---------------------------------------------------------------------------

def fetch_reference_pdb(pdb_id: str, out_dir: str) -> str:
    """Download a reference PDB directly from RCSB (cached).

    Used for known-binding-site alignment (e.g. 5E83 voxelotor, 4DST tafamidis)
    and optional validation. Returns a clean ``.pdb`` path.
    Uses direct HTTP download instead of Biopython PDBList to avoid mirror issues.
    """
    import httpx
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, f"{pdb_id.lower()}.pdb")
    if os.path.exists(target):
        log.info("Reference PDB %s cached at %s", pdb_id, target)
        return target
    
    # Try direct download from RCSB
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    log.info("Downloading PDB structure '%s' from %s", pdb_id.upper(), url)
    
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=30.0) as response:
            response.raise_for_status()
            with open(target, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        
        # Verify the file is valid (has ATOM records)
        with open(target, "r") as f:
            content = f.read()
            if "ATOM" not in content and "HETATM" not in content:
                os.remove(target)
                raise ValueError("Downloaded file contains no ATOM/HETATM records")
        
        log.info("Downloaded reference PDB %s -> %s", pdb_id.upper(), target)
        return target
        
    except Exception as e:
        log.warning("Direct download failed for %s: %s", pdb_id.upper(), e)
        raise FileNotFoundError(f"Could not download reference PDB {pdb_id.upper()}: {e}")

# ---------------------------------------------------------------------------
# CIF -> PDB conversion (Boltz-2 output normalization)
# ---------------------------------------------------------------------------

def cif_to_pdb(cif_path: str, pdb_path: str) -> str:
    """Convert an mmCIF structure to PDB.

    Prefers gemmi (robust, preserves chain ids); falls back to Biopython
    ``MMCIFParser`` + ``PDBIO``. Returns ``pdb_path`` on success, else raises.
    """
    os.makedirs(os.path.dirname(os.path.abspath(pdb_path)), exist_ok=True)
    # Try gemmi first (if available)
    try:
        import gemmi
        st = gemmi.read_structure(cif_path)
        st.setup_entities()
        st.write_pdb(pdb_path)
        if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0:
            return pdb_path
    except Exception as e:  # pragma: no cover - gemmi optional
        log.debug("gemmi CIF->PDB unavailable/failed (%s); using biopython", e)

    # Biopython fallback
    from Bio.PDB import MMCIFParser, PDBIO
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("boltz", cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)
    if not (os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0):
        raise RuntimeError(f"CIF->PDB conversion produced empty file: {pdb_path}")
    return pdb_path


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def apply_mutation(seq: str, mut: str) -> str:
    """Apply a single-point mutation 'XnnY' (1-based) to a sequence.

    'WT' returns the sequence unchanged. Out-of-range positions are ignored
    (returns the original sequence) so a bad mutation never crashes folding.
    """
    if not mut or mut.upper() == "WT":
        return seq
    try:
        pos = int(mut[1:-1])
        new = mut[-1]
    except (ValueError, IndexError):
        log.warning("Could not parse mutation '%s'; leaving sequence unchanged", mut)
        return seq
    if pos < 1 or pos > len(seq):
        log.warning("Mutation %s out of range for sequence length %d; unchanged", mut, len(seq))
        return seq
    return seq[:pos - 1] + new + seq[pos:]
