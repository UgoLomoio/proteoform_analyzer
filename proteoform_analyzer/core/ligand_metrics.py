"""Druggability metrics + target+ligand complex builder for designed ligands.

Used by the ligand-design step (DiffSBDD) so the 3D Structure Viewer can show
target+drug complexes and the structure-summary card can report druggability
metrics.

Metrics follow the DiffSBDD analysis set (``_vendored/diffsbdd/analysis/
metrics.py::MoleculeProperties``): QED, synthetic accessibility (SA), logP and
Lipinski rules, computed with RDKit. SA uses the Ertl & Schuffenhauer sascorer
— RDKit's bundled copy (``rdkit.Contrib.SA_Score``, ships the fpscores data) is
tried first, then the vendored DiffSBDD copy. We report the raw SA score
(1 = easy to synthesize, 10 = hard), not the 0–1 pocket2mol rescaling.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re

log = logging.getLogger("proteoform_analyzer.ligand_metrics")

_sascorer = None
_sascorer_tried = False


def _get_sascorer():
    """Return a sascorer module (with ``calculateScore``), or None."""
    global _sascorer, _sascorer_tried
    if _sascorer_tried:
        return _sascorer
    _sascorer_tried = True
    try:
        from rdkit.Contrib.SA_Score import sascorer as s
        _sascorer = s
        return _sascorer
    except Exception:
        pass
    try:
        vendored = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "_vendored",
            "diffsbdd", "analysis", "SA_Score", "sascorer.py"))
        if os.path.exists(vendored):
            spec = importlib.util.spec_from_file_location("pa_sascorer", vendored)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _sascorer = mod
    except Exception as e:
        log.warning("sascorer unavailable (SA score will be omitted): %s", e)
    return _sascorer


def _safe(fn, mol):
    try:
        return fn(mol)
    except Exception:
        return None


def compute_sdf_metrics(sdf_path: str):
    """Druggability metrics for every molecule in an SDF.

    Returns a ``pandas.DataFrame`` with one row per molecule and columns:
    ``index``, ``ligand``, ``smiles``, ``QED``, ``SA``, ``logP``, ``MW``,
    ``HBD``, ``HBA``, ``rotatable_bonds``, ``TPSA``, ``lipinski_violations``.
    Metrics that fail (e.g. unsanitizable generated molecules) are None.
    """
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED
    from rdkit.Chem import rdMolDescriptors

    sa_mod = _get_sascorer()
    rows = []
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                catchErrors=True)
        except Exception:
            pass
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"lig_{i}"
        row = {"index": i, "ligand": name}
        # Metrics are defined on heavy-atom structures: explicit Hs inflate the
        # sascorer non-fragment penalty (e.g. caffeine 2.3 -> 6.2).
        try:
            mol = Chem.RemoveHs(mol)
        except Exception:
            pass
        row["smiles"] = _safe(Chem.MolToSmiles, mol)
        row["QED"] = _safe(QED.qed, mol)
        row["SA"] = _safe(sa_mod.calculateScore, mol) if sa_mod else None
        row["logP"] = _safe(Crippen.MolLogP, mol)
        row["MW"] = _safe(Descriptors.ExactMolWt, mol)
        row["HBD"] = _safe(Lipinski.NumHDonors, mol)
        row["HBA"] = _safe(Lipinski.NumHAcceptors, mol)
        row["rotatable_bonds"] = _safe(rdMolDescriptors.CalcNumRotatableBonds, mol)
        row["TPSA"] = _safe(rdMolDescriptors.CalcTPSA, mol)
        # Classic Lipinski rule-of-5 violations (0-4; lower = more drug-like)
        try:
            v = 0
            v += 0 if (row["MW"] is not None and row["MW"] <= 500) else 1
            v += 0 if (row["HBD"] is not None and row["HBD"] <= 5) else 1
            v += 0 if (row["HBA"] is not None and row["HBA"] <= 10) else 1
            v += 0 if (row["logP"] is not None and row["logP"] <= 5) else 1
            row["lipinski_violations"] = v
        except Exception:
            row["lipinski_violations"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def _ligand_pdb_lines(mol, chain: str = "L", resname: str = "LIG",
                      resnum: int = 1) -> list[str]:
    """Render an RDKit molecule conformer as HETATM records on one chain."""
    from rdkit import Chem

    out = []
    for line in Chem.MolToPDBBlock(mol).splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        line = "HETATM" + line[6:]
        line = line[:17] + resname.rjust(3) + line[20:]      # resName cols 18-20
        line = line[:21] + chain[:1] + line[22:]             # chainID col 22
        line = line[:22] + str(resnum).rjust(4) + line[26:]  # resSeq cols 23-26
        out.append(line)
    return out


def build_complexes_and_metrics(target_pdb: str, sdf_path: str, out_dir: str,
                                target_name: str | None = None):
    """Build target+ligand complex PDBs from a DiffSBDD SDF and score them.

    For every pose in ``sdf_path`` a complex PDB is written to
    ``out_dir/complexes/<target>__<ligand>_complex.pdb`` (target ATOM/TER
    records + ligand HETATM records on chain L, resname LIG — the 3Dmol.js
    viewer styles these as licorice). Druggability metrics for all poses are
    returned as a DataFrame with a ``complex`` column (the complex PDB stem)
    so the viewer summary card can join by structure name.

    Returns ``(complex_paths, metrics_df)``.
    """
    from rdkit import Chem

    cx_dir = os.path.join(out_dir, "complexes")
    os.makedirs(cx_dir, exist_ok=True)
    stem = target_name or os.path.basename(target_pdb).replace(".pdb", "")

    with open(target_pdb) as f:
        target_lines = [l for l in f.read().splitlines()
                        if l.startswith(("ATOM", "TER", "HETATM"))]

    metrics = compute_sdf_metrics(sdf_path)
    complex_paths: list[str] = []
    complex_stems: dict[int, str] = {}

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        lig_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"lig_{i}"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(lig_name)) or f"lig_{i}"
        cx_stem = f"{stem}__{safe}"
        cx_path = os.path.join(cx_dir, f"{cx_stem}_complex.pdb")
        try:
            lines = list(target_lines)
            lines.extend(_ligand_pdb_lines(mol))
            lines.append("TER")
            lines.append("END")
            with open(cx_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            complex_paths.append(cx_path)
            complex_stems[i] = f"{cx_stem}_complex"
        except Exception as e:
            log.warning("complex build failed for ligand %s: %s", lig_name, e)

    if len(metrics):
        metrics["complex"] = metrics["index"].map(complex_stems)
        metrics["target"] = stem
    return complex_paths, metrics
