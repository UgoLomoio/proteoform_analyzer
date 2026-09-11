"""Step: molecular docking.

Two engines (config.docking_engine):
  - "vina" (default, local): AutoDock Vina binary + meeko/rdkit ligand prep.
  - "diffdock" (optional, web service): DiffDock web API; skip if unavailable.

For Vina: prepares the receptor (PDBQT) from a mutant PDB, prepares the ligand
(SDF -> PDBQT via meeko/rdkit), defines a box around the binding site, and runs
vina to get binding affinities (kcal/mol) across variants.
"""
from __future__ import annotations

import os
import sys
import shutil
import subprocess
import logging
import numpy as np
import pandas as pd

from ..pipeline import StepResult
from ._complex_pdb import write_vina_complex, write_boltz_complex

log = logging.getLogger("proteoform_analyzer.docking")

VINA_BIN = shutil.which("vina") or "/usr/local/bin/vina"


def _vendored_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "_vendored")


def _default_ligand_sdf(config) -> str | None:
    """Pick a default ligand SDF based on the protein being analyzed.

    Voxelotor (GBT440) is the FDA-approved HbS polymerization inhibitor for
    hemoglobin; tafamidis is the TTR stabilizer. Falls back to any available SDF.
    """
    if config.ligand_sdf and os.path.exists(config.ligand_sdf):
        return config.ligand_sdf
    lig_dir = os.path.join(_vendored_path(), "ligands")
    # protein-specific selection
    uids = set(config.uniprot_ids)
    if uids & {"P69905", "P68871"}:  # hemoglobin (HBA/HBB)
        for name in ["voxelotor.sdf"]:
            p = os.path.join(lig_dir, name)
            if os.path.exists(p):
                return p
    if "P02766" in uids:  # TTR
        for name in ["tafamidis.sdf", "acoramidis.sdf"]:
            p = os.path.join(lig_dir, name)
            if os.path.exists(p):
                return p
    # generic fallback: any available SDF
    for name in ["voxelotor.sdf", "tafamidis.sdf", "acoramidis.sdf", "diflunisal.sdf", "tolcapone.sdf"]:
        p = os.path.join(lig_dir, name)
        if os.path.exists(p):
            return p
    return None


def _ligand_resname(lig_name: str) -> str:
    """Derive a short (<=3 char) PDB residue name for a docked ligand.

    Uses a curated map for the known ligands (so the viewer shows a recognizable
    HET code), else the first three alphanumeric characters, uppercased.
    """
    known = {
        "voxelotor": "VOX", "tafamidis": "TAF", "acoramidis": "ACO",
        "diflunisal": "DIF", "tolcapone": "TOL", "diflunisal_ideal": "DIF",
        "t44_ideal": "T44", "t44": "T44",
    }
    key = (lig_name or "").strip().lower()
    if key in known:
        return known[key]
    alnum = "".join(c for c in key if c.isalnum())
    return (alnum[:3] or "LIG").upper()


def _sdf_to_pdbqt(sdf_path: str, pdbqt_path: str) -> bool:
    """Convert SDF -> PDBQT. Uses meeko (vina-compatible) with obabel fallback."""
    # Try meeko first (produces vina-compatible PDBQT)
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
        mol = Chem.MolFromMolFile(sdf_path, sanitize=False, removeHs=False)
        if mol is None:
            log.warning("rdkit could not parse %s", sdf_path)
        else:
            Chem.SanitizeMol(mol)
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            result = PDBQTWriterLegacy.write_string(mol_setups[0])
            pdbqt_str = result[0] if isinstance(result, tuple) else result
            if pdbqt_str:
                with open(pdbqt_path, "w") as f:
                    f.write(pdbqt_str)
                return True
            log.warning("meeko produced empty PDBQT for %s", sdf_path)
    except Exception as e:
        log.warning("meeko SDF->PDBQT failed: %s", e)

    # Fallback: obabel (may need format fix for vina)
    try:
        import subprocess
        r = subprocess.run(["obabel", sdf_path, "-O", pdbqt_path, "-xr"],
                           capture_output=True, text=True, timeout=60)
        if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
            return True
    except Exception as e:
        log.warning("obabel SDF->PDBQT failed: %s", e)
    return False


def _pdb_to_pdbqt(pdb_path: str, pdbqt_path: str) -> bool:
    """Convert receptor PDB -> PDBQT.

    Strips non-protein HETATMs (heme, waters) first, then converts via obabel
    (robust) with a meeko fallback.
    """
    # First strip HETATMs to a clean protein-only PDB
    from Bio.PDB import PDBParser, PDBIO, Select
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("r", pdb_path)

        class ProteinOnly(Select):
            def accept_residue(self, residue):
                from Bio.PDB.Polypeptide import is_aa
                return is_aa(residue, standard=True)

        clean_pdb = pdbqt_path + ".clean.pdb"
        io = PDBIO()
        io.set_structure(struct)
        io.save(clean_pdb, ProteinOnly())
    except Exception as e:
        log.warning("protein-only filter failed: %s", e)
        clean_pdb = pdb_path

    # Try obabel (robust PDB->PDBQT)
    try:
        import subprocess
        r = subprocess.run(["obabel", clean_pdb, "-O", pdbqt_path, "-xr"],
                           capture_output=True, text=True, timeout=60)
        if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
            return True
    except Exception as e:
        log.warning("obabel PDB->PDBQT failed: %s", e)

    # Fallback: meeko polymer prep
    try:
        from meeko import PolymerPreparation
        preparator = PolymerPreparation()
        preparator.prepare_pdb(clean_pdb, pdbqt_path)
        return True
    except Exception:
        # last resort: copy the clean PDB (vina can read PDB in some modes)
        try:
            shutil.copy(clean_pdb, pdbqt_path)
            return True
        except Exception:
            return False


# Known binding-site centers from reference PDB complexes, in the reference frame.
# These are transformed into the target structure's frame by CA superposition.
# Format: {pdb_id_of_reference: (center_xyz, box_size, reference_chains_for_alignment)}
KNOWN_BINDING_SITES = {
    # Voxelotor (5L7) bound to hemoglobin alpha chain - PDB 5E83
    # Center = centroid of voxelotor ligand; box = ligand extent + 15 A padding
    "voxelotor": {
        "ref_pdb_id": "5E83",
        "ligand_resname": "5L7",
        "box_size": 24.0,
        "align_chains": ["A", "C"],  # alpha chains for alignment
    },
    # Tafamidis bound to TTR - PDB 4DST (analogous approach)
    "tafamidis": {
        "ref_pdb_id": "4DST",
        "ligand_resname": "9LI",
        "box_size": 20.0,
        "align_chains": ["A", "B"],
    },
}


def _fetch_ref_pdb(pdb_id: str, cache_dir: str = "/tmp/ref_pdbs") -> str:
    """Fetch a reference PDB (cached) via the shared helper.

    Reference co-crystals are used only for known-binding-site alignment, not as
    a structure source.
    """
    from ._pdb_utils import fetch_reference_pdb
    return fetch_reference_pdb(pdb_id, cache_dir)


def _binding_site_center(pdb_path: str, ligand_name: str = None) -> tuple[tuple, float]:
    """Compute the binding-site center for docking.

    If a known binding site is available (e.g., voxelotor in PDB 5E83), the center
    is derived from the reference ligand coordinates, transformed into the target
    structure's frame by CA superposition of the relevant chains. This gives a
    biologically accurate docking box at the known ligand binding site.

    Falls back to the geometric center of the structure if no reference is available.
    Returns (center_xyz, box_size).
    """
    from Bio.PDB import PDBParser, Superimposer

    parser = PDBParser(QUIET=True)
    target = parser.get_structure("t", pdb_path)[0]

    # Determine ligand name from the file name if not provided
    if ligand_name is None:
        fname = os.path.basename(pdb_path).lower()
        if "voxelotor" in fname or "p69905" in fname or "p68871" in fname or "1a3n" in fname:
            ligand_name = "voxelotor"
        elif "tafamidis" in fname or "p02766" in fname:
            ligand_name = "tafamidis"

    # Try to use a known binding site
    if ligand_name and ligand_name in KNOWN_BINDING_SITES:
        site_info = KNOWN_BINDING_SITES[ligand_name]
        try:
            ref_path = _fetch_ref_pdb(site_info["ref_pdb_id"])
            ref_struct = parser.get_structure("ref", ref_path)[0]

            # Get ligand coordinates from reference
            ligand_coords = []
            for chain in ref_struct:
                for res in chain:
                    if res.get_resname() == site_info["ligand_resname"]:
                        for atom in res:
                            ligand_coords.append(atom.get_coord())
            if not ligand_coords:
                log.warning("Ligand %s not found in reference %s; using geometric center",
                            site_info["ligand_resname"], site_info["ref_pdb_id"])
                raise ValueError("No ligand in reference")

            ligand_coords = np.array(ligand_coords)
            ref_center = ligand_coords.mean(axis=0)

            # Align reference to target using CA atoms of specified chains
            def get_cas(struct, chain_ids):
                atoms = []
                for chain in struct:
                    if chain.id in chain_ids:
                        for res in chain:
                            if res.id[0] == " " and "CA" in res:
                                atoms.append(res["CA"])
                return atoms

            ref_cas = get_cas(ref_struct, site_info["align_chains"])
            # For target, try the same chains; if not found, use all chains
            tgt_cas = get_cas(target, site_info["align_chains"])
            if len(tgt_cas) < 10:
                tgt_cas = [res["CA"] for chain in target for res in chain
                           if res.id[0] == " " and "CA" in res]

            n = min(len(ref_cas), len(tgt_cas))
            if n < 10:
                log.warning("Too few CA atoms for alignment (%d); using geometric center", n)
                raise ValueError("Insufficient atoms for alignment")

            sup = Superimposer()
            sup.set_atoms(tgt_cas[:n], ref_cas[:n])
            rot, tran = sup.rotran
            # Transform ref center into target frame
            center = np.dot(ref_center, rot) + tran
            box_size = site_info["box_size"]
            log.info("Binding site from ref %s: center=(%.2f, %.2f, %.2f), box=%.1f A, align RMSD=%.2f",
                     site_info["ref_pdb_id"], center[0], center[1], center[2],
                     box_size, sup.rms)
            return tuple(center), box_size
        except Exception as e:
            log.warning("Known binding site failed (%s); falling back to geometric center", e)

    # Fallback: geometric center of the structure
    coords = []
    for chain in target:
        for res in chain:
            if res.id[0] == " " and "CA" in res:
                coords.append(res["CA"].get_coord())
    coords = np.array(coords)
    center = tuple(coords.mean(axis=0))
    log.info("Using geometric center: (%.2f, %.2f, %.2f), box=20.0 A", *center)
    return center, 20.0


def _run_vina(receptor_pdbqt, ligand_pdbqt, center, size, out_dir, name):
    """Run vina docking. Returns (affinity, log_path).

    Note: vina v1.2.5 does not support --log; results are parsed from the
    output PDBQT (REMARK VINA RESULT lines).  Exit code may be non-zero
    even on success, so we check for output files.
    """
    out_pdbqt = os.path.join(out_dir, f"{name}_vina_out.pdbqt")
    cmd = [VINA_BIN, "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt,
           "--center_x", str(center[0]), "--center_y", str(center[1]),
           "--center_z", str(center[2]),
           "--size_x", str(size), "--size_y", str(size), "--size_z", str(size),
           "--out", out_pdbqt, "--exhaustiveness", "8", "--num_modes", "5"]
    log.info("vina: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=300)
    except Exception as e:
        log.warning("vina run failed for %s: %s", name, e)
        return None, None
    # check for output file
    if not os.path.exists(out_pdbqt) or os.path.getsize(out_pdbqt) == 0:
        log.warning("vina produced no output for %s", name)
        return None, None
    # parse affinity from the output PDBQT (REMARK VINA RESULT line)
    aff = None
    with open(out_pdbqt) as f:
        for line in f:
            if "VINA RESULT" in line:
                parts = line.split()
                try:
                    aff = float(parts[3])
                except (ValueError, IndexError):
                    pass
                break
    return aff, out_pdbqt


def _run_diffdock(config, paths, pdbs, ligand_sdf):
    """DiffDock web-service path (wired, may skip if service unavailable)."""
    # The original code used a DiffDock web service. We wire a placeholder that
    # documents the limitation.
    log.warning("DiffDock web service not available in this environment; skipping.")
    return StepResult("docking", "skipped",
                      "DiffDock web service unavailable (use 'vina' or 'boltz2' engine)")


# ---------------------------------------------------------------------------
# Boltz-2 co-folding docking
# ---------------------------------------------------------------------------

def _sdf_to_smiles(sdf_path: str) -> str | None:
    """Derive a canonical SMILES from an SDF via RDKit."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromMolFile(sdf_path, sanitize=True, removeHs=True)
        if mol is None:
            mol = Chem.MolFromMolFile(sdf_path, sanitize=False, removeHs=True)
            if mol is not None:
                Chem.SanitizeMol(mol)
        if mol is None:
            log.warning("RDKit could not parse %s for SMILES", sdf_path)
            return None
        return Chem.MolToSmiles(mol)
    except Exception as e:
        log.warning("SDF->SMILES failed for %s: %s", sdf_path, e)
        return None


def _receptor_sequences_for_docking(config, pdb_path):
    """Return list of (chain_id, sequence) to fold as the receptor.

    Prefers the UniProt sequences on config (deduplicated per unique subunit so a
    homo-tetramer folds one representative chain, keeping the co-folding job
    small); falls back to extracting the sequence from the PDB.

    When mature (post-cleavage) regions are resolved, the mature sequences are
    used: the receptor that binds the ligand in vivo is the processed protein.
    """
    seqs = None
    try:
        from .sequence import get_mature_sequences
        seqs = get_mature_sequences(config)
    except Exception:  # pragma: no cover - defensive
        seqs = None
    if not seqs:
        seqs = getattr(config, "_sequences", None)
    if seqs:
        out = []
        letters = [chr(65 + i) for i in range(len(seqs))]
        for i, s in enumerate(seqs):
            out.append((letters[i], s))
        return out
    # Fallback: extract one chain sequence from the PDB
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import is_aa
        from ._pdb_utils import three_to_one
        parser = PDBParser(QUIET=True)
        model = parser.get_structure("r", pdb_path)[0]
        for chain in model:
            resnames = [r.get_resname() for r in chain if is_aa(r, standard=True)]
            if resnames:
                seq = "".join(three_to_one(r) for r in resnames)
                return [(chain.id, seq)]
    except Exception as e:
        log.warning("Could not extract receptor sequence from %s: %s", pdb_path, e)
    return []


def _boltz2_dock_yaml(receptor_seqs, ligand_smiles) -> dict:
    """Build a Boltz-2 YAML dict for protein(s) + one small-molecule ligand."""
    seqs = []
    used = set()
    for cid, s in receptor_seqs:
        seqs.append({"protein": {"id": cid, "sequence": s}})
        used.add(cid)
    # pick a ligand chain id not already used
    lig_id = next((c for c in "XYZLBCDEFG" if c not in used), "X")
    seqs.append({"ligand": {"id": lig_id, "smiles": ligand_smiles}})
    return {"version": 1, "sequences": seqs}


def _parse_boltz_confidence(out_dir: str) -> dict:
    """Parse Boltz-2 confidence JSON(s) into a flat metrics dict."""
    metrics = {}
    conf = None
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.endswith("_summary.json") or f.startswith("confidence") or \
               f.endswith("_confidence_0.json"):
                try:
                    import json
                    with open(os.path.join(root, f)) as fh:
                        conf = json.load(fh)
                    break
                except Exception:
                    continue
        if conf:
            break
    if isinstance(conf, dict):
        for k in ("confidence_score", "ptm", "iptm", "complex_plddt",
                  "complex_iplddt", "ligand_iptm", "protein_iptm"):
            if k in conf:
                metrics[k] = conf[k]
    return metrics


def _find_boltz_cif(cif_hint, job_out: str) -> str | None:
    """Return a usable Boltz output mmCIF path.

    ``cif_hint`` may already be the path; otherwise search ``job_out`` for a
    ``*.cif`` (the co-folded receptor+ligand model).
    """
    if cif_hint and isinstance(cif_hint, str) and os.path.exists(cif_hint) \
            and cif_hint.endswith((".cif", ".mmcif")):
        return cif_hint
    if job_out and os.path.isdir(job_out):
        cands = []
        for root, _, files in os.walk(job_out):
            for f in files:
                if f.endswith((".cif", ".mmcif")):
                    cands.append(os.path.join(root, f))
        # prefer rank-0 / model_0 if present
        cands.sort(key=lambda p: ("model_0" not in p and "rank_0" not in p
                                  and "_0." not in p, len(p)))
        if cands:
            return cands[0]
    return None


def _register_boltz_complex(cif_hint, job_out: str, out_dir: str,
                            job_label: str) -> str | None:
    """Convert a Boltz co-fold mmCIF into a viewable complex PDB.

    Writes to ``docking/complexes/<job_label>_complex.pdb`` (the Boltz model
    already contains the receptor AND ligand). Returns the path or ``None``.
    """
    cif = _find_boltz_cif(cif_hint, job_out)
    if not cif:
        log.warning("Boltz complex: no mmCIF found for %s", job_label)
        return None
    cx_dir = os.path.join(out_dir, "complexes")
    cx_path = os.path.join(cx_dir, f"{job_label}_complex.pdb")
    return write_boltz_complex(cif, cx_path)


def _run_boltz2_dock(config, paths, pdbs) -> StepResult:
    """Dock by co-folding receptor + ligand (SMILES) with Boltz-2.

    Runs one Boltz-2 co-fold per (structure, ligand). Reports confidence metrics
    (pTM / ipTM / complex pLDDT) as a binding-confidence proxy — NOT a Vina
    kcal/mol affinity. Backend: hosted Boltz API, else a local `boltz` binary
    (synchronous). No graft fallback for docking; skips cleanly if neither is
    available.
    """
    import yaml as _yaml
    ligand_sdfs = _resolve_ligands(config)
    if not ligand_sdfs:
        return StepResult("docking", "skipped", "No ligand SDF available for Boltz-2 docking")

    out_dir = os.path.join(paths["docking"], "boltz2")
    os.makedirs(out_dir, exist_ok=True)

    # Resolve ligand SMILES up front
    lig_smiles = {}
    for sdf in ligand_sdfs:
        smi = _sdf_to_smiles(sdf)
        if smi:
            lig_smiles[os.path.basename(sdf)] = smi
    if not lig_smiles:
        return StepResult("docking", "skipped",
                          "Could not derive SMILES from any ligand SDF (RDKit)")

    # Execution mode (mirror boltz2_fold). Backend chosen by the shared resolver:
    # API (key) -> local binary (synchronous) -> clean skip. Docking has NO graft
    # fallback (a grafted backbone-identical structure carries no docking signal),
    # so backend=="none" skips cleanly. The Biomni HPC backend was removed in
    # v3.4.0.
    from .boltz2_fold import _resolve_local_binary, _run_local_boltz
    from . import _boltz_backend as bb
    backend = bb.resolve_backend(config, "dock")
    log.info("Boltz docking backend resolved to: %s", backend)
    if backend == "none":
        return StepResult(
            "docking", "skipped",
            "Boltz-2 docking has no available backend. Provide a Boltz API key "
            "(boltz2.api_key or $BOLTZ_API_KEY) or a local 'boltz' binary "
            "(boltz2.prefer_local=True). "
            "(Docking has no graft fallback — a backbone-identical graft carries "
            "no binding signal.)")
    local_bin = _resolve_local_binary(config)

    results = []
    outputs = []
    n_done = 0

    for name, pdb in pdbs:
        rec_seqs = _receptor_sequences_for_docking(config, pdb)
        if not rec_seqs:
            log.warning("No receptor sequence for %s; skipping", name)
            continue
        for lig_file, smi in lig_smiles.items():
            lig_name = lig_file.replace(".sdf", "")
            job_label = f"{name}__{lig_name}"
            spec = _boltz2_dock_yaml(rec_seqs, smi)
            yaml_path = os.path.join(out_dir, f"{job_label}.yaml")
            with open(yaml_path, "w") as f:
                _yaml.safe_dump(spec, f, sort_keys=False)
            outputs.append(yaml_path)

            row = {"structure": name, "ligand": lig_file, "engine": "boltz2"}
            if backend == "api":
                job_out = os.path.join(out_dir, "api", job_label)
                try:
                    _cif, metrics = bb.run_api_dock(config, spec, job_out,
                                                    name=job_label)
                    row.update(metrics)
                    # Boltz co-fold already contains receptor+ligand: cif->pdb
                    cx = _register_boltz_complex(_cif, job_out, out_dir, job_label)
                    if cx:
                        row["complex_pdb"] = cx
                        outputs.append(cx)
                    n_done += 1
                except Exception as e:
                    if bb.is_auth_error(e):
                        log.warning("Boltz API key invalid or unauthorized for "
                                    "docking; falling back to local boltz.")
                        # Switch to local boltz for this and remaining jobs.
                        if local_bin:
                            backend = "local"
                            job_out = os.path.join(out_dir, "local", job_label)
                            cfg_local = config
                            cfg_local.boltz2.use_msa_server = config.boltz2.dock_use_msa_server
                            cif = _run_local_boltz(local_bin, yaml_path, job_out, cfg_local)
                            if cif:
                                metrics = _parse_boltz_confidence(job_out)
                                row.update(metrics)
                                cx = _register_boltz_complex(cif, job_out, out_dir, job_label)
                                if cx:
                                    row["complex_pdb"] = cx
                                    outputs.append(cx)
                                n_done += 1
                            results.append(row)
                            continue
                        # No local boltz: docking has no graft fallback -> skip.
                        row["status"] = ("skipped: Boltz API unavailable (invalid "
                                         "key) and no local 'boltz' install")
                        results.append(row)
                        continue
                    log.error("Boltz API docking failed for %s: %s", job_label, e)
                    row["status"] = f"api_failed: {e}"
                results.append(row)
            else:  # local backend (synchronous)
                job_out = os.path.join(out_dir, "local", job_label)
                # local boltz uses dock_use_msa_server flag
                cfg_local = config
                cfg_local.boltz2.use_msa_server = config.boltz2.dock_use_msa_server
                cif = _run_local_boltz(local_bin, yaml_path, job_out, cfg_local)
                if cif:
                    metrics = _parse_boltz_confidence(job_out)
                    row.update(metrics)
                    cx = _register_boltz_complex(cif, job_out, out_dir, job_label)
                    if cx:
                        row["complex_pdb"] = cx
                        outputs.append(cx)
                    n_done += 1
                results.append(row)

    df = pd.DataFrame(results)
    csv = os.path.join(out_dir, "docking_summary_boltz2.csv")
    df.to_csv(csv, index=False)
    outputs.append(csv)

    status = "ok" if n_done else "skipped"
    if backend == "api":
        msg = f"Boltz-2 docking (API): {n_done} complexes co-folded"
    else:
        msg = f"Boltz-2 docking (local): {n_done} complexes folded"
    return StepResult("docking", status, msg, outputs=outputs, data=df)




def _extract_md_snapshots(traj_path: str, n_snapshots: int, out_dir: str, name: str) -> list[str]:
    """Extract N representative snapshots from an MD trajectory via uniform sampling.

    Returns list of PDB file paths. Falls back to just the final frame if the
    trajectory has too few frames.
    """
    import mdtraj
    os.makedirs(out_dir, exist_ok=True)
    traj = mdtraj.load(traj_path)
    n_frames = traj.n_frames
    if n_frames == 0:
        return []
    if n_frames <= n_snapshots:
        indices = list(range(n_frames))
    else:
        # Uniform sampling including first and last
        indices = [int(i * (n_frames - 1) / (n_snapshots - 1)) for i in range(n_snapshots)]
    snapshot_paths = []
    for idx in indices:
        snap_path = os.path.join(out_dir, f"{name}_snap{idx}.pdb")
        traj[idx].save_pdb(snap_path)
        snapshot_paths.append(snap_path)
    log.info("Extracted %d snapshots from %s (trajectory had %d frames)",
             len(snapshot_paths), name, n_frames)
    return snapshot_paths


def _find_md_trajectory(name: str, md_dir: str) -> str | None:
    """Find the MD trajectory PDB for a given structure name."""
    traj_path = os.path.join(md_dir, name, "trajectory.pdb")
    if os.path.exists(traj_path):
        return traj_path
    return None


def _resolve_ligands(config) -> list[str]:
    """Resolve the list of ligand SDFs to screen.

    Priority: config.ligand_sdfs (multi) > config.ligand_sdf (single) > auto-select.
    """
    ligands = []
    if config.ligand_sdfs:
        for p in config.ligand_sdfs:
            if os.path.exists(p):
                ligands.append(p)
    if not ligands and config.ligand_sdf and os.path.exists(config.ligand_sdf):
        ligands.append(config.ligand_sdf)
    if not ligands:
        auto = _default_ligand_sdf(config)
        if auto:
            ligands.append(auto)
    return ligands


def run_docking(config, paths: dict) -> StepResult:
    """Run docking across WT + mutants.

    Supports:
    - Multi-ligand screening: iterates over config.ligand_sdfs (or auto-selected).
    - Ensemble docking: if config.ensemble_docking, extracts MD snapshots and
      docks each, reporting mean +/- SD affinity per structure.
    """
    out_dir = os.path.join(paths["docking"], config.docking_engine)
    os.makedirs(out_dir, exist_ok=True)

    # Structures resolve through boltz-experiments first (see _structure_source).
    # Cover canonical WT + mutants AND PTM-modified structures (ptms/ptms/).
    # Proteoform (mutation+PTM combo) structures are excluded to avoid a
    # combinatorial explosion of docking runs.
    from ._structure_source import iter_all_structure_pdbs
    pdbs = iter_all_structure_pdbs(config, paths, "docking",
                                   include_ptms=True, include_proteoforms=False)
    if not pdbs:
        return StepResult("docking", "skipped", "No PDBs to dock")

    if config.docking_engine == "diffdock":
        ligand = _default_ligand_sdf(config)
        return _run_diffdock(config, paths, pdbs, ligand)

    if config.docking_engine == "boltz2":
        return _run_boltz2_dock(config, paths, pdbs)

    # --- Vina ---
    if not os.path.exists(VINA_BIN):
        return StepResult("docking", "skipped", f"vina binary not found at {VINA_BIN}")

    ligand_sdfs = _resolve_ligands(config)
    if not ligand_sdfs:
        return StepResult("docking", "skipped", "No ligand SDF available")

    ensemble = config.ensemble_docking
    n_snapshots = config.ensemble_n_snapshots
    md_dir = paths.get("molecular_dynamics", "")

    results = []
    outputs = []

    for ligand_sdf in ligand_sdfs:
        lig_name = os.path.basename(ligand_sdf).replace(".sdf", "")
        lig_out_dir = os.path.join(out_dir, lig_name)
        os.makedirs(lig_out_dir, exist_ok=True)

        # Prepare ligand PDBQT
        lig_pdbqt = os.path.join(lig_out_dir, "ligand.pdbqt")
        if not _sdf_to_pdbqt(ligand_sdf, lig_pdbqt):
            log.warning("Ligand SDF->PDBQT failed for %s; skipping", lig_name)
            continue
        outputs.append(lig_pdbqt)
        log.info("Prepared ligand PDBQT from %s", lig_name)

        for name, pdb in pdbs:
            if ensemble and md_dir and os.path.isdir(md_dir):
                # Ensemble docking: extract MD snapshots, dock each
                traj_path = _find_md_trajectory(name, md_dir)
                if traj_path and os.path.exists(traj_path):
                    snap_dir = os.path.join(lig_out_dir, "snapshots", name)
                    snapshots = _extract_md_snapshots(traj_path, n_snapshots, snap_dir, name)
                    if not snapshots:
                        # Fallback to static structure
                        snapshots = [pdb]
                else:
                    snapshots = [pdb]

                affinities = []
                complex_pdb = None
                for i, snap_pdb in enumerate(snapshots):
                    snap_name = f"{name}_snap{i}" if len(snapshots) > 1 else name
                    rec_pdbqt = os.path.join(lig_out_dir, f"{snap_name}_receptor.pdbqt")
                    if not _pdb_to_pdbqt(snap_pdb, rec_pdbqt):
                        continue
                    outputs.append(rec_pdbqt)
                    center, box_size = _binding_site_center(
                        snap_pdb, ligand_name=lig_name)
                    aff, logf = _run_vina(rec_pdbqt, lig_pdbqt, center, box_size,
                                          lig_out_dir, snap_name)
                    if logf:
                        outputs.append(logf)
                        # Write one representative complex (first successful pose)
                        if complex_pdb is None:
                            cx_dir = os.path.join(out_dir, "complexes")
                            cx_path = os.path.join(
                                cx_dir, f"{name}__{lig_name}_complex.pdb")
                            complex_pdb = write_vina_complex(
                                snap_pdb, logf, cx_path,
                                ligand_resname=_ligand_resname(lig_name))
                            if complex_pdb:
                                outputs.append(complex_pdb)
                    if aff is not None:
                        affinities.append(aff)
                    log.info("%s/%s snap%d: affinity = %s", lig_name, name, i, aff)

                if affinities:
                    mean_aff = float(np.mean(affinities))
                    std_aff = float(np.std(affinities)) if len(affinities) > 1 else 0.0
                    results.append({
                        "structure": name, "ligand": os.path.basename(ligand_sdf),
                        "affinity_kcal_mol": mean_aff,
                        "affinity_std": std_aff,
                        "n_snapshots": len(affinities),
                        "ensemble": True,
                        "complex_pdb": complex_pdb,
                    })
                    log.info("%s/%s: ensemble mean = %.3f +/- %.3f kcal/mol (n=%d)",
                             lig_name, name, mean_aff, std_aff, len(affinities))
                else:
                    results.append({
                        "structure": name, "ligand": os.path.basename(ligand_sdf),
                        "affinity_kcal_mol": None, "ensemble": True,
                    })
            else:
                # Standard single-structure docking
                rec_pdbqt = os.path.join(lig_out_dir, f"{name}_receptor.pdbqt")
                if not _pdb_to_pdbqt(pdb, rec_pdbqt):
                    log.warning("receptor prep failed for %s", name)
                    continue
                outputs.append(rec_pdbqt)
                center, box_size = _binding_site_center(pdb, ligand_name=lig_name)
                aff, logf = _run_vina(rec_pdbqt, lig_pdbqt, center, box_size,
                                      lig_out_dir, name)
                if logf:
                    outputs.append(logf)
                # Merge receptor + best docked pose into a viewable complex PDB
                complex_pdb = None
                if logf:
                    cx_dir = os.path.join(out_dir, "complexes")
                    cx_path = os.path.join(cx_dir, f"{name}__{lig_name}_complex.pdb")
                    complex_pdb = write_vina_complex(
                        pdb, logf, cx_path,
                        ligand_resname=_ligand_resname(lig_name))
                    if complex_pdb:
                        outputs.append(complex_pdb)
                results.append({"structure": name, "ligand": os.path.basename(ligand_sdf),
                                "affinity_kcal_mol": aff, "ensemble": False,
                                "complex_pdb": complex_pdb})
                log.info("%s/%s: affinity = %s kcal/mol", lig_name, name, aff)

    df = pd.DataFrame(results)
    csv = os.path.join(out_dir, "docking_summary.csv")
    df.to_csv(csv, index=False)
    outputs.append(csv)
    n_ok = sum(1 for r in results if r["affinity_kcal_mol"] is not None)
    n_ligands = len(ligand_sdfs)
    mode = "ensemble" if ensemble else "single"
    return StepResult(
        "docking", "ok" if n_ok else "skipped",
        f"Vina docking ({mode}): {n_ok}/{len(results)} docked, {n_ligands} ligand(s)",
        outputs=outputs, data=df,
    )
