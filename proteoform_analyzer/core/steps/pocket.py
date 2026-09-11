"""Step: binding-site prediction + pocket drift analysis.

Tiered binding-site prediction:
  1. Reference co-crystal (KNOWN_BINDING_SITES) — most accurate
  2. P2Rank (vendored Java tool) — accurate, ~2 sec/structure
  3. Pure-Python alpha-sphere (scipy) — always available, less accurate
  4. User-specified residues or coordinates

After predicting pockets for all structures, computes pocket drift:
volume change, center displacement, and residue composition change vs WT.
"""
from __future__ import annotations

import os
import re
import sys
import shutil
import subprocess
import logging
import json
import numpy as np
import pandas as pd

from ..pipeline import StepResult

log = logging.getLogger("proteoform_analyzer.pocket")


def _vendored_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "_vendored")


# Known binding sites from reference co-crystal structures
KNOWN_BINDING_SITES = {
    "voxelotor": {"ref_pdb_id": "5E83", "ligand_resname": "5L7", "box_size": 24.0,
                  "align_chains": ["A", "C"]},
    "tafamidis": {"ref_pdb_id": "4DST", "ligand_resname": "9LI", "box_size": 20.0,
                  "align_chains": ["A", "B"]},
}


def _fetch_ref_pdb(pdb_id: str, cache_dir: str = "/tmp/ref_pdbs") -> str:
    """Fetch a reference PDB (cached) via the shared helper.

    Reference co-crystals are used only for known-binding-site alignment, not as
    a structure source.
    """
    from ._pdb_utils import fetch_reference_pdb
    return fetch_reference_pdb(pdb_id, cache_dir)


def _reference_binding_site(pdb_path, ligand_name, target_struct):
    """Tier 1: use reference co-crystal to define binding site."""
    from Bio.PDB import PDBParser, Superimposer
    if ligand_name not in KNOWN_BINDING_SITES:
        return None
    site_info = KNOWN_BINDING_SITES[ligand_name]
    try:
        ref_path = _fetch_ref_pdb(site_info["ref_pdb_id"])
        parser = PDBParser(QUIET=True)
        ref = parser.get_structure("ref", ref_path)[0]
        ligand_coords = []
        for chain in ref:
            for res in chain:
                if res.get_resname() == site_info["ligand_resname"]:
                    for atom in res:
                        ligand_coords.append(atom.get_coord())
        if not ligand_coords:
            return None
        ligand_coords = np.array(ligand_coords)
        ref_center = ligand_coords.mean(axis=0)

        def get_cas(struct, chain_ids):
            atoms = []
            for chain in struct:
                if chain.id in chain_ids:
                    for res in chain:
                        if res.id[0] == " " and "CA" in res:
                            atoms.append(res["CA"])
            return atoms

        ref_cas = get_cas(ref, site_info["align_chains"])
        tgt_cas = get_cas(target_struct, site_info["align_chains"])
        if len(tgt_cas) < 10:
            tgt_cas = [res["CA"] for chain in target_struct for res in chain
                       if res.id[0] == " " and "CA" in res]
        n = min(len(ref_cas), len(tgt_cas))
        if n < 10:
            return None
        sup = Superimposer()
        sup.set_atoms(tgt_cas[:n], ref_cas[:n])
        rot, tran = sup.rotran
        center = np.dot(ref_center, rot) + tran
        return {"center": tuple(center), "box_size": site_info["box_size"],
                "method": "reference", "ref_pdb": site_info["ref_pdb_id"],
                "rmsd": sup.rms}
    except Exception as e:
        log.warning("Reference binding site failed: %s", e)
        return None


def _p2rank_predict(pdb_path, out_dir):
    """Tier 2: P2Rank pocket prediction."""
    prank_dir = os.path.join(_vendored_path(), "p2rank")
    prank_bin = os.path.join(prank_dir, "prank")
    if not os.path.exists(prank_bin):
        return None
    try:
        cmd = [prank_bin, "predict", "-f", pdb_path, "-o", out_dir,
               "-threads", "4", "-log_to_console", "0"]
        subprocess.run(cmd, capture_output=True, timeout=120, cwd=prank_dir)
        # Parse pocket_predictions.csv
        csv_path = None
        for root, _, files in os.walk(out_dir):
            for f in files:
                if f.endswith("_predictions.csv") or f == "pocket_predictions.csv":
                    csv_path = os.path.join(root, f)
                    break
        if not csv_path or not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        if df.empty:
            return None
        # Top pocket (highest score)
        row = df.iloc[0]
        center = (float(row.get("center_x", 0)), float(row.get("center_y", 0)),
                  float(row.get("center_z", 0)))
        score = float(row.get("score", 0))
        volume = float(row.get("volume", 0))
        box_size = max(volume ** (1/3) + 10, 20.0) if volume > 0 else 24.0
        return {"center": center, "box_size": box_size, "method": "p2rank",
                "score": score, "volume": volume}
    except Exception as e:
        log.warning("P2Rank prediction failed: %s", e)
        return None


def _alpha_spheres(pdb_path, r_min: float = 1.5, r_max: float = 6.0):
    """Compute pocket-like alpha-spheres (concave voids) for a structure.

    Delaunay-triangulates the heavy atoms and keeps tetrahedron circumspheres
    (empty of atoms by the Delaunay property) whose radius falls in the
    drug-binding-pocket range ``r_min``-``r_max`` Å. Returns
    ``(centers, radii)`` as ``(N, 3)`` / ``(N,)`` arrays, or ``(None, None)``
    when the structure is unusable or no pocket-like void exists.
    """
    try:
        from scipy.spatial import Delaunay
        from numpy.linalg import norm, solve

        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("p", pdb_path)[0]

        # Collect heavy atom coordinates
        coords = []
        for chain in struct:
            for res in chain:
                if res.id[0] != " ":
                    continue
                for atom in res:
                    if atom.element != "H":
                        coords.append(atom.get_coord())
        coords = np.array(coords)
        if len(coords) < 10:
            return None, None

        tri = Delaunay(coords)
        centers, radii = [], []
        for simplex in tri.simplices:
            # Circumsphere of the tetrahedron. With x = center - pts[0] and
            # bi = pts[i] - pts[0], equal-distance to all vertices gives
            # bi . x = |bi|^2 / 2 — i.e. the edge vectors are the ROWS of the
            # system, and the absolute center is pts[0] + x. (The previous
            # version transposed the system and forgot the pts[0] offset, so
            # both centers and radii were wrong.)
            pts = coords[simplex]
            ba = pts[1] - pts[0]
            ca = pts[2] - pts[0]
            da = pts[3] - pts[0]
            A = np.array([ba, ca, da])
            try:
                b = 0.5 * np.array([np.dot(ba, ba), np.dot(ca, ca), np.dot(da, da)])
                offset = solve(A, b)
                center = pts[0] + offset
                radius = norm(offset)
            except np.linalg.LinAlgError:
                continue
            # Filter: drug-binding pockets have alpha-spheres 1.5-6.0 Å radius
            if r_min <= radius <= r_max:
                centers.append(center)
                radii.append(radius)

        if not centers:
            return None, None
        return np.array(centers), np.array(radii)
    except Exception as e:
        log.debug("Alpha-sphere computation failed for %s: %s", pdb_path, e)
        return None, None


def _alphasphere_predict(pdb_path):
    """Tier 3: pure-Python alpha-sphere pocket detection.

    Uses Delaunay triangulation to find alpha-spheres (concave voids),
    filters by radius, clusters them, and returns the largest cluster.
    """
    from sklearn.cluster import DBSCAN

    alpha_spheres, _radii = _alpha_spheres(pdb_path)
    if alpha_spheres is None or len(alpha_spheres) < 3:
        return None

    try:
        # Cluster alpha-spheres by spatial proximity
        clustering = DBSCAN(eps=3.0, min_samples=2).fit(alpha_spheres)
        labels = clustering.labels_
        if len(set(labels)) <= 1 and -1 in labels:
            return None

        # Find largest cluster
        best_label = max(set(labels) - {-1}, key=list(labels).count) if -1 in labels \
            else max(set(labels), key=list(labels).count)
        cluster = alpha_spheres[labels == best_label]
        center = cluster.mean(axis=0)
        extent = cluster.max(axis=0) - cluster.min(axis=0)
        volume = len(cluster) * 4.0  # rough volume estimate
        box_size = max(extent.max() + 10, 20.0)
        return {"center": tuple(center), "box_size": float(box_size),
                "method": "alphasphere", "score": float(len(cluster)),
                "volume": float(volume)}
    except Exception as e:
        log.warning("Alpha-sphere prediction failed: %s", e)
        return None


def _user_binding_site(config, target_struct):
    """Tier 4: user-specified binding site."""
    if config.binding_site_center:
        c = config.binding_site_center
        return {"center": tuple(c), "box_size": config.binding_site_box_size,
                "method": "user"}
    if config.binding_site_residues:
        from Bio.PDB import PDBParser
        coords = []
        for chain in target_struct:
            for res in chain:
                res_id = f"{chain.id}:{res.id[1]}"
                if res_id in config.binding_site_residues and "CA" in res:
                    coords.append(res["CA"].get_coord())
        if coords:
            center = np.array(coords).mean(axis=0)
            return {"center": tuple(center), "box_size": config.binding_site_box_size,
                    "method": "user"}
    return None


def _fpocket_pockets(pdb_path) -> list[dict]:
    """Run fpocket and return ALL detected pockets, ranked as reported.

    Each entry: ``{"rank": int, "center": (x, y, z), "volume": float,
    "score": float}`` — volume/druggability parsed from ``<stem>_info.txt``,
    center from the pocket's alpha-sphere atoms (``pockets/pocketN_atm.pdb``).
    Returns ``[]`` when fpocket is unavailable or produced nothing usable.
    """
    fpocket_bin = shutil.which("fpocket")
    if not fpocket_bin:
        return []

    import subprocess
    import tempfile
    # fpocket writes next to the input; use an isolated temp copy to avoid
    # clobbering canonical PDBs and to keep runs independent.
    tmpd = tempfile.mkdtemp(prefix="fpocket_")
    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    local_pdb = os.path.join(tmpd, stem + ".pdb")
    try:
        shutil.copy(pdb_path, local_pdb)
        r = subprocess.run([fpocket_bin, "-f", local_pdb],
                           capture_output=True, text=True, timeout=180)
        out_dir = os.path.join(tmpd, stem + "_out")
        if r.returncode != 0 or not os.path.isdir(out_dir):
            log.info("fpocket produced no output for %s (rc=%s)", stem, r.returncode)
            return []

        # Parse per-pocket descriptors ("Pocket N :" blocks).
        info_txt = os.path.join(out_dir, stem + "_info.txt")
        descriptors: dict[int, dict] = {}
        if os.path.exists(info_txt):
            cur_pocket = None
            with open(info_txt) as fh:
                for line in fh:
                    s = line.strip()
                    if s.lower().startswith("pocket"):
                        # e.g. "Pocket 1 :"
                        try:
                            cur_pocket = int(s.split()[1])
                            descriptors.setdefault(cur_pocket, {})
                        except Exception:
                            cur_pocket = None
                    elif cur_pocket is not None and "Volume" in line:
                        try:
                            descriptors[cur_pocket]["volume"] = float(
                                line.split(":")[1].strip().split()[0])
                        except Exception:
                            pass
                    elif cur_pocket is not None and "Druggability Score" in line:
                        try:
                            descriptors[cur_pocket]["score"] = float(
                                line.split(":")[1].strip().split()[0])
                        except Exception:
                            pass

        # Centers from each pocket's alpha-sphere atoms.
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        pockets: list[dict] = []
        pockets_dir = os.path.join(out_dir, "pockets")
        if os.path.isdir(pockets_dir):
            for fname in sorted(os.listdir(pockets_dir)):
                m = re.match(r"pocket(\d+)_atm\.pdb$", fname)
                if not m:
                    continue
                rank = int(m.group(1))
                st = parser.get_structure(f"p{rank}",
                                          os.path.join(pockets_dir, fname))
                coords = [a.get_coord() for a in st.get_atoms()]
                if not coords:
                    continue
                desc = descriptors.get(rank, {})
                pockets.append({
                    "rank": rank,
                    "center": tuple(np.array(coords).mean(axis=0)),
                    "volume": float(desc.get("volume", 0.0)),
                    "score": float(desc.get("score", 0.0)),
                })
        pockets.sort(key=lambda p: p["rank"])
        return pockets
    except Exception as e:
        log.info("fpocket prediction failed for %s: %s", stem, e)
        return []
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def _fpocket_predict(pdb_path):
    """No-Java pocket detection via the ``fpocket`` binary (if installed).

    fpocket is a small C program (``conda install -c conda-forge fpocket`` or a
    system package). It writes ``<stem>_out/`` containing ``<stem>_info.txt``
    (per-pocket descriptors) and ``pockets/pocket1_atm.pdb`` (alpha-sphere
    centers for the top pocket). We parse the #1 pocket's Druggability/Volume
    and compute its center from the alpha-sphere atoms.

    Returns a pocket dict with real (non-zero) ``score``/``volume`` on success,
    or ``None`` if fpocket is unavailable or produced nothing usable.
    """
    pockets = _fpocket_pockets(pdb_path)
    if not pockets:
        return None
    top = pockets[0]
    volume = float(top["volume"])
    box_size = max(volume ** (1 / 3) + 10, 20.0) if volume > 0 else 20.0
    # fpocket druggability score is 0-1; scale to a comparable magnitude.
    return {"center": top["center"], "box_size": float(box_size),
            "method": "fpocket", "score": float(top["score"]),
            "volume": volume}


def _volume_near_center(pdb_path, center, box_size: float) -> dict | None:
    """Measure a REAL pocket volume around a known binding-site center.

    The ``reference`` (co-crystal alignment) and ``user`` binding-site methods
    only provide a *location* — a fixed box — and carry no pocket volume, so
    pocket volumes (and pocket-drift volume changes) used to be identically 0
    for every structure. This measures an actual per-structure volume around
    the site so drift analysis responds to local geometry changes:

      1. fpocket: the detected pocket whose center is closest to the site
         (accepted only when it lies within the site box) contributes its real
         measured volume/druggability.
      2. alpha-spheres whose centers fall inside the site box; the volume uses
         the same per-sphere convention as :func:`_alphasphere_predict`
         (``n_spheres * 4.0``), so numbers are comparable across methods.

    Returns ``{"volume", "score", "volume_method"}`` or ``None`` when no
    detector found any void near the site.
    """
    c = np.array(center, dtype=float)

    # Tier 1: fpocket pocket nearest to the site center.
    pockets = _fpocket_pockets(pdb_path)
    if pockets:
        best, best_d = None, None
        for p in pockets:
            d = float(np.linalg.norm(np.array(p["center"], dtype=float) - c))
            if best_d is None or d < best_d:
                best, best_d = p, d
        # Accept only a pocket genuinely at the site (center inside the box).
        if best is not None and best_d is not None \
                and best_d <= max(float(box_size), 20.0):
            return {"volume": float(best.get("volume", 0.0)),
                    "score": float(best.get("score", 0.0)),
                    "volume_method": "fpocket"}

    # Tier 2: alpha-spheres inside the site box.
    centers, _radii = _alpha_spheres(pdb_path)
    if centers is not None and len(centers):
        half = max(float(box_size), 20.0) / 2.0
        inside = np.all(np.abs(centers - c) <= half, axis=1)
        n = int(inside.sum())
        if n > 0:
            return {"volume": float(n) * 4.0, "score": float(n),
                    "volume_method": "alphasphere"}
    return None


def _predict_pocket(pdb_path, config, ligand_name=None):
    """Run tiered binding-site prediction for a single structure."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    target = parser.get_structure("t", pdb_path)[0]

    method = config.binding_site_method

    # Determine ligand name for reference lookup
    if ligand_name is None:
        uids = set(config.uniprot_ids)
        if uids & {"P69905", "P68871"}:
            ligand_name = "voxelotor"
        elif "P02766" in uids:
            ligand_name = "tafamidis"

    # Tier 1: reference (only when a preferred method wants it)
    if method in ("auto", "reference") and ligand_name:
        result = _reference_binding_site(pdb_path, ligand_name, target)
        if result:
            # The reference method only provides a fixed box (location) — no
            # volume. Measure a real per-structure volume around the site so
            # pocket drift responds to local geometry (Boltz tetramers etc.).
            vol = _volume_near_center(pdb_path, result["center"],
                                      result.get("box_size", 24.0))
            if vol:
                result.update(vol)
            return result

    # Tier 2: P2Rank (needs Java; silently returns None if unavailable)
    if method in ("auto", "p2rank"):
        p2rank_out = os.path.join("/tmp/p2rank_out", os.path.basename(pdb_path))
        os.makedirs(p2rank_out, exist_ok=True)
        result = _p2rank_predict(pdb_path, p2rank_out)
        if result:
            return result

    # Tier 3: user-specified (explicit center/residues)
    if method in ("auto", "user"):
        result = _user_binding_site(config, target)
        if result:
            # Same fixed-box limitation as 'reference': measure a real volume.
            vol = _volume_near_center(pdb_path, result["center"],
                                      result.get("box_size", 24.0))
            if vol:
                result.update(vol)
            return result

    # --- Robust fallback cascade -----------------------------------------
    # IMPORTANT: if the *preferred* method failed (e.g. 'p2rank' with no Java,
    # or 'reference' with no matching ligand), we must NOT drop straight to a
    # geometric center with volume=0 — that produced the "all-zeros pocket +
    # zero drift" bug. Instead, always try the real no-dependency void
    # detectors (fpocket binary, then the pure-Python alpha-sphere method),
    # which yield genuine volumes/centers. This runs for EVERY method, not
    # just 'auto'.
    result = _fpocket_predict(pdb_path)
    if result:
        if method not in ("auto", "fpocket"):
            log.info("Pocket for %s: '%s' unavailable, used fpocket fallback.",
                     os.path.basename(pdb_path), method)
        return result

    result = _alphasphere_predict(pdb_path)
    if result:
        if method not in ("auto", "alphasphere"):
            log.info("Pocket for %s: '%s' unavailable, used alpha-sphere fallback.",
                     os.path.basename(pdb_path), method)
        return result

    # Last resort: geometric center. Marked method='geometric' and flagged so
    # the step can report that NO real detector succeeded (rather than silently
    # emitting zeros that look like a real pocket of size 0).
    coords = []
    for chain in target:
        for res in chain:
            if res.id[0] == " " and "CA" in res:
                coords.append(res["CA"].get_coord())
    coords = np.array(coords)
    center = tuple(coords.mean(axis=0))
    log.warning("Pocket for %s: no pocket detector succeeded (P2Rank/fpocket/"
                "alpha-sphere all unavailable or empty); using geometric center "
                "with unknown volume.", os.path.basename(pdb_path))
    return {"center": center, "box_size": 24.0, "method": "geometric",
            "score": 0.0, "volume": 0.0, "detector_failed": True}


def run_pocket(config, paths: dict) -> StepResult:
    """Predict binding pockets for all structures + compute pocket drift."""
    from .docking import _default_ligand_sdf

    # Collect all current-job PDBs: canonical WT + mutants (resolved through
    # boltz-experiments first), PTM-modified (ptms/ptms/) and proteoform
    # (proteoforms/) structures — see _structure_source.
    from ._structure_source import iter_all_structure_pdbs
    all_pdbs = iter_all_structure_pdbs(config, paths, "pocket",
                                       include_ptms=True, include_proteoforms=True)

    if not all_pdbs:
        return StepResult("pocket", "skipped", "No PDBs found")

    out_dir = paths.get("pockets", os.path.join(paths["results"], "pockets"))
    os.makedirs(out_dir, exist_ok=True)

    ligand_name = None
    ligand_sdf = _default_ligand_sdf(config)
    if ligand_sdf:
        ligand_name = os.path.basename(ligand_sdf).replace(".sdf", "")

    results = []
    outputs = []
    for name, pdb in all_pdbs:
        pocket = _predict_pocket(pdb, config, ligand_name)
        if pocket:
            results.append({
                "structure": name,
                "method": pocket["method"],
                "center_x": pocket["center"][0],
                "center_y": pocket["center"][1],
                "center_z": pocket["center"][2],
                "box_size": pocket.get("box_size", 24.0),
                "volume": pocket.get("volume", 0.0),
                "score": pocket.get("score", 0.0),
                "volume_method": pocket.get("volume_method", ""),
                "detector_failed": bool(pocket.get("detector_failed", False)),
            })

    df = pd.DataFrame(results)
    csv = os.path.join(out_dir, "pocket_predictions.csv")
    df.to_csv(csv, index=False)
    outputs.append(csv)

    # Pocket drift analysis: compare each structure's pocket to WT
    wt_name = next((r["structure"] for r in results if r["structure"].lower().startswith("wt")), None)
    if wt_name:
        wt_row = next(r for r in results if r["structure"] == wt_name)
        drift_rows = []
        for r in results:
            if r["structure"] == wt_name:
                continue
            vol_change = r["volume"] - wt_row["volume"]
            center_disp = np.sqrt(
                (r["center_x"] - wt_row["center_x"])**2 +
                (r["center_y"] - wt_row["center_y"])**2 +
                (r["center_z"] - wt_row["center_z"])**2
            )
            drift_rows.append({
                "proteoform": r["structure"],
                # Keep full precision: rounding to 2 dp erased small but real
                # pocket changes (e.g. 0.004 A^3 or 0.007 A -> 0.00). The GUI
                # formats these for display; the CSV stays high-precision.
                "volume_change": float(vol_change),
                "center_displacement_A": float(center_disp),
                "wt_volume": float(wt_row["volume"]),
                "mut_volume": float(r["volume"]),
            })
        drift_df = pd.DataFrame(drift_rows)
        drift_csv = os.path.join(out_dir, "pocket_drift.csv")
        drift_df.to_csv(drift_csv, index=False)
        outputs.append(drift_csv)

    # --- Zero-volume warning ----------------------------------------------
    # The 'reference'/'user' binding-site methods assign a FIXED box around a
    # known site; the volume attached to them is now MEASURED per structure
    # (fpocket pocket nearest the site, else alpha-spheres inside the box —
    # see _volume_near_center), so pocket-drift volume changes respond to
    # local geometry. Only warn when every volume is STILL zero, i.e. no
    # detector found any void anywhere (install fpocket/Java for P2Rank).
    method_cfg = getattr(config, "binding_site_method", None)
    all_volumes_zero = bool(results) and all(
        float(r.get("volume", 0.0)) == 0.0 for r in results)
    ref_warning = None
    if all_volumes_zero:
        ref_warning = (
            "All predicted pocket volumes are 0, so pocket-drift volume changes "
            "are identically 0 (no pocket detector produced a real volume — the "
            "fpocket/alpha-sphere fallback around the binding site also found no "
            "void). This is a limitation of the available detectors/inputs, not "
            "evidence that the pocket is unchanged. Install `fpocket` or Java "
            "(for P2Rank) to enable detector-based volumes.")
    if ref_warning:
        import json as _json
        warn_path = os.path.join(out_dir, "pocket_warnings.json")
        with open(warn_path, "w") as f:
            _json.dump({"binding_site_method": method_cfg,
                        "all_volumes_zero": all_volumes_zero,
                        "warning": ref_warning}, f, indent=2)
        outputs.append(warn_path)

    n_ok = sum(1 for r in results if r["method"] != "geometric")
    n_failed = sum(1 for r in results if r.get("detector_failed"))
    methods_used = ", ".join(sorted(set(r["method"] for r in results)))
    if n_ok == 0 and results:
        # No real detector worked for ANY structure -> volumes/drift are all
        # placeholder zeros. Report this loudly instead of pretending success.
        status = "ok"  # step ran, but flag the degradation in the message
        msg = (f"Pocket prediction: {len(results)} structures, but NO pocket "
               f"detector succeeded (P2Rank needs Java; fpocket binary and the "
               f"pure-Python alpha-sphere method both returned nothing). Volumes "
               f"and pocket-drift are placeholder zeros. Install Java (for "
               f"P2Rank) or `fpocket`, or check structure quality, to get real "
               f"pockets.")
    else:
        status = "ok" if results else "skipped"
        msg = (f"Pocket prediction: {len(results)} structures, {n_ok} with real "
               f"predicted sites (methods: {methods_used})")
        if n_failed:
            msg += (f"; {n_failed} fell back to geometric center (no detector "
                    f"succeeded for those — install Java/fpocket for full coverage)")
    if ref_warning:
        msg += " | WARNING: " + ref_warning
    return StepResult("pocket", status, msg, outputs=outputs, data=df)
