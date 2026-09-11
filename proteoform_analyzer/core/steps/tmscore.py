"""Step: TM-score structural comparison via TMalign (tmalign_wrapper.py strategy).

Computes pairwise TM-scores between the WT structure and each mutant, and a
full mutant-vs-mutant matrix. Uses the same temporary-file + CA-only + TER-safe
strategy as tmalign_wrapper.py. Falls back to a pure-Python CA superposition
estimate if the binary is unavailable or fails.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer

from ..pipeline import StepResult

log = logging.getLogger("proteoform_analyzer.tmscore")

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_TMALIGN = _PACKAGE_ROOT / "_vendored" / "tmalign_exe" / "TMalign_cpp"


def _save_ca_pdb_with_ter(source: str, destination: str) -> None:
    """
    Write a CA-only PDB, keeping all chains and inserting TER between them.
    This format is accepted by TMalign for multi-chain structures.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", source)

    with open(destination, "w", encoding="utf-8") as out:
        atom_serial = 1
        for model in struct:
            for chain in model:
                for res in chain:
                    if "CA" not in res:
                        continue
                    ca = res["CA"]
                    # PDB ATOM line with fixed-width columns
                    line = (
                        f"ATOM  {atom_serial:5d}  CA  {res.resname:<3} {chain.id}"
                        f"{res.id[1]:4d}    "
                        f"{ca.coord[0]:8.3f}{ca.coord[1]:8.3f}{ca.coord[2]:8.3f}"
                        f"{ca.occupancy or 1.0:6.2f}{ca.bfactor or 0.0:6.2f}          C\n"
                    )
                    out.write(line)
                    atom_serial += 1
                out.write("TER\n")
            out.write("END\n")


def _parse_tm_score(lines: list[str]) -> Optional[float]:
    """Parse the first TM-score reported by TMalign (same regex as tmalign_wrapper.py)."""
    pattern = re.compile(r"TM-score\s*=\s*(\d*\.\d*)", re.IGNORECASE)
    for line in lines:
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None


def _run_tmalign_binary(a: str, b: str, args: str = "") -> Optional[float]:
    """
    Run TMalign using the tmalign_wrapper.py strategy:

    - Create temporary PDBs.
    - Write CA-only, no-TER versions of the inputs.
    - Call TMalign on those temp files.
    - Parse TM-score from stdout.
    """
    if not _TMALIGN.exists():
        log.warning("TMalign binary not found: %s", _TMALIGN)
        return None

    mobile_tmp = target_tmp = matrix_tmp = None
    try:
        mobile_tmp = tempfile.mktemp(".pdb", "mobile_")
        target_tmp = tempfile.mktemp(".pdb", "target_")
        matrix_tmp = tempfile.mktemp(".txt", "matrix_")

        _save_ca_pdb_with_ter(a, mobile_tmp)
        _save_ca_pdb_with_ter(b, target_tmp)

        command = [str(_TMALIGN), mobile_tmp, target_tmp, "-m", matrix_tmp, *args.split()]
        log.debug("Running TMalign: %s", command)

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
        lines = process.stdout.splitlines()
        #log.info("TMalign raw output:\n%s", process.stdout)

        # TMalign >= 2012/04/17 also writes matrix/alignment to the -m file
        if matrix_tmp and os.path.exists(matrix_tmp):
            with open(matrix_tmp) as f:
                lines += f.read().splitlines()

        score = _parse_tm_score(lines)
        if score is None:
            log.warning("TMalign produced no parseable score (exit=%s)", process.returncode)
        return score

    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("TMalign execution failed: %s", exc)
        return None
    finally:
        for filename in (mobile_tmp, target_tmp, matrix_tmp):
            if filename:
                try:
                    os.remove(filename)
                except OSError:
                    pass


def _tmscore_fallback(a: str, b: str) -> float:
    """Estimate TM-score from ordered C-alpha coordinates."""
    parser = PDBParser(QUIET=True)
    structures = [parser.get_structure(name, filename) for name, filename in (("a", a), ("b", b))]

    def ca_atoms(structure):
        return [
            residue["CA"]
            for model in structure
            for chain in model
            for residue in chain
            if "CA" in residue
        ]

    ca_a, ca_b = map(ca_atoms, structures)
    n = min(len(ca_a), len(ca_b))
    if n == 0:
        return 0.0

    fixed, moving = ca_a[:n], ca_b[:n]
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    coords_a = np.asarray([atom.get_coord() for atom in fixed])
    coords_b = np.asarray([atom.get_coord() for atom in moving])
    # Apply superposition to moving set
    coords_b = (coords_b - sup.rotran[1]) @ sup.rotran[0]

    distances = np.linalg.norm(coords_a - coords_b, axis=1)
    d0 = 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8 if n > 15 else 0.5
    d0 = max(d0, 0.5)
    return float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))


def _score(a: str, b: str, args: str, use_binary: bool) -> float:
    if a == b:
        return 1.0
    if use_binary:
        score = _run_tmalign_binary(a, b, args)
        if score is not None:
            return score
    return _tmscore_fallback(a, b)


def compute_tmscores(config, paths: dict) -> StepResult:
    """Compute WT-vs-mutant scores and a symmetric all-vs-all matrix."""
    pdb_dir = Path(paths["pdbs_monomer"] if config.is_monomer else paths["pdbs"])
    out_dir = Path(paths["tmalign"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Structures resolve through boltz-experiments first (see _structure_source).
    # Cover the whole current job: canonical WT + mutants, PTM-modified
    # (ptms/ptms/) and proteoform (proteoforms/) structures.
    from ._structure_source import iter_structure_pdbs, iter_all_structure_pdbs
    pdbs = dict(iter_all_structure_pdbs(config, paths, "tmscore",
                                        include_ptms=True,
                                        include_proteoforms=True))
    if not pdbs:
        return StepResult("tmscore", "skipped", f"No PDBs in {pdb_dir}")

    wt_name = next((name for name in pdbs if name.lower().startswith("wt")), None)
    if wt_name is None:
        return StepResult("tmscore", "skipped", "No WT structure found")

    args = getattr(config, "tmscore_args", "")
    use_binary = _TMALIGN.exists()
    log.info("TM-score engine: %s", "binary" if use_binary else "fallback")

    rows = [
        {"Mutant": name, "TM-score": _score(pdbs[wt_name], pdb, args, use_binary)}
        for name, pdb in pdbs.items()
    ]
    df_wt = pd.DataFrame(rows)
    wt_csv = out_dir / f"tm_scores_{config.name}.csv"
    df_wt.to_csv(wt_csv, index=False)

    # The all-vs-all matrix is O(n^2) TM-align calls. With PTM + proteoform
    # structures included, n can explode; cap the matrix to the canonical
    # WT/mutant set when the full job is large (the WT-vs-all table above
    # still covers every structure).
    if len(pdbs) > 40:
        canonical = dict(iter_structure_pdbs(config, paths, "tmscore"))
        if 1 < len(canonical) < len(pdbs):
            log.info("TM-score all-vs-all matrix restricted to the %d canonical "
                     "WT/mutant structures (full job has %d incl. PTMs/"
                     "proteoforms); the WT-vs-all table covers all structures.",
                     len(canonical), len(pdbs))
            matrix_pdbs = canonical
        else:
            matrix_pdbs = pdbs
    else:
        matrix_pdbs = pdbs

    names = list(matrix_pdbs)
    matrix = np.eye(len(names), dtype=float)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            matrix[i, j] = matrix[j, i] = _score(
                matrix_pdbs[names[i]], matrix_pdbs[names[j]], args, use_binary
            )

    df_all = pd.DataFrame(matrix, index=names, columns=names)
    all_csv = out_dir / f"tm_scores_{config.name}_all.csv"
    df_all.to_csv(all_csv)

    heat = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(max(6, len(names) * 0.7), max(5, len(names) * 0.7)))
        sns.heatmap(
            df_all,
            annot=len(names) <= 12,
            fmt=".4f",
            cmap="RdYlGn_r",
            vmin=float(np.nanmin(matrix)),
            vmax=1.0,
            cbar_kws={"label": "TM-score"},
            annot_kws={"fontsize": 7},
        )
        plt.title(f"Pairwise TM-scores ({config.name})")
        plt.tight_layout()
        heat = out_dir / f"tm_scores_{config.name}_heatmap.svg"
        plt.savefig(heat, format="svg")
        plt.savefig(heat.with_suffix(".png"), format="png", dpi=150)
        plt.close()
    except Exception as exc:
        log.warning("Heatmap generation failed: %s", exc)

    outputs = [str(wt_csv), str(all_csv)]
    if heat is not None:
        outputs.append(str(heat))

    return StepResult(
        "tmscore",
        "ok",
        f"Computed TM-scores for {len(names)} structures "
        f"(engine={'binary' if use_binary else 'fallback'})",
        outputs=outputs,
        data={"wt": df_wt, "all": df_all},
    )