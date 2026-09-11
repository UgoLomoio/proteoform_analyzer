"""FoldX wrapper: PTM ΔΔG via PTM-mimetic substitutions.

FoldX parameterizes only the 20 standard amino acids and cannot read ptmpsi's
modified residues (ptmpsi appends PTM atoms to the *standard* residue name, so
a phospho-Ser is written as SER with extra P/O atoms — FoldX has no energy
terms for those). The established workaround, used here, is to score each PTM
as its accepted **mimetic substitution** with FoldX BuildModel:

  - phosphorylation  SER/THR -> GLU   (phosphomimetic)
  - acetylation      LYS     -> GLN   (acetyl-mimetic)

(see ``core/ptm_rules.py::PTM_MIMETIC_MAP``). PTMs without an accepted mimetic
are skipped with an explicit warning — never silently approximated.

Scope: each PTM is scored on the WT background (mimetic alone) and each
mutation × PTM proteoform is scored as mutation+mimetic combined in one
BuildModel line, mirroring the proteoform step's pairwise enumeration.

FoldX is academic-licensed and cannot be redistributed, so this module only
ever shells out to a user-provided binary (``config.foldx_binary`` or
``$FOLDX_BINARY``). When no binary resolves, every entry point degrades
gracefully (returns no rows) and the rest of the ddG step is unaffected.

FoldX CLI used (FoldX 4/5):
  1. ``foldx --command=RepairPDB --pdb=<pdb>``        (standard pre-processing)
  2. ``foldx --command=BuildModel --pdb=<repaired> --mutant-file=<list>
          --numberOfRuns=<n>``
     mutant-file lines: ``SA15E;`` (single) or ``VA30M,SA15E;`` (combined),
     i.e. <WT-aa1><chain><PDB-resnum><new-aa1>, comma-separated, ';'-terminated.
     Output: ``Average_*.fxout`` (preferred) or ``Dif_*.fxout``, tab-separated,
     with a "total energy" column = ΔΔG in kcal/mol.
"""
from __future__ import annotations

import os
import shutil
import logging
import subprocess

from ..ptm_rules import (parse_residue_spec, ptm_mimetic, one_to_three,
                         three_to_one)

log = logging.getLogger("proteoform_analyzer.foldx")


# ---------------------------------------------------------------------------
# Binary / rotabase resolution
# ---------------------------------------------------------------------------

def resolve_foldx(config):
    """Resolve ``(foldx_binary, rotabase_txt)``; return None when unavailable.

    Resolution order for the binary: ``config.foldx_binary`` → ``$FOLDX_BINARY``
    → ``foldx`` on PATH. For rotabase.txt: ``config.foldx_rotabase`` →
    ``<binary dir>/rotabase.txt`` → None (FoldX is still tried; some builds
    bundle the rotabase).
    """
    binary = (getattr(config, "foldx_binary", None)
              or os.environ.get("FOLDX_BINARY")
              or shutil.which("foldx"))
    if not binary:
        return None
    if not os.path.isfile(binary):
        log.warning("Configured FoldX binary '%s' not found; PTM ddG skipped.",
                    binary)
        return None
    rotabase = getattr(config, "foldx_rotabase", None)
    if not rotabase:
        candidate = os.path.join(os.path.dirname(os.path.abspath(binary)),
                                 "rotabase.txt")
        rotabase = candidate if os.path.isfile(candidate) else None
    elif not os.path.isfile(rotabase):
        log.warning("Configured FoldX rotabase '%s' not found; trying without.",
                    rotabase)
        rotabase = None
    return binary, rotabase


# ---------------------------------------------------------------------------
# Site -> PDB residue number mapping (name-verified)
# ---------------------------------------------------------------------------

def _site_to_pdb_resnum(expected_aa3: str, candidate_positions, chain: str,
                        pdb_path: str):
    """Map a sequence-numbered site to the PDB *author* residue number.

    Uses the name-verified mapping of ``_resnum`` (the same helper the ptm and
    proteoform steps use) and then translates the resolved ordinal to the
    author residue number FoldX needs. Returns ``(author_resnum, aa3)`` or
    None when the site cannot be verified in the structure (never guesses).
    """
    from ._resnum import map_site_to_structure_candidates, parse_pdb_residues

    hit = map_site_to_structure_candidates(
        expected_aa3, candidate_positions, chain, pdb_path=pdb_path)
    if hit is None:
        return None
    _selector, ordinal = hit
    for author, name, ordn in (parse_pdb_residues(pdb_path).get(chain) or []):
        if ordn == ordinal:
            return author, name
    return None


# ---------------------------------------------------------------------------
# FoldX process execution
# ---------------------------------------------------------------------------

def _copy_for_foldx(pdb_path: str, dest_dir: str) -> str:
    """Copy a PDB into dest_dir keeping only ATOM/TER/END records.

    FoldX is strict about its input; HETATM records (ligands, waters) and
    remarks can make RepairPDB/BuildModel fail, so they are stripped here.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(pdb_path))
    with open(pdb_path) as f, open(dest, "w") as out:
        for line in f:
            if line.startswith(("ATOM", "TER", "END")):
                out.write(line)
    return dest


def _run_foldx_cmd(binary: str, args: list[str], cwd: str,
                   rotabase: str | None, timeout: int = 3600) -> bool:
    """Run one FoldX command in ``cwd`` (rotabase.txt linked in if given)."""
    if rotabase:
        dst = os.path.join(cwd, "rotabase.txt")
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(rotabase), dst)
            except OSError:
                shutil.copy(rotabase, dst)
    cmd = [binary] + args
    log.info("FoldX: %s (cwd=%s)", " ".join(cmd), cwd)
    try:
        proc = subprocess.run(cmd, cwd=cwd, timeout=timeout,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    except Exception as e:
        log.warning("FoldX command failed to run: %s", e)
        return False
    if proc.returncode != 0:
        log.warning("FoldX exited with code %s: %s", proc.returncode,
                    " ".join(cmd))
        return False
    return True


def _parse_fxout_ddg(out_dir: str, n_lines: int) -> list[float | None]:
    """Parse per-line ΔΔG (total energy, kcal/mol) from BuildModel output.

    Prefers ``Average_*.fxout`` (one row per mutant-file line, averaged over
    runs) and falls back to ``Dif_*.fxout`` (one row per line per run; rows
    for the same line are averaged). Rows are paired with mutant-file lines
    by order. Returns a list of length ``n_lines`` with None for unparsed
    entries.
    """
    import glob as _glob

    def _read_total_energy(path):
        with open(path) as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        if not lines:
            return []
        header = lines[0].split("\t")
        try:
            ie = next(i for i, h in enumerate(header)
                      if h.strip().lower() == "total energy")
        except StopIteration:
            return []
        vals = []
        for l in lines[1:]:
            fields = l.split("\t")
            if len(fields) <= ie:
                continue
            try:
                vals.append(float(fields[ie]))
            except ValueError:
                vals.append(None)
        return vals

    avg = sorted(_glob.glob(os.path.join(out_dir, "Average_*.fxout")))
    if avg:
        vals = _read_total_energy(avg[0])
        if vals:
            return [(vals[i] if i < len(vals) else None) for i in range(n_lines)]

    dif = sorted(_glob.glob(os.path.join(out_dir, "Dif_*.fxout")))
    if dif:
        vals = _read_total_energy(dif[0])
        if vals:
            # Dif files hold numberOfRuns rows per mutant-file line; average
            # consecutive rows belonging to the same line.
            n_runs = max(round(len(vals) / max(n_lines, 1)), 1)
            out = []
            for i in range(n_lines):
                chunk = [v for v in vals[i * n_runs:(i + 1) * n_runs]
                         if v is not None]
                out.append(sum(chunk) / len(chunk) if chunk else None)
            return out
    return [None] * n_lines


def run_buildmodel(binary: str, rotabase: str | None, pdb_path: str,
                   mutant_lines: list[str], out_dir: str,
                   n_runs: int = 1) -> list[float | None]:
    """Repair ``pdb_path`` and score ``mutant_lines`` with BuildModel.

    Returns one ΔΔG (kcal/mol, positive = destabilizing) per mutant-file line,
    or None for lines that could not be parsed. Returns an all-None list when
    the FoldX run itself failed.
    """
    repair_dir = os.path.join(out_dir, "repair")
    bm_dir = os.path.join(out_dir, "buildmodel")
    os.makedirs(repair_dir, exist_ok=True)
    os.makedirs(bm_dir, exist_ok=True)

    local_pdb = _copy_for_foldx(pdb_path, repair_dir)
    stem = os.path.basename(local_pdb)

    if not _run_foldx_cmd(binary, ["--command=RepairPDB", f"--pdb={stem}"],
                          repair_dir, rotabase):
        log.warning("FoldX RepairPDB failed for %s", pdb_path)
        return [None] * len(mutant_lines)

    repaired = os.path.join(repair_dir, stem.replace(".pdb", "_Repair.pdb"))
    if not os.path.isfile(repaired):
        log.warning("FoldX RepairPDB produced no %s", repaired)
        return [None] * len(mutant_lines)

    bm_pdb = os.path.join(bm_dir, os.path.basename(repaired))
    shutil.copy(repaired, bm_pdb)
    with open(os.path.join(bm_dir, "individual_list.txt"), "w") as f:
        for line in mutant_lines:
            f.write(line.rstrip(";") + ";\n")

    if not _run_foldx_cmd(binary,
                          ["--command=BuildModel", f"--pdb={os.path.basename(bm_pdb)}",
                           "--mutant-file=individual_list.txt",
                           f"--numberOfRuns={int(n_runs)}"],
                          bm_dir, rotabase):
        log.warning("FoldX BuildModel failed for %s", pdb_path)
        return [None] * len(mutant_lines)

    return _parse_fxout_ddg(bm_dir, len(mutant_lines))


# ---------------------------------------------------------------------------
# PTM ΔΔG orchestration
# ---------------------------------------------------------------------------

def run_foldx_ptm_ddg(config, paths: dict, wt_pdb: str,
                      mutation_lists: list[list[str]], mature_regions: dict):
    """Score all configured/observed PTMs (and proteoforms) with FoldX.

    Returns ``(rows, note)`` where ``rows`` is a list of dicts with keys
    ``uniprot_id``, ``mutation``, ``ptm``, ``mimetic_mutation``, ``structure``,
    ``ddg_kcal_mol``, ``method`` and ``note`` is a short human-readable summary
    of what happened (for the step message).
    """
    resolved = resolve_foldx(config)
    if resolved is None:
        note = ("PTM ddG skipped: no FoldX binary configured "
                "(set config.foldx_binary, $FOLDX_BINARY, or --foldx-binary)")
        log.info(note)
        return [], note
    binary, rotabase = resolved

    from .ptm import _resolve_sites, _chains_for_uid
    from .sequence import to_mature_pos, parse_mutation_pos

    sites = _resolve_sites(config, paths)
    if not sites:
        return [], "PTM ddG skipped: no PTM sites resolved"

    # Chains analyzed (mirrors the ThermoMPNN tier): chain A for
    # homo-oligomers, one chain per unique subunit for hetero-oligomers.
    if getattr(config, "is_hetero", False):
        chains = [chr(65 + i) for i in range(config.n_unique_subunits)]
    else:
        chains = ["A"]

    wt_stem = os.path.basename(wt_pdb).replace(".pdb", "")

    # --- Build the FoldX mutant-file lines -------------------------------
    # entries: dict(line, row-meta). One line per PTM (WT background) plus one
    # per mutation × PTM combination (proteoform background).
    entries: list[dict] = []
    n_unmappable = 0
    n_unverified = 0

    for residue, ptm_type, uid in sites:
        parsed = parse_residue_spec(residue)
        if parsed is None:
            continue
        aa3, seq_pos = parsed
        new_aa = ptm_mimetic(aa3, ptm_type)
        if new_aa is None:
            n_unmappable += 1
            log.warning("FoldX PTM ddG: no accepted mimetic substitution for "
                        "%s on %s; site %s skipped", ptm_type, aa3, residue)
            continue
        region = mature_regions.get(uid) if uid else None
        mature_pos = to_mature_pos(region, seq_pos)
        if mature_pos is None:
            log.warning("FoldX PTM ddG: site %s (%s) lies in the cleaved "
                        "region; skipped", residue, uid)
            continue

        # Map the PTM site onto the structure (name-verified), trying the
        # site's own chains first, then the analyzed chains.
        site_chains = _chains_for_uid(config, uid)
        tried = [c for c in site_chains if c in chains] + \
                [c for c in chains if c not in site_chains]
        ptm_map = None  # (chain, author_resnum)
        for chain in tried:
            mapped = _site_to_pdb_resnum(aa3, [mature_pos, seq_pos], chain,
                                         wt_pdb)
            if mapped is not None:
                ptm_map = (chain, mapped[0])
                break
        if ptm_map is None:
            n_unverified += 1
            log.warning("FoldX PTM ddG: site %s could not be verified in %s; "
                        "skipped", residue, os.path.basename(wt_pdb))
            continue
        ptm_chain, ptm_resnum = ptm_map
        wt_aa1 = three_to_one(aa3)
        if wt_aa1 is None:
            continue
        ptm_label = f"{ptm_type}_{residue}"
        mimetic_label = f"{wt_aa1}{seq_pos}{new_aa}"

        # WT-background line: mimetic alone -> WT_<ptm>_<residue> structure.
        line = f"{wt_aa1}{ptm_chain}{ptm_resnum}{new_aa}"
        entries.append({
            "line": line,
            "row": {"uniprot_id": uid or (config.uniprot_ids[0] if config.uniprot_ids else ""),
                    "mutation": "",
                    "ptm": ptm_label,
                    "mimetic_mutation": mimetic_label,
                    "structure": f"{wt_stem}_{ptm_label}",
                    "method": "foldx_mimetic"},
        })

        # Proteoform lines: each mutation (any subunit) x this PTM site.
        for midx, muid in enumerate(config.uniprot_ids):
            muts = mutation_lists[midx] if midx < len(mutation_lists) else []
            for mut in muts:
                if not mut or mut.upper() == "WT":
                    continue
                mpos = parse_mutation_pos(mut)
                if mpos is None:
                    continue
                m_wt1, m_new1 = mut[0].upper(), mut[-1].upper()
                m_aa3 = one_to_three(m_wt1)
                if m_aa3 is None:
                    continue
                m_region = mature_regions.get(muid)
                m_mature = to_mature_pos(m_region, mpos)
                if m_mature is None:
                    continue  # cleaved mutation; already reported by ddg
                m_map = None
                m_chains = _chains_for_uid(config, muid)
                m_tried = [c for c in m_chains if c in chains] + \
                          [c for c in chains if c not in m_chains]
                for chain in m_tried:
                    mapped = _site_to_pdb_resnum(m_aa3, [m_mature, mpos],
                                                 chain, wt_pdb)
                    if mapped is not None:
                        m_map = (chain, mapped[0])
                        break
                if m_map is None:
                    continue
                m_chain, m_resnum = m_map
                if m_chain == ptm_chain and m_resnum == ptm_resnum:
                    log.warning("FoldX PTM ddG: mutation %s and PTM %s target "
                                "the same residue; combined proteoform line "
                                "skipped", mut, residue)
                    continue
                combo = (f"{m_wt1}{m_chain}{m_resnum}{m_new1},"
                         f"{wt_aa1}{ptm_chain}{ptm_resnum}{new_aa}")
                entries.append({
                    "line": combo,
                    "row": {"uniprot_id": muid,
                            "mutation": mut,
                            "ptm": ptm_label,
                            "mimetic_mutation": mimetic_label,
                            "structure": f"Proteoform_{muid}_{mut}_{ptm_label}",
                            "method": "foldx_mimetic"},
                })

    if not entries:
        return [], (f"FoldX PTM ddG: nothing to score "
                    f"({n_unmappable} site(s) without a mimetic, "
                    f"{n_unverified} unverified in the structure)")

    # --- Run FoldX ---------------------------------------------------------
    out_dir = os.path.join(paths.get("ddg", paths.get("results", ".")), "foldx")
    os.makedirs(out_dir, exist_ok=True)
    ddgs = run_buildmodel(binary, rotabase, wt_pdb,
                          [e["line"] for e in entries], out_dir,
                          n_runs=getattr(config, "foldx_n_runs", 1))

    rows = []
    n_ok = 0
    for entry, ddg in zip(entries, ddgs):
        if ddg is None:
            log.warning("FoldX PTM ddG: no value parsed for line '%s'",
                        entry["line"])
            continue
        row = dict(entry["row"])
        row["ddg_kcal_mol"] = round(float(ddg), 4)
        rows.append(row)
        n_ok += 1

    note = (f"FoldX PTM ddG: {n_ok}/{len(entries)} mimetic substitutions "
            f"scored ({len(sites)} PTM site(s); {n_unmappable} without a "
            f"mimetic, {n_unverified} unverified)")
    return rows, note
