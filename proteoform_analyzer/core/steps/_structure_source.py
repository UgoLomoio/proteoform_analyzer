"""Structure-source resolution: prefer the user's ``boltz-experiments`` directory.

The pipeline's own structures live in ``results/<name>/pdbs/{tetramer,monomer}/``
(canonical PDBs converted from Boltz-2 outputs, or graft fallbacks). When a
``boltz-experiments`` directory is available — a standard ``boltz predict``
output tree::

    boltz-experiments/<job>/predictions/<name>/<name>_model_*.cif

— the structures found THERE take precedence for every downstream tool
(ThermoMPNN/ddG, docking, DiffSBDD ligand design, RFAntibody, PCN, MD,
TM-score, pocket prediction, PTM/proteoform grafting), because they are the
real Boltz-2 predictions rather than the pipeline's converted/grafted copies.

Lookup rules
------------
* Root: ``$BOLTZ_EXPERIMENTS_DIR`` if set, else ``boltz-experiments/`` at the
  project root (next to ``app.py``), else ``./boltz-experiments/`` (CWD).
* Name mapping: canonical PDB stems end in ``-tetramer``/``-monomer`` while
  Boltz job labels use ``-assembly``/``-monomer`` (see
  ``boltz2_fold._job_label``); both spellings are tried.
* Within a predictions dir, ``*_model_0.cif`` is preferred, then sorted
  ``*_model_*.cif``, then any ``.pdb``/``.cif``. Across multiple jobs containing
  the same prediction name, the most recently modified wins.
* mmCIF hits are converted to PDB (via :func:`_pdb_utils.cif_to_pdb`) into a
  per-run cache ``results/<name>/boltz_experiments_pdbs/`` so downstream tools
  always receive plain PDB files.
* Fallback: if a structure is not found in ``boltz-experiments`` (or the
  directory does not exist), the pipeline's own PDB from ``pdbs/`` is used
  silently (a ``log.debug`` note records the source for troubleshooting).

Current-job filtering
---------------------
Results directories are reused across runs, so ``pdbs/`` may contain structures
folded for a *previous* job (different mutation set, different cap, different
subunits). :func:`expected_structure_stems` computes the exact set of canonical
stems the *current* job asks for (WT + the configured/resolved mutations), and
:func:`iter_structure_pdbs` / :func:`iter_ptm_pdbs` /
:func:`iter_proteoform_pdbs` / :func:`iter_all_structure_pdbs` only return
structures belonging to that set. When the job's mutation set cannot be
determined (no sequence-step state, no explicit ``config.mutations``, no
persisted ``<uid>_mutations.txt``), no filtering is applied (backwards
compatible).
"""
from __future__ import annotations

import glob
import logging
import os

log = logging.getLogger("proteoform_analyzer.structure_source")

_ENV_VAR = "BOLTZ_EXPERIMENTS_DIR"
_DIR_NAME = "boltz-experiments"


def _project_root() -> str:
    """App root (the directory containing app.py)."""
    here = os.path.dirname(os.path.abspath(__file__))  # .../core/steps
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def boltz_experiments_root() -> str | None:
    """Return the user's boltz-experiments directory, or None if absent."""
    cands = []
    env = os.environ.get(_ENV_VAR, "").strip()
    if env:
        cands.append(env)
    cands.append(os.path.join(_project_root(), _DIR_NAME))
    cands.append(os.path.join(os.getcwd(), _DIR_NAME))
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return None


def _candidate_names(stem: str) -> list[str]:
    """Prediction names to try for a canonical PDB stem."""
    cands = [stem]
    if stem.endswith("-tetramer"):
        cands.append(stem[: -len("-tetramer")] + "-assembly")
    return cands


def _find_prediction(root: str, stem: str) -> str | None:
    """Find the best structure file for ``stem`` in a boltz predict tree."""
    candidates = _candidate_names(stem)
    hits = []

    for current_root, dirs, files in os.walk(root):
        current_name = os.path.basename(current_root)

        # Match either the exact stem or the -assembly variant
        if current_name not in candidates:
            continue

        # Find all CIF files in this prediction directory
        cif_files = [
            os.path.join(current_root, f)
            for f in files
            if f.lower().endswith(('.cif', '.mmcif'))
        ]

        if not cif_files:
            continue

        # Prefer files with naming patterns:
        # <stem>_predicted.cif, <stem>_predicted_structure.cif,
        # sab_pred_*_predicted.cif, model_0.cif, model.cif
        preferred_patterns = [
            f"{current_name}_predicted.cif",
            f"{current_name}_predicted_structure.cif",
            "model_0.cif",
            "model.cif",
        ]

        preferred = [
            p for p in cif_files
            if os.path.basename(p).lower() in {pat.lower() for pat in preferred_patterns}
            or "sab_pred" in os.path.basename(p).lower()
        ]

        selected = preferred[0] if preferred else sorted(cif_files)[0]
        hits.append(selected)

    if not hits:
        return None

    # Return the most recently modified prediction
    return max(hits, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Current-job structure set
# ---------------------------------------------------------------------------

def _job_mutation_lists(config, paths: dict) -> list[list[str]] | None:
    """Mutation lists (per UniProt id) that define the *current* job.

    Precedence:
      1. ``config._mutation_lists`` — set by the sequence step in this run.
      2. ``config.mutations`` — explicitly user-configured mutations.
      3. ``<results>/<uid>_mutations.txt`` — persisted by the sequence step
         (covers re-runs of individual steps without the sequence step).
    Returns ``None`` when none of these is available (caller must not filter).
    """
    uids = list(getattr(config, "uniprot_ids", []) or [])
    if not uids:
        return None

    runtime = getattr(config, "_mutation_lists", None)
    if runtime:
        return [list(m) for m in runtime]

    configured = getattr(config, "mutations", None) or []
    if configured and any(configured):
        out = []
        for idx, _uid in enumerate(uids):
            muts = list(configured[idx]) if idx < len(configured) and configured[idx] else []
            if "WT" not in [m.upper() for m in muts]:
                muts = muts + ["WT"]
            out.append(muts)
        return out

    results = paths.get("results", "") if paths else ""
    if results and os.path.isdir(results):
        out = []
        for uid in uids:
            mf = os.path.join(results, f"{uid}_mutations.txt")
            if not os.path.exists(mf):
                return None  # incomplete persisted state -> do not filter
            try:
                with open(mf) as fh:
                    muts = [ln.strip() for ln in fh if ln.strip()]
            except Exception:
                return None
            out.append(muts or ["WT"])
        return out

    return None


def expected_structure_stems(config, paths: dict) -> list[str] | None:
    """Canonical PDB stems the *current* job asks for.

    Mirrors the structure step's worklist (:func:`boltz2_fold.build_structures`):
    the WT assembly (folded once from the first subunit) plus
    ``Mut_<uid>_<mutation>-<tag>`` for every configured mutation. ``<tag>`` is
    ``monomer``/``tetramer`` per ``config.is_monomer``.

    Returns ``None`` when the job's mutation set is unknown — callers should
    then fall back to every locally present PDB (no filtering).
    """
    uids = list(getattr(config, "uniprot_ids", []) or [])
    if not uids:
        return None
    mut_lists = _job_mutation_lists(config, paths)
    if mut_lists is None:
        return None

    # Drop mutations that fall in proteolytically cleaved regions: the
    # structure step never folds them, so they must not be expected here.
    try:
        from .sequence import resolve_mature_regions, parse_mutation_pos, is_cleaved_site

        mature_regions = resolve_mature_regions(config, paths)
        if mature_regions:
            filtered: list[list[str]] = []
            for idx, uid in enumerate(uids):
                region = mature_regions.get(uid)
                muts = mut_lists[idx] if idx < len(mut_lists) else []
                kept: list[str] = []
                for mut in muts:
                    pos = parse_mutation_pos(mut)
                    if pos is not None and is_cleaved_site(region, pos):
                        continue
                    kept.append(mut)
                filtered.append(kept)
            mut_lists = filtered
    except Exception:
        pass

    tag = "monomer" if getattr(config, "is_monomer", False) else "tetramer"
    stems: list[str] = []
    # WT is folded once, from the first subunit (see boltz2_fold worklist).
    stems.append(f"wt-{uids[0]}-{tag}")
    for idx, uid in enumerate(uids):
        muts = mut_lists[idx] if idx < len(mut_lists) else []
        for mut in muts:
            mut = str(mut).strip()
            if not mut:
                continue
            if mut.upper() == "WT":
                stem = f"wt-{uid}-{tag}"
            else:
                stem = f"Mut_{uid}_{mut}-{tag}"
            if stem not in stems:
                stems.append(stem)
    return stems


def _ptm_suffix_filters(config) -> list[str] | None:
    """Explicitly configured PTM ``_<ptm_type>_<RESIDUE>`` filename suffixes.

    Returns a list of suffixes when the job configures PTMs explicitly
    (``ptm.pairs`` or the legacy ``residues`` x ``ptm_types`` cross-product),
    or ``None`` in auto-fetch mode (the resolved site set is not known ahead of
    the PTM step, so no suffix filtering is possible).
    """
    ptm_cfg = getattr(config, "ptm", None)
    if ptm_cfg is None:
        return None
    suffixes: list[str] = []
    pairs = getattr(ptm_cfg, "pairs", None) or []
    for triple in pairs:
        try:
            residue, ptm_type = str(triple[0]).strip(), str(triple[1]).strip()
        except Exception:
            continue
        if residue and ptm_type:
            suffixes.append(f"_{ptm_type.lower()}_{residue.upper()}")
    residues = getattr(ptm_cfg, "residues", None) or []
    ptm_types = getattr(ptm_cfg, "ptm_types", None) or []
    for residue in residues:
        for ptm_type in ptm_types:
            suffixes.append(f"_{str(ptm_type).lower().strip()}_{str(residue).upper().strip()}")
    # de-duplicate, preserve order
    seen = set()
    out = []
    for s in suffixes:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out or None


def _matches_ptm_file(stem: str, base_stems: set[str], suffixes: list[str] | None) -> bool:
    """Does a ``ptms/ptms/<stem>.pdb`` file belong to the current job?

    PTM files are named ``<base_stem>_<ptm_type>_<RESIDUE>.pdb`` where
    ``<base_stem>`` is a canonical structure stem (e.g.
    ``Mut_P02766_E7V-tetramer_phosphorylation_SER8``). The file belongs to the
    current job when its base stem is one of the job's canonical stems and —
    when PTMs are explicitly configured — the ``_<ptm_type>_<RESIDUE>`` suffix
    matches a configured pair.
    """
    for base in base_stems:
        if stem == base or stem.startswith(base + "_"):
            if suffixes is None:
                return True
            tail = stem[len(base):]  # "_<ptm_type>_<RESIDUE...>"
            for suf in suffixes:
                if tail.lower().startswith(suf.lower()):
                    return True
    return False


def resolve_structure_pdb(paths: dict, stem: str, step_name: str = "") -> str | None:
    """Resolve a canonical structure stem to a PDB path.

    Prefers the user's ``boltz-experiments`` predictions; falls back to the
    pipeline's own ``pdbs/{tetramer,monomer}/.pdb`` silently (a ``log.debug``
    note records which source each structure came from).
    Returns None when the structure is available in neither place.
    """
    tag = step_name or "pipeline"
    root = boltz_experiments_root()

    if root:
        hit = _find_prediction(root, stem)
        if hit:
            if hit.lower().endswith(".pdb"):
                log.info("[%s] structure '%s' <- boltz-experiments: %s",
                         tag, stem, hit)
                return hit

            # Write to canonical pdbs directory, not a separate cache
            if stem.endswith("-monomer"):
                out_dir = paths["pdbs_monomer"]
            else:
                out_dir = paths["pdbs"]

            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, f"{stem}.pdb")

            try:
                if (not os.path.exists(out) or
                        os.path.getmtime(out) < os.path.getmtime(hit)):
                    from ._pdb_utils import cif_to_pdb
                    cif_to_pdb(hit, out)
                    log.info("[%s] structure '%s' <- boltz-experiments: %s",
                             tag, stem, hit)
                return out
            except Exception as e:
                log.warning("[%s] CIF->PDB failed for boltz-experiments '%s' "
                            "(%s): %s; falling back to pipeline structure",
                            tag, stem, hit, e)
        else:
            # Silent fallback: the pipeline's own PDB is used without
            # user-facing noise (debug only, for troubleshooting).
            log.debug("[%s] structure '%s' not found in boltz-experiments (%s); "
                      "using pipeline structure from pdbs/", tag, stem, root)

    # Fallback to pipeline's own PDBs
    for d_key in ("pdbs", "pdbs_monomer"):
        d = paths.get(d_key, "")
        if d:
            p = os.path.join(d, f"{stem}.pdb")
            if os.path.exists(p):
                return p

    return None


def iter_structure_pdbs(config, paths: dict, step_name: str = "",
                        current_job_only: bool = True) -> list[tuple[str, str]]:
    """``(stem, pdb_path)`` for every canonical structure of this run.

    The worklist (which structures exist) comes from the pipeline's own pdbs
    dir; each path is swapped for the boltz-experiments version when available.

    With ``current_job_only=True`` (default), structures that do not belong to
    the current job's mutation set (stale PDBs left in ``pdbs/`` by previous
    runs with a different configuration) are skipped. When the current job's
    mutation set cannot be determined, every local PDB is returned.
    """
    pdb_dir = paths["pdbs_monomer"] if config.is_monomer else paths["pdbs"]
    out: list[tuple[str, str]] = []
    if not os.path.isdir(pdb_dir):
        return out

    stems_filter: set[str] | None = None
    if current_job_only:
        expected = expected_structure_stems(config, paths)
        if expected is not None:
            stems_filter = set(expected)

    for f in sorted(os.listdir(pdb_dir)):
        if not f.endswith(".pdb"):
            continue
        stem = f[:-4]
        if stems_filter is not None and stem not in stems_filter:
            log.debug("[%s] skipping stale structure '%s' (not part of the "
                      "current job)", step_name or "pipeline", stem)
            continue
        resolved = resolve_structure_pdb(paths, stem, step_name)
        out.append((stem, resolved or os.path.join(pdb_dir, f)))
    return out


def iter_ptm_pdbs(config, paths: dict,
                  current_job_only: bool = True) -> list[tuple[str, str]]:
    """``(stem, pdb_path)`` for every PTM-modified structure of this run.

    PTM structures live in ``results/<name>/ptms/ptms/`` and are named
    ``<base_stem>_<ptm_type>_<RESIDUE>.pdb``. With ``current_job_only=True``
    (default), only PTM structures whose base structure belongs to the current
    job's mutation set are returned (and, when PTMs are explicitly configured,
    only the configured ``ptm_type``/``residue`` combinations).
    """
    ptm_dir = os.path.join(paths.get("ptms", ""), "ptms")
    out: list[tuple[str, str]] = []
    if not ptm_dir or not os.path.isdir(ptm_dir):
        return out

    base_stems: set[str] | None = None
    suffixes: list[str] | None = None
    if current_job_only:
        expected = expected_structure_stems(config, paths)
        if expected is not None:
            base_stems = set(expected)
            suffixes = _ptm_suffix_filters(config)

    for f in sorted(os.listdir(ptm_dir)):
        if not f.endswith(".pdb"):
            continue
        stem = f[:-4]
        if base_stems is not None and not _matches_ptm_file(stem, base_stems, suffixes):
            log.debug("skipping stale PTM structure '%s' (not part of the "
                      "current job)", stem)
            continue
        out.append((stem, os.path.join(ptm_dir, f)))
    return out


def iter_proteoform_pdbs(config, paths: dict,
                         current_job_only: bool = True) -> list[tuple[str, str]]:
    """``(stem, pdb_path)`` for every proteoform (mutation+PTM) structure.

    Proteoforms live in ``results/<name>/proteoforms/`` and are named
    ``Proteoform_<uid>_<mutation>_<ptm_type>_<RESIDUE>.pdb``. With
    ``current_job_only=True`` (default), proteoforms whose mutation is not part
    of the current job (or, for explicitly configured PTMs, whose
    ``ptm_type``/``residue`` is not configured) are skipped.
    """
    pf_dir = paths.get("proteoforms", "")
    out: list[tuple[str, str]] = []
    if not pf_dir or not os.path.isdir(pf_dir):
        return out

    allowed_muts: set[tuple[str, str]] | None = None
    suffixes: list[str] | None = None
    if current_job_only:
        mut_lists = _job_mutation_lists(config, paths)
        if mut_lists is not None:
            uids = list(getattr(config, "uniprot_ids", []) or [])
            allowed_muts = set()
            for idx, uid in enumerate(uids):
                for mut in (mut_lists[idx] if idx < len(mut_lists) else []):
                    mut = str(mut).strip()
                    if mut and mut.upper() != "WT":
                        allowed_muts.add((uid, mut))
            suffixes = _ptm_suffix_filters(config)

    for f in sorted(os.listdir(pf_dir)):
        if not f.endswith(".pdb"):
            continue
        stem = f[:-4]
        if allowed_muts is not None:
            parts = stem.split("_")
            # Proteoform_<uid>_<mut>_<ptm_type>_<RESIDUE>
            if len(parts) >= 5 and parts[0] == "Proteoform":
                uid, mut = parts[1], parts[2]
                if (uid, mut) not in allowed_muts:
                    log.debug("skipping stale proteoform '%s' (mutation not "
                              "part of the current job)", stem)
                    continue
                if suffixes is not None:
                    tail = "_" + "_".join(parts[3:])
                    if not any(tail.lower().startswith(s.lower()) for s in suffixes):
                        log.debug("skipping proteoform '%s' (PTM not configured "
                                  "in the current job)", stem)
                        continue
            # Unparseable names are kept (never over-filter).
        out.append((stem, os.path.join(pf_dir, f)))
    return out


def iter_all_structure_pdbs(config, paths: dict, step_name: str = "",
                            include_ptms: bool = True,
                            include_proteoforms: bool = False,
                            current_job_only: bool = True) -> list[tuple[str, str]]:
    """``(stem, pdb_path)`` for every structure the current job asks for.

    Unions, de-duplicated by stem and in a stable order:
      1. canonical WT + missense-mutant PDBs from ``pdbs/{tetramer,monomer}/``
         (each resolved through ``boltz-experiments`` when available);
      2. PTM-modified PDBs from ``ptms/ptms/`` (when ``include_ptms``);
      3. proteoform (mutation+PTM) PDBs from ``proteoforms/`` (when
         ``include_proteoforms``).

    All three sources are restricted to the current job's mutation/PTM set when
    it can be determined (see :func:`expected_structure_stems`).
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    for stem, pdb in iter_structure_pdbs(config, paths, step_name,
                                         current_job_only=current_job_only):
        if stem not in seen:
            seen.add(stem)
            out.append((stem, pdb))

    if include_ptms:
        for stem, pdb in iter_ptm_pdbs(config, paths,
                                       current_job_only=current_job_only):
            if stem not in seen:
                seen.add(stem)
                out.append((stem, pdb))

    if include_proteoforms:
        for stem, pdb in iter_proteoform_pdbs(config, paths,
                                              current_job_only=current_job_only):
            if stem not in seen:
                seen.add(stem)
                out.append((stem, pdb))

    return out
