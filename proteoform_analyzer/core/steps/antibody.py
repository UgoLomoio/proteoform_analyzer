"""Step: de novo antibody / nanobody design against a binding site (RFAntibody).

Opt-in step (``config.antibody.enabled`` AND ``"antibody"`` in the steps list).
Designs binders to a chosen epitope on a proteoform structure using a **local**
RFAntibody install. The step drives RFAntibody through its **quiver-based CLI**
(the same interface as the upstream RFAntibody examples)::

    rfdiffusion  -t target.pdb -f framework.pdb -q 1_rfdiffusion.qv \\
                 -n <num_designs> -l "H1:7,H2:6,H3:5-13" -h "T305,T456"
    proteinmpnn  -q 1_rfdiffusion.qv --output-quiver 2_proteinmpnn.qv \\
                 -l "H1,H2,H3" -n <seqs_per_struct>
    rf2          -q 2_proteinmpnn.qv --output-quiver 3_rf2.qv -r <recycles>
    qvscorefile  3_rf2.qv            # writes 3_rf2.sc -> copied to scores.tsv
    qvextract    3_rf2.qv -o designs/

Environment resolution (in order):

  1. ``config.antibody.local_python`` if set and runnable.
  2. ``<local_rfantibody_dir>/.venv`` (created by ``uv sync``) when its python
     runs and the console scripts (``rfdiffusion`` etc.) exist.
  3. If ``config.antibody.auto_bootstrap`` is True (default): install ``uv``
     if needed and run ``uv sync`` in the checkout (recreating a broken
     ``.venv``), then download any missing model weights (~750 MB total)
     from the IPD servers. If bootstrapping is disabled or fails, the step
     skips cleanly with actionable instructions.

The console scripts shell out to ``python <script>`` internally, so the venv's
``bin/`` dir is prepended to PATH and ``RFANTIBODY_ROOT`` / ``RFANTIBODY_WEIGHTS``
/ ``RFANTIBODY_SCRIPTS`` are exported for every stage. Each stage logs stdout +
stderr to ``<antibody>/logs/stage<N>_<name>.log``; on failure the log tail is
included in the step message.

HLT target convention (RFAntibody): Heavy='H', Light='L', Target='T', chain order
H,L,T. The target chain(s) we build here are relabelled to 'T'; hotspots are given
in target numbering as ``T<resid>``.
"""
from __future__ import annotations

import os
import sys
import json
import glob
import shutil
import logging
import subprocess

from ..pipeline import StepResult
from ._pdb_utils import fetch_reference_pdb, three_to_one
from .epitope import get_epitope_predictor, EpitopePredictorUnavailable

log = logging.getLogger("proteoform_analyzer.antibody")

# Framework PDBs bundled with an RFAntibody checkout, relative to its root
# (scripts/examples/example_inputs/...). Resolved against local_rfantibody_dir.
_FRAMEWORK_REL_PATHS = {
    "nanobody": os.path.join("scripts", "examples", "example_inputs", "h-NbBCII10.pdb"),
    "scfv": os.path.join("scripts", "examples", "example_inputs", "hu-4D5-8_Fv.pdb"),
}

# Framework-appropriate default CDR loop length ranges for RFdiffusion
# (per the upstream RFAntibody examples). Nanobodies have heavy-chain loops
# only; scFvs add the three light-chain loops.
_DEFAULT_LOOPS = {
    "nanobody": "H1:7,H2:6,H3:5-13",
    "scfv": "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13",
}

# Model weights required by the three stages, with their IPD download URLs.
_WEIGHT_URLS = {
    "RFdiffusion_Ab.pt":
        "https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt",
    "ProteinMPNN_v48_noise_0.2.pt":
        "https://files.ipd.uw.edu/pub/RFantibody/ProteinMPNN_v48_noise_0.2.pt",
    "RF2_ab.pt":
        "https://files.ipd.uw.edu/pub/RFantibody/RF2_ab.pt",
}

# Scripts every runnable checkout must provide under scripts/.
_REQUIRED_SCRIPTS = (
    "rfdiffusion_inference.py",
    "proteinmpnn_interface_design.py",
    "rf2_predict.py",
)

# Console scripts the venv must provide (entry points of the rfantibody pkg).
_REQUIRED_CONSOLE_SCRIPTS = ("rfdiffusion", "proteinmpnn", "rf2",
                             "qvscorefile", "qvextract")

_STAGE_TIMEOUT_S = 4 * 3600  # generous per-stage cap (GPU diffusion is slow)


# ---------------------------------------------------------------------------
# Target structure resolution
# ---------------------------------------------------------------------------

def _resolve_target_pdb(config, paths) -> str | None:
    """Locate the proteoform PDB to design against (WT or a specific mutant).

    ``config.antibody.target_structure``:
      - 'wt'            -> the WT structure
      - 'Mut_<uid>_<mut>' or a mutation token like 'E7V' -> that mutant if present
    Falls back to WT, then to any available structure.

    Structures resolve through boltz-experiments first (see _structure_source):
    the canonical stem is identified in the pipeline's pdbs dir, then swapped
    for the user's Boltz-2 prediction when available.
    """
    from ._structure_source import resolve_structure_pdb
    tag = "monomer" if config.is_monomer else "tetramer"
    d = paths["pdbs_monomer"] if config.is_monomer else paths["pdbs"]
    want = (config.antibody.target_structure or "wt").strip()

    stems = []
    if os.path.isdir(d):
        stems = [f[:-4] for f in sorted(os.listdir(d))
                 if f.endswith(f"-{tag}.pdb")]

    def _wt_stem():
        for s in stems:
            if s.lower().startswith("wt"):
                return s
        return None

    stem = None
    if want.lower() == "wt":
        stem = _wt_stem()
    else:
        # exact "Mut_<uid>_<mut>" label (with or without the -<tag> suffix)
        cand = want if want.endswith(f"-{tag}") else f"{want}-{tag}"
        if cand in stems:
            stem = cand
        else:
            # bare mutation token: match any subunit
            hits = sorted(s for s in stems
                          if s.startswith("Mut_") and s.endswith(f"_{want}-{tag}"))
            if hits:
                stem = hits[0]

    if stem is None:
        wt = _wt_stem()
        if wt is not None:
            log.info("Antibody target '%s' not found; using WT structure.", want)
            stem = wt
    if stem is None and stems:
        stem = stems[0]
    if stem is None:
        return None
    return resolve_structure_pdb(paths, stem, "antibody")


# ---------------------------------------------------------------------------
# HLT target preparation
# ---------------------------------------------------------------------------

def _read_chain_sequence(pdb_path: str, chain_id: str):
    """Return (positions, resnames, one_letter_seq) for a chain (CA atoms)."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("t", pdb_path)
    positions, resnames, seq = [], [], []
    model = next(iter(structure))
    if chain_id not in [c.id for c in model]:
        # default to first chain
        chain = next(iter(model))
    else:
        chain = model[chain_id]
    for res in chain:
        if res.id[0] != " ":
            continue  # skip hetero/water
        if "CA" not in res:
            continue
        positions.append(res.id[1])
        resnames.append(res.resname)
        seq.append(three_to_one(res.resname))
    return positions, resnames, "".join(seq)


def _collapse_altlocs(structure) -> int:
    """Collapse alternate conformations (altlocs) to a single location.

    Experimental PDBs often carry disordered atoms/residues (altloc 'A'/'B').
    Bio.PDB writes *every* altloc as a separate ATOM record, and RFAntibody's
    HLT parser keys residues by their CA line — so duplicated records make a
    residue appear twice and the duplicate entries end up with all-zero
    coordinates. Those null frames crash RFdiffusion's frame propagation
    (``ValueError: Non-positive determinant ... in rotation matrix``; see
    RosettaCommons/RFantibody issues #79/#84). For each disordered atom keep
    the highest-occupancy altloc (ties -> first, i.e. 'A') with a blank altloc;
    disordered *residues* (same resseq, several resnames) keep the child with
    the highest total occupancy. Returns the number of collapsed groups.
    """
    from Bio.PDB.Atom import DisorderedAtom
    from Bio.PDB.Residue import DisorderedResidue

    n_fixed = 0
    for model in structure:
        for chain in model:
            for res_id in list(chain.child_dict.keys()):
                residue = chain.child_dict[res_id]
                if isinstance(residue, DisorderedResidue):
                    children = residue.disordered_get_list()
                    best = max(children, key=lambda r: sum(
                        (a.get_occupancy() or 0.0)
                        for a in r.get_unpacked_list()))
                    chain.detach_child(res_id)
                    chain.add(best)
                    residue = best
                    n_fixed += 1
                n_res = 0
                for atom_id in list(residue.child_dict.keys()):
                    atom = residue.child_dict[atom_id]
                    if isinstance(atom, DisorderedAtom):
                        best = max(atom.disordered_get_list(),
                                   key=lambda a: (a.get_occupancy() or 0.0))
                        residue.detach_child(atom_id)
                        best.set_altloc(" ")
                        best.disordered_flag = 0  # plain Atom again
                        residue.add(best)
                        n_fixed += 1
                        n_res += 1
                if n_res and getattr(residue, "disordered", 0):
                    residue.disordered = 0
    return n_fixed


def _backbone_gaps(pdb_path: str, chain_id: str = "T") -> list[str]:
    """Residue ids (resseq+icode) in ``chain_id`` missing backbone N/CA/C."""
    have: dict[str, set] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            atom = line[12:16].strip()
            if atom in ("N", "CA", "C"):
                have.setdefault(line[22:27], set()).add(atom)
    return sorted(r for r, atoms in have.items() if atoms != {"N", "CA", "C"})


def _write_hlt_target(config, paths, target_pdb: str) -> str:
    """Relabel the target chain(s) to 'T' and write an HLT-style target PDB.

    Only the target chain is kept (cropping to a pocket is left to RFdiffusion,
    which accepts a full or cropped target). Chain id is set to 'T'; residue
    numbering is preserved so hotspot ``T<resid>`` tokens stay valid.
    Alternate conformations are collapsed first: RFAntibody's HLT parser cannot
    handle duplicate ATOM records (they turn into all-zero frames and crash
    RFdiffusion stage 1).
    """
    from Bio.PDB import PDBParser, PDBIO, Select

    out_dir = paths["antibody"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "target.pdb")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("t", target_pdb)
    n_altloc = _collapse_altlocs(structure)
    if n_altloc:
        log.info("Target %s: collapsed %d alternate conformation(s) to "
                 "highest occupancy", os.path.basename(target_pdb), n_altloc)
    model = next(iter(structure))
    keep_chain = config.antibody.target_chain
    chain_ids = [c.id for c in model]
    if keep_chain not in chain_ids:
        keep_chain = chain_ids[0]

    # Relabel kept chain -> 'T'
    for chain in model:
        if chain.id == keep_chain:
            chain.id = "T"

    class _TSel(Select):
        def accept_chain(self, chain):
            return chain.id == "T"
        def accept_residue(self, residue):
            return residue.id[0] == " "

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, select=_TSel())

    # Defensive check: RFdiffusion builds rigid frames from N/CA/C; a residue
    # missing any of them becomes a null frame and crashes stage 1.
    gaps = _backbone_gaps(out_path, "T")
    if gaps:
        log.warning("Target %s: %d residue(s) lack a complete N/CA/C backbone "
                    "(%s%s); RFdiffusion may fail if these lie near the "
                    "epitope", os.path.basename(target_pdb), len(gaps),
                    ", ".join(g.strip() for g in gaps[:5]),
                    ", ..." if len(gaps) > 5 else "")
    return out_path, keep_chain


# ---------------------------------------------------------------------------
# Framework resolution
# ---------------------------------------------------------------------------

def _resolve_framework(config, rfab_dir: str):
    """Return the local framework PDB path to use for docking, or None.

    Resolution order:
      1. ``config.antibody.local_framework_pdb`` if it exists.
      2. ``config.antibody.framework`` if it is itself an existing path.
      3. The framework bundled with the RFAntibody checkout for the chosen class
         (nanobody/scfv), relative to ``rfab_dir``.
    Returns None if no framework file can be located (caller skips cleanly).
    """
    explicit = getattr(config.antibody, "local_framework_pdb", None)
    if explicit and os.path.exists(explicit):
        return explicit

    fw = (config.antibody.framework or "nanobody").strip()
    if os.path.exists(fw):
        return fw

    key = fw.lower()
    if key not in _FRAMEWORK_REL_PATHS:
        log.warning("Unknown framework '%s'; defaulting to nanobody.", fw)
        key = "nanobody"
    cand = os.path.join(rfab_dir, _FRAMEWORK_REL_PATHS[key])
    if os.path.exists(cand):
        return cand
    log.warning("Bundled framework PDB not found at %s", cand)
    return None


def _sanitize_framework(paths, framework_pdb: str, rfab_root: str) -> str:
    """Collapse altlocs in a *user-supplied* framework PDB when needed.

    Frameworks bundled with the RFAntibody checkout are already clean and are
    passed through untouched. A custom framework with alternate conformations
    would crash RFdiffusion stage 1 exactly like an altloc-bearing target, so
    a collapsed copy is written next to the target and used instead.
    """
    if os.path.abspath(framework_pdb).startswith(os.path.abspath(rfab_root)):
        return framework_pdb  # bundled framework: known clean, leave as-is
    from Bio.PDB import PDBParser, PDBIO
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("f", framework_pdb)
        n_altloc = _collapse_altlocs(structure)
    except Exception as e:
        log.warning("Could not check framework %s for altlocs (%s); using it "
                    "as-is", framework_pdb, e)
        return framework_pdb
    if not n_altloc:
        return framework_pdb
    out_path = os.path.join(paths["antibody"], "framework_sanitized.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    log.info("Framework %s: collapsed %d alternate conformation(s); using %s",
             os.path.basename(framework_pdb), n_altloc, out_path)
    return out_path


def _framework_key(config) -> str:
    """Best-guess framework class key ('nanobody'|'scfv') for defaults."""
    fw = (config.antibody.framework or "nanobody").strip()
    if os.path.exists(fw):
        # custom path: guess from the filename, else nanobody
        base = os.path.basename(fw).lower()
        if "fv" in base or "fab" in base or "igg" in base or "mab" in base:
            return "scfv"
        return "nanobody"
    key = fw.lower()
    return key if key in _DEFAULT_LOOPS else "nanobody"


def _resolve_loops(config) -> str:
    """CDR loop length ranges for RFdiffusion (config or framework default)."""
    if getattr(config.antibody, "design_loops", None):
        return config.antibody.design_loops
    return _DEFAULT_LOOPS[_framework_key(config)]


def _loop_names(loops: str) -> str:
    """'H1:7,H2:6,H3:5-13' -> 'H1,H2,H3' (ProteinMPNN -l loop_string)."""
    names = []
    for tok in loops.split(","):
        name = tok.split(":")[0].strip()
        if name and name not in names:
            names.append(name)
    return ",".join(names)


# ---------------------------------------------------------------------------
# Hotspot resolution
# ---------------------------------------------------------------------------

def _resolve_hotspots(config, paths, target_pdb: str, target_src_chain: str) -> list[str]:
    """Resolve hotspot residues as RFAntibody 'T<resid>' tokens.

    Sources:
      - 'user':  config.antibody.hotspot_residues (accepts '305', 'A305', 'T305')
      - 'bcell'/'mhc_i'/'mhc_ii': epitope predictor top-k over the target chain seq
    """
    src = (config.antibody.hotspot_source or "bcell").lower()

    if src == "user":
        toks = []
        for h in config.antibody.hotspot_residues:
            h = str(h).strip()
            if not h:
                continue
            # strip a leading chain letter if present
            resid = "".join(ch for ch in h if ch.isdigit())
            if resid:
                toks.append(f"T{resid}")
        return toks

    # AI epitope prediction over the (source-chain) sequence
    positions, resnames, seq = _read_chain_sequence(target_pdb, "T")
    if not seq:
        log.warning("Empty target sequence; cannot predict epitopes.")
        return []
    try:
        predictor = get_epitope_predictor(src)
    except ValueError as e:
        log.error("%s", e)
        return []
    try:
        top = predictor.top_k(seq, config.antibody.epitope_top_k, chain="T")
    except EpitopePredictorUnavailable as e:
        log.error("Epitope predictor unavailable (%s): %s", src, e)
        return []
    # map sequence index (1-based over CA residues) -> actual residue number
    toks = []
    for r in top:
        idx = r.position - 1
        if 0 <= idx < len(positions):
            toks.append(f"T{positions[idx]}")
    log.info("Predicted %d hotspots via %s: %s", len(toks), src, toks)
    return toks


# ---------------------------------------------------------------------------
# Local RFAntibody environment: venv resolution, bootstrap, weights
# ---------------------------------------------------------------------------

def _python_works(py: str) -> bool:
    """True if the interpreter exists and can execute a trivial command."""
    if not py:
        return False
    if os.sep in py and not os.path.isfile(py):
        return False
    try:
        proc = subprocess.run([py, "-c", "import sys"], timeout=60,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc.returncode == 0
    except Exception:
        return False


def _venv_ready(venv_bin: str) -> bool:
    """True if the venv has a working python and the rfantibody CLI scripts."""
    py = os.path.join(venv_bin, "python")
    if not _python_works(py):
        return False
    return all(os.path.isfile(os.path.join(venv_bin, s))
               for s in _REQUIRED_CONSOLE_SCRIPTS)


def _bootstrap_venv(root: str, venv_dir: str, log_path: str) -> bool:
    """Create/populate the checkout's .venv via ``uv sync``.

    Installs ``uv`` into the current environment if no ``uv`` binary is on
    PATH. A pre-existing but broken .venv (e.g. packaged from another machine,
    whose python symlinks dangle) is removed first so ``uv sync`` recreates it.
    """
    uv = shutil.which("uv")
    if uv:
        uv_cmd = [uv]
    else:
        log.info("uv not found on PATH; installing into the current environment")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "uv"],
                           check=True, timeout=600)
        except Exception as e:
            log.error("Failed to install uv: %s", e)
            return False
        uv_cmd = [sys.executable, "-m", "uv"]

    venv_py = os.path.join(venv_dir, "bin", "python")
    if os.path.isdir(venv_dir) and not _python_works(venv_py):
        log.info("Existing .venv is broken (python not runnable); recreating it")
        try:
            shutil.rmtree(venv_dir)
        except Exception as e:
            log.error("Could not remove broken .venv (%s): %s", venv_dir, e)
            return False

    log.info("Bootstrapping RFAntibody environment: %s sync (cwd=%s) — this "
             "downloads torch/DGL and may take 10-30 min on first run",
             " ".join(uv_cmd), root)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as lf:
        lf.write(f"$ {' '.join(uv_cmd)} sync   # cwd={root}\n\n")
        lf.flush()
        try:
            proc = subprocess.run(uv_cmd + ["sync"], cwd=root, stdout=lf,
                                  stderr=subprocess.STDOUT, timeout=3600)
        except Exception as e:
            lf.write(f"\n[launcher error] {e}\n")
            return False
    if proc.returncode != 0:
        log.error("uv sync failed (exit %s); see %s", proc.returncode, log_path)
        return False
    return True


def _ensure_weights(weights_dir: str, auto_download: bool, log_dir: str):
    """Ensure the three model checkpoints exist; download missing ones.

    Per-file download from the IPD servers (the bundled download_weights.sh
    skips whenever the weights *directory* exists, so it cannot be relied on).
    Returns (ok, note).
    """
    missing = [f for f in _WEIGHT_URLS
               if not os.path.isfile(os.path.join(weights_dir, f))]
    if not missing:
        return True, ""
    if not auto_download:
        return False, f"missing weights {missing} and auto_bootstrap=False"

    import requests
    os.makedirs(weights_dir, exist_ok=True)
    dl_log = os.path.join(log_dir, "bootstrap_weights.log")
    os.makedirs(log_dir, exist_ok=True)
    with open(dl_log, "w") as lf:
        for fname in missing:
            url = _WEIGHT_URLS[fname]
            dest = os.path.join(weights_dir, fname)
            tmp = dest + ".part"
            log.info("Downloading %s from %s (~100-400 MB)...", fname, url)
            lf.write(f"{url} -> {dest}\n")
            lf.flush()
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)
                os.replace(tmp, dest)
            except Exception as e:
                log.error("Weight download failed for %s: %s", fname, e)
                lf.write(f"ERROR: {e}\n")
                if os.path.exists(tmp):
                    os.remove(tmp)
                return False, f"download of {fname} failed (see {dl_log})"
            size_mb = os.path.getsize(dest) / 1e6
            lf.write(f"  ok ({size_mb:.0f} MB)\n")
            log.info("Downloaded %s (%.0f MB)", fname, size_mb)
    return True, ""


def _setup_local_rfantibody(config, log_dir: str):
    """Resolve or bootstrap a runnable local RFAntibody install.

    Returns ``(rfab, note)`` where ``rfab`` is a dict with keys ``root``,
    ``venv_bin``, ``python``, ``weights_dir`` — or None with an explanatory
    note when no runnable install could be set up.
    """
    root = getattr(config.antibody, "local_rfantibody_dir", None)
    if not root or not os.path.isdir(root):
        return None, ("config.antibody.local_rfantibody_dir is unset or not a "
                      "directory")
    root = os.path.abspath(root)

    for s in _REQUIRED_SCRIPTS:
        if not os.path.isfile(os.path.join(root, "scripts", s)):
            return None, f"RFAntibody script missing under {root}/scripts: {s}"

    venv_bin, py, src = _ensure_venv(root, config, log_dir)
    if venv_bin is None:
        return None, f"no runnable RFAntibody python environment ({src})"
    log.info("RFAntibody python: %s (%s)", py, src)

    weights_dir = getattr(config.antibody, "local_weights_dir", None) \
        or os.path.join(root, "weights")
    ok, note = _ensure_weights(weights_dir,
                               getattr(config.antibody, "auto_bootstrap", True),
                               log_dir)
    if not ok:
        return None, note

    return {
        "root": root,
        "venv_bin": venv_bin,
        "python": py,
        "weights_dir": weights_dir,
    }, src


# ---------------------------------------------------------------------------
# Quiver-CLI stage command builders
# ---------------------------------------------------------------------------

def _stage_env(rfab: dict) -> dict:
    """Subprocess environment: venv bin first on PATH + RFANTIBODY_* vars."""
    env = os.environ.copy()
    env["PATH"] = rfab["venv_bin"] + os.pathsep + env.get("PATH", "")
    env["RFANTIBODY_ROOT"] = rfab["root"]
    env["RFANTIBODY_WEIGHTS"] = rfab["weights_dir"]
    env["RFANTIBODY_SCRIPTS"] = os.path.join(rfab["root"], "scripts")
    return env


def _stage1_cmd(
    rfab,
    config,
    target_pdb: str,
    framework_pdb: str,
    hotspots: list[str],
    loops: str,
    quiver_out: str,
) -> list[str]:
    """RFdiffusion: dock the framework and sample CDR-loop geometries."""
    target_pdb = os.path.abspath(target_pdb)
    framework_pdb = os.path.abspath(framework_pdb)
    quiver_out = os.path.abspath(quiver_out)

    log.info("Stage 1: RFdiffusion docking and CDR-loop sampling")
    cmd = [
        os.path.join(rfab["venv_bin"], "rfdiffusion"),
        "-t", target_pdb,
        "-f", framework_pdb,
        "-q", quiver_out,
        "-n", str(int(config.antibody.num_designs)),
        "-l", loops,
        "-w", os.path.join(rfab["weights_dir"], "RFdiffusion_Ab.pt"),
    ]
    if hotspots:
        cmd += ["-h", ",".join(hotspots)]
    return cmd

def _stage2_cmd(rfab, config, qv_in: str, qv_out: str, loops: str) -> list[str]:
    """ProteinMPNN: design CDR-loop sequences for the docked backbones."""
    qv_in = os.path.abspath(qv_in)
    qv_out = os.path.abspath(qv_out)
    log.info("Stage 2: ProteinMPNN sequence design for %s loops", loops)
    return [
        os.path.join(rfab["venv_bin"], "proteinmpnn"),
        "-q", qv_in,
        "--output-quiver", qv_out,
        "-l", _loop_names(loops),
        "-n", str(int(getattr(config.antibody, "mpnn_seqs_per_struct", 2))),
        "-w", os.path.join(rfab["weights_dir"], "ProteinMPNN_v48_noise_0.2.pt"),
    ]

def _stage3_cmd(rfab, config, qv_in: str, qv_out: str) -> list[str]:
    """RF2: predict the designed complexes and score binding confidence."""
    qv_in = os.path.abspath(qv_in)
    qv_out = os.path.abspath(qv_out)
    log.info("Stage 3: RF2 prediction and scoring of designed complexes")
    return [
        os.path.join(rfab["venv_bin"], "rf2"),
        "-q", qv_in,
        "--output-quiver", qv_out,
        "-r", str(int(getattr(config.antibody, "rf2_recycles", 10))),
        "-w", os.path.join(rfab["weights_dir"], "RF2_ab.pt"),
    ]

def _qvscorefile_cmd(rfab, qv_in: str) -> list[str]:
    """Extract the RF2 score table."""
    qv_in = os.path.abspath(qv_in)
    log.info("Extracting RF2 score table from %s", qv_in)
    return [
        os.path.join(rfab["venv_bin"], "qvscorefile"),
        qv_in,
    ]

def _qvextract_cmd(rfab, qv_in: str, out_dir: str) -> list[str]:
    """Extract designed PDB structures from the final Quiver file."""
    qv_in = os.path.abspath(qv_in)
    out_dir = os.path.abspath(out_dir)
    log.info("Extracting PDB structures from %s to %s", qv_in, out_dir)
    return [
        os.path.join(rfab["venv_bin"], "qvextract"),
        qv_in,
        "-o", out_dir,
    ]

def _tail(path: str, n: int = 30) -> str:
    """Last n lines of a log file (for failure messages)."""
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:]).strip()
    except Exception:
        return ""


def _run_stage(cmd: list[str], cwd: str, env: dict, log_path: str,
               stage_name: str, timeout: int = _STAGE_TIMEOUT_S):
    """Run one RFAntibody stage synchronously, capturing all output to a log.

    Returns (ok, detail). On failure ``detail`` holds the log tail so the
    StepResult message shows *why* the stage died instead of failing silently.
    """
    log.info("RFAntibody %s: %s", stage_name, " ".join(cmd))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as lf:
        lf.write("$ " + " ".join(cmd) + "\n\n")
        lf.flush()
        try:
            proc = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                                  stdout=lf, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            lf.write(f"\n[timeout] stage exceeded {timeout}s\n")
            return False, f"timed out after {timeout}s (log: {log_path})"
        except Exception as e:
            lf.write(f"\n[launcher error] {e}\n")
            return False, f"launch error: {e} (log: {log_path})"
    if proc.returncode != 0:
        tail = _tail(log_path)
        hint = ""
        if "Non-positive determinant" in tail:
            hint = ("\n[hint] 'Non-positive determinant' means RFdiffusion "
                    "found a residue with null/duplicate backbone coordinates "
                    "(a degenerate N/CA/C frame) in the target or framework "
                    "PDB. Usual causes: alternate conformations (altlocs) or "
                    "missing backbone atoms. proteoform_analyzer collapses "
                    "altlocs automatically when building target.pdb — check "
                    "the step log for warnings about incomplete backbones, or "
                    "point the step at a cleaned/AlphaFold-DB structure.")
        return False, (f"exit code {proc.returncode} (log: {log_path})\n"
                       f"--- log tail ---\n{tail}{hint}")
    return True, ""

def _ensure_venv(root: str, config, logdir: str):
    """Resolve the RFAntibody interpreter without requiring uv.

    Priority:
      1. config.antibody.local_python, if runnable;
      2. an already-existing runnable <root>/.venv;
      3. uv sync, only when uv is already installed and auto_bootstrap=True;
      4. the current Python environment, if its RFAntibody commands are on PATH.
    """
    venv_dir = os.path.join(root, ".venv")
    venv_bin = os.path.join(venv_dir, "bin")
    explicit = getattr(config.antibody, "local_python", None)

    if explicit:
        if _python_works(explicit):
            explicit_bin = os.path.dirname(os.path.abspath(explicit))
            return explicit_bin, explicit, "config.antibody.local_python"
        log.warning(
            "config.antibody.local_python %s is not runnable; "
            "falling back to the checkout .venv/current environment",
            explicit,
        )

    if _venv_ready(venv_bin):
        return venv_bin, os.path.join(venv_bin, "python"), "existing .venv"

    # Do not install uv. If uv is not already present, use the active env.
    uv = shutil.which("uv")
    if uv and getattr(config.antibody, "auto_bootstrap", True):
        bootlog = os.path.join(logdir, "bootstrap_uvsync.log")
        if _bootstrap_venv(root, venv_dir, bootlog) and _venv_ready(venv_bin):
            return venv_bin, os.path.join(venv_bin, "python"), "bootstrapped .venv"
        log.warning("uv bootstrap failed; falling back to current Python environment")
    else:
        log.info("uv not found or auto_bootstrap=False; using current Python environment directly")

    current_python = os.path.abspath(sys.executable)
    current_bin = os.path.dirname(current_python)
    if _python_works(current_python):
        return current_bin, current_python, "current environment (no uv)"

    return None, None, "current Python environment is not runnable"

# ---------------------------------------------------------------------------
# Antibody–target complex PDBs for the 3D viewer
# ---------------------------------------------------------------------------

_AB_CHAIN_IDS = {"H", "L"}


def _pdb_chain_ids(pdb_path: str) -> set[str]:
    """Return the set of chain IDs present in a PDB file (ATOM/HETATM)."""
    chains = set()
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    chains.add(line[21])
    except OSError:
        pass
    return chains


def _merge_pdb_records(out_path: str, parts: list[tuple[str, set[str]]]) -> None:
    """Concatenate ATOM/HETATM/TER records from ``(path, keep_chains)`` parts.

    ``keep_chains`` filters ATOM/HETATM records by chain ID; an empty set
    keeps every chain. All inputs are assumed to share one coordinate frame
    (true here: target.pdb is derived from the original target PDB without
    any transformation, and the designed antibody chains are placed relative
    to the 'T' chain of target.pdb).
    """
    with open(out_path, "w") as out:
        for path, keep in parts:
            with open(path) as f:
                for line in f:
                    if not line.startswith(("ATOM", "HETATM", "TER")):
                        continue
                    if keep and line.startswith(("ATOM", "HETATM")) \
                            and line[21] not in keep:
                        continue
                    out.write(line)
        out.write("END\n")


def _build_viewer_complexes(paths: dict, target_pdb: str,
                            designs_dir: str) -> list[str]:
    """Write antibody–target complex PDBs for the 3D structure viewer.

    For each extracted design (antibody chains H[/L], optionally the 'T'
    target chain) write into ``antibody/complexes/``:

    * ``<design>_complex.pdb`` — antibody + target chain. When qvextract
      already wrote the target chain (full HLT complex) this is a copy;
      otherwise the design's antibody chains are merged with the 'T'-chain
      ``target.pdb`` used as the RFdiffusion input.
    * ``<design>_fullassembly.pdb`` — antibody chains + the FULL original
      target assembly (e.g. the tetramer), written only when the original
      target has more than one chain and none of its chain IDs collide with
      the antibody chains H/L.
    """
    complexes_dir = os.path.join(paths["antibody"], "complexes")
    os.makedirs(complexes_dir, exist_ok=True)
    hlt_target = os.path.join(paths["antibody"], "target.pdb")

    target_chains = _pdb_chain_ids(target_pdb) if target_pdb else set()
    full_ok = len(target_chains) > 1 and not (target_chains & _AB_CHAIN_IDS)

    written = []
    for design in sorted(glob.glob(os.path.join(designs_dir, "*.pdb"))):
        stem = os.path.splitext(os.path.basename(design))[0]
        dchains = _pdb_chain_ids(design)
        ab_present = dchains & _AB_CHAIN_IDS
        has_target = bool(dchains - _AB_CHAIN_IDS)

        # (a) antibody + target complex
        cplx = os.path.join(complexes_dir, f"{stem}_complex.pdb")
        try:
            if has_target:
                shutil.copy(design, cplx)
            elif ab_present and os.path.isfile(hlt_target):
                _merge_pdb_records(cplx, [(design, ab_present),
                                          (hlt_target, {"T"})])
            else:
                continue
            written.append(cplx)
        except OSError as exc:
            log.warning("Could not write complex for %s: %s", stem, exc)
            continue

        # (b) antibody + full target assembly (e.g. tetramer)
        if full_ok and ab_present:
            full = os.path.join(complexes_dir, f"{stem}_fullassembly.pdb")
            try:
                _merge_pdb_records(full, [(design, ab_present),
                                          (target_pdb, target_chains)])
                written.append(full)
            except OSError as exc:
                log.warning("Could not write full-assembly complex for %s: %s",
                            stem, exc)
    if written:
        log.info("Wrote %d antibody-target complex PDB(s) to %s",
                 len(written), complexes_dir)
    return written


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_antibody(config, paths: dict) -> StepResult:
    """Design antibodies/nanobodies against a binding site via local RFAntibody.

    Runs the quiver-based RFAntibody pipeline (rfdiffusion -> proteinmpnn ->
    rf2 -> qvscorefile/qvextract) synchronously on the local machine using the
    checkout's own ``.venv`` (auto-bootstrapped with ``uv sync`` plus a
    per-file weights download when ``config.antibody.auto_bootstrap`` is True).
    GPU strongly recommended; without a runnable install the step skips
    cleanly with actionable instructions.
    """
    if not getattr(config.antibody, "enabled", False):
        return StepResult("antibody", "skipped",
                          "Antibody design not enabled (config.antibody.enabled=False)")

    os.makedirs(paths["antibody"], exist_ok=True)
    log_dir = os.path.join(paths["antibody"], "logs")
    outputs = []

    target_pdb = _resolve_target_pdb(config, paths)
    if not target_pdb:
        return StepResult("antibody", "skipped",
                          "No target structure available (run 'structure' first)")

    # Build HLT target (chain -> 'T')
    hlt_target, src_chain = _write_hlt_target(config, paths, target_pdb)
    outputs.append(hlt_target)

    # Resolve hotspots
    hotspots = _resolve_hotspots(config, paths, hlt_target, src_chain)
    if not hotspots:
        return StepResult(
            "antibody", "skipped",
            f"No hotspot residues resolved (source='{config.antibody.hotspot_source}'). "
            f"Provide config.antibody.hotspot_residues or use an available epitope source.")

    # Drop hotspots that do not exist in the target numbering — RFdiffusion
    # would otherwise crop around a nonexistent residue or crash opaquely.
    valid_nums = set(_read_chain_sequence(hlt_target, "T")[0])
    if valid_nums:
        bad = [h for h in hotspots
               if not h[1:].isdigit() or int(h[1:]) not in valid_nums]
        if bad:
            log.warning("Hotspot(s) %s not present in target residue numbering "
                        "(%d..%d); dropping them. Check for numbering offsets "
                        "(e.g. UniProt precursor vs mature/PDB numbering).",
                        bad, min(valid_nums), max(valid_nums))
            hotspots = [h for h in hotspots if h not in bad]
        if not hotspots:
            return StepResult(
                "antibody", "skipped",
                f"All resolved hotspots fall outside the target residue "
                f"numbering ({min(valid_nums)}..{max(valid_nums)}). Check for "
                f"a numbering offset between the hotspot source and "
                f"{os.path.basename(target_pdb)}.",
                outputs=outputs)

    # --- Resolve / bootstrap a runnable local RFAntibody install ----------
    rfab, note = _setup_local_rfantibody(config, log_dir)
    if rfab is None:
        return StepResult(
            "antibody", "skipped",
            f"No runnable local RFAntibody install ({note}). Fix: (1) point "
            f"config.antibody.local_rfantibody_dir at a full RFAntibody checkout "
            f"(scripts/ + pyproject.toml), (2) leave auto_bootstrap=True so the "
            f"step runs 'uv sync' and downloads the ~750 MB of model weights on "
            f"first use, or pre-populate .venv + weights/ yourself "
            f"(see https://github.com/RosettaCommons/RFantibody). "
            f"GPU strongly recommended.",
            outputs=outputs)

    # Framework (relative to the resolved RFAntibody checkout)
    framework_pdb = _resolve_framework(config, rfab["root"])
    if framework_pdb is None:
        return StepResult(
            "antibody", "skipped",
            f"No framework PDB found for framework='{config.antibody.framework}'. "
            f"Provide config.antibody.local_framework_pdb, an existing framework path, "
            f"or a valid class ('nanobody'/'scfv') resolvable inside the RFAntibody checkout.",
            outputs=outputs)
    framework_pdb = _sanitize_framework(paths, framework_pdb, rfab["root"])

    loops = _resolve_loops(config)

    # Record design context for provenance
    context = {
        "target_pdb": os.path.basename(target_pdb),
        "target_chain_src": src_chain,
        "framework": config.antibody.framework,
        "framework_pdb": os.path.basename(framework_pdb),
        "hotspots": hotspots,
        "num_designs": int(config.antibody.num_designs),
        "design_loops": loops,
        "mpnn_seqs_per_struct": int(getattr(config.antibody, "mpnn_seqs_per_struct", 2)),
        "rf2_recycles": int(getattr(config.antibody, "rf2_recycles", 10)),
        "rf2_pae_max": float(getattr(config.antibody, "rf2_pae_max", 10.0)),
        "rmsd_max": float(getattr(config.antibody, "rmsd_max", 2.0)),
        "backend": "local-quiver-cli",
        "rfantibody_dir": rfab["root"],
        "weights_dir": rfab["weights_dir"],
    }
    ctx_path = os.path.join(paths["antibody"], "design_context.json")
    with open(ctx_path, "w") as f:
        json.dump(context, f, indent=2)
    outputs.append(ctx_path)

    # --- Quiver-based pipeline (synchronous; stage N feeds stage N+1) -----
    run_root = os.path.join(paths["antibody"], "rfantibody_run")

    # Quiver stores tagged samples. RFantibody refuses to overwrite an existing
    # tag such as samples_design_0, so each pipeline run needs clean stage files.
    if os.path.isdir(run_root):
        shutil.rmtree(run_root)

    os.makedirs(run_root, exist_ok=True)

    qv1 = os.path.join(run_root, "1_rfdiffusion.qv")
    qv2 = os.path.join(run_root, "2_proteinmpnn.qv")
    qv3 = os.path.join(run_root, "3_rf2.qv")
    env = _stage_env(rfab)
    cwd = rfab["root"]

    stages = [
        ("stage 1 (RFdiffusion)",
         _stage1_cmd(rfab, config, hlt_target, framework_pdb, hotspots, loops, qv1),
         os.path.join(log_dir, "stage1_rfdiffusion.log"), qv1),
        ("stage 2 (ProteinMPNN)",
         _stage2_cmd(rfab, config, qv1, qv2, loops),
         os.path.join(log_dir, "stage2_proteinmpnn.log"), qv2),
        ("stage 3 (RF2)",
         _stage3_cmd(rfab, config, qv2, qv3),
         os.path.join(log_dir, "stage3_rf2.log"), qv3),
    ]
    for name, cmd, stage_log, qv_out in stages:
        ok, detail = _run_stage(cmd, cwd, env, stage_log, name)
        if not ok:
            return StepResult("antibody", "failed",
                              f"RFAntibody {name} failed: {detail}",
                              outputs=outputs)
        if not os.path.isfile(qv_out):
            return StepResult(
                "antibody", "failed",
                f"RFAntibody {name} exited cleanly but did not produce "
                f"{os.path.basename(qv_out)} (log: {stage_log})\n"
                f"--- log tail ---\n{_tail(stage_log)}",
                outputs=outputs)
        outputs.append(qv_out)

    # Stage 4: score table (3_rf2.sc -> scores.tsv)
    scores_tsv = os.path.join(paths["antibody"], "scores.tsv")
    ok, detail = _run_stage(_qvscorefile_cmd(rfab, qv3), cwd, env,
                            os.path.join(log_dir, "stage4_qvscorefile.log"),
                            "stage 4 (qvscorefile)")
    sc_file = os.path.splitext(qv3)[0] + ".sc"
    if ok and os.path.isfile(sc_file):
        shutil.copy(sc_file, scores_tsv)
        outputs.append(scores_tsv)
    else:
        log.warning("qvscorefile did not produce %s (%s); continuing without "
                    "a score table", sc_file, detail)

    # Stage 5: extract designed complex PDBs
    designs_dir = os.path.join(paths["antibody"], "designs")

    if os.path.isdir(designs_dir):
        shutil.rmtree(designs_dir)

    os.makedirs(designs_dir, exist_ok=True)
    ok, detail = _run_stage(_qvextract_cmd(rfab, qv3, designs_dir), cwd, env,
                            os.path.join(log_dir, "stage5_qvextract.log"),
                            "stage 5 (qvextract)")
    if not ok:
        return StepResult("antibody", "failed",
                          f"RFAntibody stage 5 (qvextract) failed: {detail}",
                          outputs=outputs)

    n = len(glob.glob(os.path.join(designs_dir, "*.pdb")))
    outputs.append(designs_dir)

    # Build antibody–target complex PDBs for the 3D viewer: a guaranteed
    # target+antibody complex per design, plus a full-assembly complex (e.g.
    # the whole tetramer with the antibody bound) when the target is
    # multi-chain.
    try:
        complexes = _build_viewer_complexes(paths, target_pdb, designs_dir)
        if complexes:
            outputs.append(os.path.join(paths["antibody"], "complexes"))
    except Exception as exc:  # never fail the step over viewer helpers
        log.warning("Could not build antibody-target viewer complexes: %s", exc)

    if n == 0:
        return StepResult(
            "antibody", "skipped",
            "RFAntibody pipeline ran but produced no designed complexes "
            f"(no .pdb extracted from {os.path.basename(qv3)}). Check the stage "
            f"logs under {log_dir}.",
            outputs=outputs, data=context)

    context["n_designs_collected"] = n
    msg = (f"RFAntibody complete (local quiver CLI): collected {n} designed "
           f"complex(es) against {len(hotspots)} hotspot(s). "
           f"Filter by RF2 pAE<{config.antibody.rf2_pae_max}, "
           f"RMSD<{config.antibody.rmsd_max}A (see scores.tsv).")
    log.info(msg)
    return StepResult("antibody", "ok", msg, outputs=outputs, data=context)
