"""Step: molecular dynamics.

Two engines (config.md.engine):
  - "openmm" (default, offline): OpenMM + amber14 + tip3p, short fast simulation.
  - "gromacs" (optional): shells to gmx if available, else skips gracefully.

Outputs RMSD, RMSF, and energy plots via mdtraj + matplotlib.
"""
from __future__ import annotations

import os
import logging
import numpy as np

from ..pipeline import StepResult

log = logging.getLogger("proteoform_analyzer.md")


def _list_pdbs(config, paths):
    # Structures resolve through boltz-experiments first (see _structure_source).
    # Cover canonical WT + mutants AND PTM-modified structures (ptms/ptms/).
    # Proteoform (mutation+PTM combo) structures are excluded: MD on every
    # combination would explode runtime combinatorially.
    from ._structure_source import iter_all_structure_pdbs
    return iter_all_structure_pdbs(config, paths, "md",
                                   include_ptms=True, include_proteoforms=False)



def _add_terminal_oxt(pdb_path):
    """Add OXT atoms to C-terminal residues in a PDB file (needed by OpenMM)."""
    from Bio.PDB import PDBParser, PDBIO
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", pdb_path)
    for model in struct:
        for chain in model:
            residues = [r for r in chain if r.id[0] == " "]
            if not residues:
                continue
            last_res = residues[-1]
            if "OXT" not in last_res:
                # Add OXT atom at the same position as O but offset
                from Bio.PDB import Atom
                o_atom = last_res["O"]
                oxt_coord = o_atom.get_coord().copy()
                # Place OXT slightly offset from O
                oxt_coord[0] += 0.5
                oxt = Atom.Atom("OXT", oxt_coord, 1.0, 1.0, " ", "OXT", 1, "O")
                last_res.add(oxt)
    io = PDBIO()
    io.set_structure(struct)
    io.save(pdb_path)


def _run_openmm_single(name, pdb_path, md_cfg, md_dir):
    """Run a short OpenMM simulation on one structure. Returns dict of outputs."""
    import openmm
    from openmm import app
    import mdtraj
    import simtk.openmm as mm
    from simtk import unit

    out_dir = os.path.join(md_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    # Load PDB and strip non-protein HETATMs (e.g. heme, waters, ions) that
    # the standard forcefield cannot template.  Keep only amino-acid residues.
    from Bio.PDB import PDBParser, PDBIO, Select
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", pdb_path)

    class ProteinOnly(Select):
        def accept_residue(self, residue):
            from Bio.PDB.Polypeptide import is_aa
            return is_aa(residue, standard=True)

    clean_pdb = os.path.join(out_dir, "clean.pdb")
    io = PDBIO()
    io.set_structure(struct)
    io.save(clean_pdb, ProteinOnly())

    # Add terminal OXT atoms to C-terminal residues (needed by OpenMM forcefield)
    _add_terminal_oxt(clean_pdb)

    pdb = app.PDBFile(clean_pdb)
    if getattr(md_cfg, 'implicit_solvent', False):
        # In vacuo (no solvent): fastest mode for pipeline validation
        forcefield = app.ForceField(md_cfg.forcefield)
    else:
        # Explicit solvent: water box + ions (slow but accurate)
        forcefield = app.ForceField(md_cfg.forcefield, md_cfg.water_model)

    # Modeller: add hydrogens, build system
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=7.0)

    if getattr(md_cfg, 'implicit_solvent', False):
        # In vacuo: no solvent, fastest for pipeline validation
        log.info("%s: using in vacuo (no solvent)", name)
        system = forcefield.createSystem(modeller.topology,
                                         nonbondedMethod=app.NoCutoff,
                                         constraints=app.HBonds)
    else:
        # Explicit solvent: add water box + ions
        modeller.addSolvent(forcefield, model="tip3p", padding=1.0 * unit.nanometers,
                            ionicStrength=md_cfg.ionic_strength_molar * unit.molar)
        system = forcefield.createSystem(modeller.topology,
                                         nonbondedMethod=app.PME,
                                         nonbondedCutoff=1.0 * unit.nanometers,
                                         constraints=app.HBonds)
    integrator = mm.LangevinIntegrator(md_cfg.temperature_k * unit.kelvin,
                                       1.0 / unit.picoseconds,
                                       md_cfg.timestep_fs * unit.femtoseconds)
    sim = app.Simulation(modeller.topology, system, integrator)
    sim.reporters.append(app.PDBReporter(os.path.join(out_dir, "trajectory.pdb"), 100))
    sim.reporters.append(app.StateDataReporter(
        os.path.join(out_dir, "log.csv"), 100,
        step=True, time=True, potentialEnergy=True, temperature=True, speed=True))

    # Minimize
    sim.context.setPositions(modeller.positions)
    log.info("%s: minimizing...", name)
    sim.minimizeEnergy(maxIterations=10)
    # NVT-ish: just run production in one go for speed (fast config)
    log.info("%s: running %d steps...", name, md_cfg.production_steps)
    sim.step(md_cfg.production_steps)

    # Save final state
    state = sim.context.getState(getPositions=True, getEnergy=True)
    with open(os.path.join(out_dir, "final.pdb"), "w") as f:
        app.PDBFile.writeFile(sim.topology, state.getPositions(), f)

    # Analysis: RMSD via mdtraj
    traj = mdtraj.load(os.path.join(out_dir, "trajectory.pdb"))
    rmsd = mdtraj.rmsd(traj, traj, 0) * 10.0  # nm -> Angstrom
    time_ps = np.arange(len(rmsd)) * md_cfg.timestep_fs * 100 / 1000.0  # ps

    # RMSF
    rmsf = mdtraj.rmsf(traj, traj, frame=0) * 10.0  # Angstrom

    # Save raw RMSD/RMSF arrays as CSV so the GUI can build interactive Plotly
    # overlays (result-viz #1) without re-loading trajectories each time.
    import pandas as pd
    rmsd_csv = os.path.join(out_dir, "rmsd.csv")
    pd.DataFrame({"time_ps": time_ps, "rmsd_A": rmsd}).to_csv(rmsd_csv, index=False)
    rmsf_csv = os.path.join(out_dir, "rmsf.csv")
    # residue index 1-based for readability
    pd.DataFrame({"residue": np.arange(1, len(rmsf) + 1),
                  "rmsf_A": rmsf}).to_csv(rmsf_csv, index=False)

    # Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"

    plt.figure(figsize=(8, 4))
    plt.plot(time_ps, rmsd, c="black")
    plt.xlabel("time (ps)"); plt.ylabel(r"C$_\alpha$ RMSD ($\AA$)")
    plt.title(f"RMSD - {name}")
    plt.tight_layout()
    rmsd_plot = os.path.join(out_dir, "rmsd.svg")
    plt.savefig(rmsd_plot, format="svg")
    plt.savefig(rmsd_plot.replace(".svg", ".png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(rmsf, c="black")
    plt.xlabel("Residue"); plt.ylabel(r"C$_\alpha$ RMSF ($\AA$)")
    plt.title(f"RMSF - {name}")
    plt.tight_layout()
    rmsf_plot = os.path.join(out_dir, "rmsf.svg")
    plt.savefig(rmsf_plot, format="svg")
    plt.savefig(rmsf_plot.replace(".svg", ".png"), dpi=150)
    plt.close()

    # energy from log.csv
    import pandas as pd
    log_csv = os.path.join(out_dir, "log.csv")
    energy_plot = None
    if os.path.exists(log_csv):
        df = pd.read_csv(log_csv)
        if "Potential Energy (kJ/mole)" in df.columns or "#" in df.columns:
            col = "Potential Energy (kJ/mole)" if "Potential Energy (kJ/mole)" in df.columns else df.columns[3]
            plt.figure(figsize=(8, 4))
            plt.plot(df[col].values, c="black")
            plt.xlabel("step"); plt.ylabel("Potential Energy (kJ/mol)")
            plt.title(f"Energy - {name}")
            plt.tight_layout()
            energy_plot = os.path.join(out_dir, "energy.svg")
            plt.savefig(energy_plot, format="svg")
            plt.savefig(energy_plot.replace(".svg", ".png"), dpi=150)
            plt.close()

    return {
        "rmsd_final": float(rmsd[-1]) if len(rmsd) else None,
        "rmsd_plot": rmsd_plot, "rmsf_plot": rmsf_plot, "energy_plot": energy_plot,
        "rmsd_csv": rmsd_csv, "rmsf_csv": rmsf_csv,
        "trajectory": os.path.join(out_dir, "trajectory.pdb"),
    }


def _run_gromacs_single(name, pdb_path, md_cfg, md_dir):
    """GROMACS path: detect gmx, run if available, else skip."""
    import shutil, subprocess
    if shutil.which("gmx") is None:
        return {"skipped": "gmx not found in PATH"}
    # (Full GROMACS protocol ported from molecular_dynamics_proteinstability.py
    #  would go here; for brevity we delegate to the vendored module if present.)
    log.info("%s: GROMACS path not fully implemented in fast config; skipping", name)
    return {"skipped": "gromacs fast-config path not implemented (use openmm)"}


def run_md(config, paths: dict) -> StepResult:
    """Run MD on WT + mutants with the configured engine."""
    if not config.run_md:
        return StepResult("md", "skipped", "run_md=False")

    md_cfg = config.md
    md_dir = paths["molecular_dynamics"]
    os.makedirs(md_dir, exist_ok=True)

    pdbs = _list_pdbs(config, paths)
    if not pdbs:
        return StepResult("md", "skipped", "No PDBs to simulate")

    engine = md_cfg.engine
    results = []
    outputs = []
    for name, pdb in pdbs:
        log.info("MD (%s) on %s", engine, name)
        try:
            if engine == "openmm":
                r = _run_openmm_single(name, pdb, md_cfg, md_dir)
            elif engine == "gromacs":
                r = _run_gromacs_single(name, pdb, md_cfg, md_dir)
            else:
                return StepResult("md", "skipped", f"Unknown md engine '{engine}'")
            results.append({"name": name, **r})
            for k in ("rmsd_plot", "rmsf_plot", "energy_plot", "trajectory",
                      "rmsd_csv", "rmsf_csv"):
                if r.get(k):
                    outputs.append(r[k])
        except Exception as e:
            log.error("MD failed on %s: %s", name, e)
            results.append({"name": name, "error": str(e)})

    # summary CSV
    import pandas as pd
    rows = [{"name": r.get("name"), "rmsd_final": r.get("rmsd_final"),
             "skipped": r.get("skipped", ""), "error": r.get("error", "")}
            for r in results]
    df = pd.DataFrame(rows)
    csv = os.path.join(md_dir, "md_summary.csv")
    df.to_csv(csv, index=False)
    outputs.append(csv)

    n_ok = sum(1 for r in results if "rmsd_final" in r)
    n_sk = sum(1 for r in results if r.get("skipped"))
    return StepResult(
        "md", "ok" if n_ok else "skipped",
        f"MD ({engine}): {n_ok} ok, {n_sk} skipped, {len(results)} total",
        outputs=outputs, data=df,
    )
