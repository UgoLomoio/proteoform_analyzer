"""Command-line interface for the Proteoform Analyzer.

Usage:
    python -m proteoform_analyzer.cli run --fast --protein hemoglobin
    python -m proteoform_analyzer.cli run --fast --protein ttr
    python -m proteoform_analyzer.cli run --fast --protein p53
    python -m proteoform_analyzer.cli run --uniprot P02766 --n-subunits 4 ...
    python -m proteoform_analyzer.cli list-steps
    python -m proteoform_analyzer.cli gui
"""
from __future__ import annotations

import argparse
import json
import sys
import logging
import os

from .core.config import (
    AnalysisConfig, EngineChoice, ProteoformMode, BindingSiteMethod,
    PTMConfig, MDConfig, Boltz2Config, AntibodyConfig, HotspotSource,
    hemoglobin_fast_config, ttr_fast_config, p53_fast_config,
    hemoglobin_standard_config, ttr_standard_config, p53_standard_config,
    hemoglobin_production_config, ttr_production_config, p53_production_config,
    PRESETS,
)
from .core.pipeline import run_analysis, STEP_REGISTRY


def _print_callback(step: str, status: str, message: str):
    flag = {"ok": "[+]", "skipped": "[~]", "failed": "[!]"}.get(status, "[?]")
    print(f"{flag} {step}: {message}", flush=True)


def cmd_list_steps(args):
    print("Available pipeline steps:")
    for name, (desc, _) in STEP_REGISTRY.items():
        print(f"  {name:16s} - {desc}")


def cmd_gui(args):
    from .gui import launch
    launch()


# Use the PRESETS dict from config.py (9 presets)
_PRESETS = PRESETS

# Legacy mapping for backward compat (old --fast --protein style)
_PRESETS_LEGACY = {
    "hemoglobin": hemoglobin_fast_config,
    "ttr": ttr_fast_config,
    "p53": p53_fast_config,
}


def cmd_run(args):
    """Run the pipeline from CLI flags."""
    if args.preset and args.preset in _PRESETS:
        config = _PRESETS[args.preset]()
    elif args.fast and args.protein in _PRESETS_LEGACY:
        config = _PRESETS_LEGACY[args.protein]()
    else:
        uniprot_ids = args.uniprot.split(",") if args.uniprot else []
        mutations = []
        if args.mutations:
            mut_lists = [m.strip().split() for m in args.mutations.split("|")]
            mutations = mut_lists
        # Parse stoichiometry: "2,2" -> [2,2]
        stoich = [int(x) for x in args.stoichiometry.split(",")] if args.stoichiometry else [args.n_subunits]
        # Parse chain map if provided: "P69905:A,C;P68871:B,D"
        chain_map = {}
        if args.chain_map:
            for entry in args.chain_map.split(";"):
                if ":" in entry:
                    uid, chains = entry.split(":", 1)
                    chain_map[uid] = chains.split(",")
        # Parse ligand SDFs
        ligand_sdfs = args.ligand_sdfs.split(",") if args.ligand_sdfs else None

        # Boltz-2 config (structure folding + optional local binary)
        boltz2_cfg = Boltz2Config(
            local_binary=args.boltz2_local_binary,
            prefer_local=bool(args.boltz2_local_binary),
        )

        # Antibody (RFAntibody) config
        hotspots = [h.strip() for h in args.antibody_hotspots.split(",")
                    if h.strip()] if args.antibody_hotspots else []
        antibody_cfg = AntibodyConfig(
            enabled=args.antibody,
            framework=args.antibody_framework,
            hotspot_source=args.antibody_hotspot_source,
            hotspot_residues=hotspots,
            num_designs=args.antibody_num_designs,
            local_rfantibody_dir=args.antibody_rfantibody_dir,
            local_weights_dir=args.antibody_weights_dir,
        )

        # PTM pairs: repeatable --ptm-pair "RESIDUE:PTM[:UNIPROT_ID]"
        # e.g. --ptm-pair CYS10:glutathionylation:P69905
        ptm_pairs = []
        for spec in args.ptm_pair or []:
            parts = [p.strip() for p in spec.split(":")]
            if len(parts) >= 2 and parts[0] and parts[1]:
                ptm_pairs.append((parts[0], parts[1],
                                  parts[2] if len(parts) > 2 and parts[2] else None))
            else:
                print(f"[!] Ignoring malformed --ptm-pair '{spec}' "
                      "(expected RESIDUE:PTM[:UNIPROT_ID])", flush=True)

        # Determine steps; auto-append 'antibody' if enabled and not already listed.
        steps = args.steps.split(",") if args.steps else None
        if args.antibody:
            if steps is None:
                steps = list(AnalysisConfig.default_steps()) + ["antibody"]
            elif "antibody" not in steps:
                steps = steps + ["antibody"]

        config = AnalysisConfig(
            uniprot_ids=uniprot_ids,
            n_subunits=args.n_subunits,
            subunit_stoichiometry=stoich,
            chain_map=chain_map,
            mutations=mutations,
            max_mutations=args.max_mutations,
            proteoform_mode=args.proteoform_mode,
            proteoform_cap=args.proteoform_cap,
            structure_source=args.structure_source,
            local_pdb_id=args.local_pdb_id,
            boltz2=boltz2_cfg,
            antibody=antibody_cfg,
            run_ptm=not args.no_ptm,
            ptm=PTMConfig(pairs=ptm_pairs),
            run_md=not args.no_md,
            md=MDConfig(engine=args.md_engine, production_steps=args.md_steps),
            docking_engine=args.docking_engine,
            ligand_design_engine=args.ligand_design_engine,
            binding_site_method=args.binding_site_method,
            ligand_sdfs=ligand_sdfs,
            ensemble_docking=args.ensemble_docking,
            ensemble_n_snapshots=args.ensemble_n_snapshots,
            thermompnn_dir=args.thermompnn_dir,
            thermompnn_checkpoint=args.thermompnn_checkpoint,
            steps=steps,
            work_dir=args.work_dir,
        )

    # FoldX flags apply to both preset and custom configs (presets never set
    # a FoldX binary; the CLI flags are the only way to point at one).
    if getattr(args, "foldx_binary", None):
        config.foldx_binary = args.foldx_binary
    if getattr(args, "foldx_rotabase", None):
        config.foldx_rotabase = args.foldx_rotabase
    if getattr(args, "foldx_n_runs", None):
        config.foldx_n_runs = args.foldx_n_runs

    # Mature-chain flags apply to both preset and custom configs (manual
    # regions override the preset's / auto-detected ones).
    if getattr(args, "no_mature_auto", False):
        config.mature_auto_detect = False
    if getattr(args, "mature_regions", None):
        from .core.config import parse_mature_regions
        if args.mature_regions.strip().lower() in ("none", "full", "full-length"):
            # Explicit opt-out: clear any preset-provided manual regions so
            # full-length precursor sequences are used (combine with
            # --no-mature-auto to also disable auto-detection).
            config.mature_regions = {}
            print("[i] --mature-regions none: manual mature regions cleared; "
                  "full-length precursor sequences will be used.", flush=True)
        else:
            manual_regions = parse_mature_regions(args.mature_regions,
                                                  config.uniprot_ids)
            if manual_regions:
                config.mature_regions = manual_regions
            else:
                print("[!] --mature-regions could not be parsed; ignored "
                      "(expected 'P02766:21-147;P69905:2-142')", flush=True)

    config.progress_callback = _print_callback
    print(f"=== Proteoform Analyzer: {config.name} ===", flush=True)
    print(f"Subunits: {config.n_subunits} | Stoichiometry: {config.subunit_stoichiometry} | "
          f"{'hetero' if config.is_hetero else 'homo'}", flush=True)
    print(f"UniProt IDs: {config.uniprot_ids}", flush=True)
    print(f"Proteoform mode: {config.proteoform_mode} | Binding site: {config.binding_site_method}", flush=True)
    print(f"Engines: MD={config.md.engine} | docking={config.docking_engine} | "
          f"ligand_design={config.ligand_design_engine}", flush=True)
    print(f"Steps: {config.steps}", flush=True)
    print(flush=True)

    results = run_analysis(config)
    print("\n=== Summary ===", flush=True)
    for r in results:
        flag = {"ok": "OK", "skipped": "SKIP", "failed": "FAIL"}.get(r.status, "?")
        print(f"  [{flag}] {r.step:16s} {r.message}", flush=True)
    status_csv = os.path.join(config.results_dir(), "step_status.csv")
    import pandas as pd
    pd.DataFrame([{"step": r.step, "status": r.status, "message": r.message,
                   "elapsed_s": r.elapsed_s, "n_outputs": len(r.outputs)}
                  for r in results]).to_csv(status_csv, index=False)
    print(f"\nStep-status table saved to {status_csv}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(
        prog="proteoform_analyzer",
        description="Analyze mutations, PTMs, and proteoforms in oligomeric proteins.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # run
    pr = sub.add_parser("run", help="Run the pipeline")
    pr.add_argument("--preset", default=None,
                    choices=list(PRESETS.keys()),
                    help="Preset configuration (e.g. 'Hemoglobin — Fast')")
    pr.add_argument("--fast", action="store_true", help="Use fast preset (legacy)")
    pr.add_argument("--protein", default="hemoglobin",
                    choices=["hemoglobin", "ttr", "p53"],
                    help="Preset protein (legacy, with --fast)")
    pr.add_argument("--uniprot", help="UniProt ID(s), comma-separated")
    pr.add_argument("--mutations", help="Mutations per subunit, pipe-separated (e.g. 'E7V D95H|D75A')")
    pr.add_argument("--n-subunits", type=int, default=4, help="Number of subunits (1=monomer, 4=tetramer)")
    pr.add_argument("--stoichiometry", default=None, help="Comma-separated copy counts (e.g. '2,2')")
    pr.add_argument("--chain-map", default=None, help="Chain map: 'P69905:A,C;P68871:B,D'")
    pr.add_argument("--max-mutations", type=int, default=10)
    pr.add_argument("--proteoform-mode", default="pairwise", choices=["single", "pairwise"])
    pr.add_argument("--proteoform-cap", type=int, default=100)
    pr.add_argument("--structure-source", default="boltz2", choices=["boltz2"],
                    help="Structure source (Boltz-2 folding via the Boltz API, "
                         "a local binary via --boltz2-local-binary, or a "
                         "backbone-identical graft fallback)")
    pr.add_argument("--boltz2-local-binary", default=None,
                    help="Path to a local 'boltz' binary; if set, fold locally "
                         "(also used for Boltz-2 docking)")
    pr.add_argument("--local-pdb-id", default=None,
                    help="Reference PDB ID for binding-site alignment ONLY "
                         "(not a structure source)")
    pr.add_argument("--md-engine", default="openmm", choices=["openmm", "gromacs"])
    pr.add_argument("--md-steps", type=int, default=5000)
    pr.add_argument("--docking-engine", default="vina", choices=["vina", "boltz2", "diffdock"])
    pr.add_argument("--ligand-design-engine", default="diffsbdd", choices=["diffsbdd", "boltzgen"])
    pr.add_argument("--binding-site-method", default="auto",
                    choices=["auto", "reference", "p2rank", "alphasphere", "user"])
    pr.add_argument("--ligand-sdfs", default=None, help="Comma-separated SDF paths for multi-ligand screening")
    pr.add_argument("--ensemble-docking", action="store_true", help="Dock against MD snapshots")
    pr.add_argument("--ensemble-n-snapshots", type=int, default=3)
    # --- Antibody design (RFAntibody; opt-in) ---
    pr.add_argument("--antibody", action="store_true",
                    help="Enable de novo antibody/nanobody design (RFAntibody). "
                         "Adds the 'antibody' step automatically.")
    pr.add_argument("--antibody-framework", default="nanobody",
                    choices=["nanobody", "scfv"],
                    help="Antibody framework class; resolved from the local "
                         "RFAntibody checkout's example inputs")
    pr.add_argument("--antibody-hotspots", default=None,
                    help="Comma-separated target hotspot residues for hotspot-source "
                         "'user' (e.g. '305,456' or 'A305,A456')")
    pr.add_argument("--antibody-hotspot-source", default="bcell",
                    choices=["user", "mhc_i", "mhc_ii", "bcell"],
                    help="How to choose epitope hotspots: AI prediction "
                         "(bcell, default; mhc_i/mhc_ii are stubs) or "
                         "user-provided residues via --antibody-hotspots")
    pr.add_argument("--antibody-num-designs", type=int, default=20,
                    help="Number of RFdiffusion designs to generate")
    pr.add_argument("--antibody-rfantibody-dir", default=None,
                    help="Path to a local RFAntibody checkout (scripts/ + weights/). "
                         "Required to run antibody design; otherwise the step skips.")
    pr.add_argument("--antibody-weights-dir", default=None,
                    help="Dir with the RFdiffusion_Ab.pt checkpoint "
                         "(defaults to <rfantibody-dir>/weights)")
    # --- ddG / ThermoMPNN (local; optional, falls back to ESM2 zero-shot) ---
    pr.add_argument("--thermompnn-dir", default=None,
                    help="Path to a local ThermoMPNN checkout (expects "
                         "custom_inference.py under it). If unset, ddG Tier-1 "
                         "falls back to the ESM2 zero-shot CPU scorer.")
    pr.add_argument("--thermompnn-checkpoint", default=None,
                    help="Path to a ThermoMPNN model checkpoint (required to run "
                         "ThermoMPNN locally; otherwise ESM2 zero-shot is used)")
    # --- ddG / FoldX (local; optional, PTM stability via mimetic substitutions) ---
    pr.add_argument("--foldx-binary", default=None,
                    help="Path to a local FoldX executable (FoldX 4/5; academic "
                         "license, not redistributed). Enables PTM ddG via "
                         "PTM-mimetic substitutions (e.g. phosphomimetic "
                         "SER/THR->GLU, acetyl-mimetic LYS->GLN). Also read from "
                         "$FOLDX_BINARY. If unset, PTM ddG is skipped.")
    pr.add_argument("--foldx-rotabase", default=None,
                    help="Path to the FoldX rotabase.txt (default: looked up "
                         "next to the FoldX binary).")
    pr.add_argument("--foldx-n-runs", type=int, default=1,
                    help="FoldX BuildModel numberOfRuns (replicates averaged "
                         "per mutation; default 1).")
    pr.add_argument("--ptm-pair", action="append", default=None,
                    metavar="RESIDUE:PTM[:UNIPROT_ID]",
                    help="Add an explicit PTM site, e.g. 'CYS10:glutathionylation:P69905'. "
                         "Repeatable. Only chemically valid residue/PTM combinations "
                         "are accepted. If none are given, observed PTMs are "
                         "auto-fetched from UniProt (capped at --max-mutations); "
                         "if none exist, the PTM step is skipped.")
    pr.add_argument("--no-ptm", action="store_true")
    pr.add_argument("--no-md", action="store_true")
    # --- Mature chain (proteolytic processing) ---
    pr.add_argument("--mature-regions", default=None,
                    metavar="UID:START-END[;UID:START-END...]",
                    help="Manual mature-chain regions in UniProt (precursor) "
                         "numbering, 1-based inclusive, e.g. "
                         "'P02766:21-147;P69905:2-142'. Overrides auto-detection "
                         "for the listed IDs. A bare 'START-END' is allowed with "
                         "a single --uniprot ID. Pass 'none' (or 'full') to clear "
                         "preset-provided regions and use full-length precursors.")
    pr.add_argument("--no-mature-auto", action="store_true",
                    help="Disable UniProt auto-detection of mature-chain regions "
                         "(use full-length precursor sequences unless "
                         "--mature-regions is given).")
    pr.add_argument("--steps", default=None, help="Comma-separated step names (default: all)")
    pr.add_argument("--work-dir", default="results")
    pr.set_defaults(func=cmd_run)

    pls = sub.add_parser("list-steps", help="List available pipeline steps")
    pls.set_defaults(func=cmd_list_steps)

    pg = sub.add_parser("gui", help="Launch the Gradio web GUI")
    pg.set_defaults(func=cmd_gui)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
