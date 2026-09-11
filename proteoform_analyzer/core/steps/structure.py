"""Step: structure preparation.

Structure source is **Boltz-2** (folds WT + mutant structures from sequence).
The old ``local_pdb`` (crystal fetch + ptmpsi mutate) and ``af3_server`` (manual
AlphaFold-Server JSON) sources have been removed. The actual folding + output
normalization lives in :mod:`boltz2_fold`; this module dispatches to it.

An internal reference-PDB fetch helper (:func:`fetch_reference_pdb` in
``_pdb_utils``) is retained for known-binding-site alignment and optional
validation — it is NOT a user-facing structure source.
"""
from __future__ import annotations

import logging

from ..pipeline import StepResult

log = logging.getLogger("proteoform_analyzer.structure")


def _get_chain_map(config) -> dict:
    """Return ``{uniprot_id: [chain ids]}`` for the reference assembly.

    Priority:
      1. ``config.chain_map`` (explicit; set by all presets) — used verbatim.
      2. Even split derived from ``config.subunit_stoichiometry`` (aligned with
         ``config.uniprot_ids``): sequential chain letters A, B, C, ... are
         handed out per subunit copy-count, matching the ordering used by the
         Boltz-2 folding step (:func:`boltz2_fold._chain_ids`).
      3. Final fallback: every uniprot id maps to ``["A"]``.

    This helper was part of the original ``structure.py`` and is imported by
    :mod:`proteoform`; it was lost when ``structure.py`` became a thin Boltz-2
    dispatcher in v3.0.0 and is restored here (reading the now first-class
    ``config.chain_map`` field) in v3.1.0.
    """
    # 1. User/preset-specified chain map wins.
    explicit = getattr(config, "chain_map", None)
    if explicit:
        return dict(explicit)

    uids = list(getattr(config, "uniprot_ids", []) or [])
    if not uids:
        return {}

    # 2. Even split from stoichiometry (aligned with uniprot_ids).
    stoich = list(getattr(config, "subunit_stoichiometry", []) or [])
    letters = [chr(65 + i) for i in range(sum(stoich) if stoich else len(uids))]
    chain_map: dict[str, list[str]] = {}
    cursor = 0
    for idx, uid in enumerate(uids):
        n_copies = stoich[idx] if idx < len(stoich) else 1
        assigned = letters[cursor:cursor + n_copies]
        cursor += n_copies
        # 3. Guarantee at least one chain per uniprot id.
        chain_map[uid] = assigned or ["A"]
    return chain_map


def prepare_structures(config, paths: dict) -> StepResult:
    """Prepare WT + mutant structures using the configured structure source."""
    src = config.structure_source
    if src == "boltz2":
        from .boltz2_fold import build_structures
        return build_structures(config, paths)

    # Any legacy value falls through to a clear error (local_pdb / af3_server
    # were removed in v3.0.0).
    return StepResult(
        "structure", "skipped",
        f"Unknown/legacy structure_source '{src}'. Only 'boltz2' is supported "
        f"(local_pdb and af3_server were removed in v3.0.0).",
    )
