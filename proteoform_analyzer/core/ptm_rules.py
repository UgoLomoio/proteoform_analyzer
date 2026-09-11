"""Residue <-> PTM compatibility rules: the single source of truth.

Derived directly from the vendored ptmpsi implementation so the GUI, the CLI
and the pipeline all agree on which post-translational modifications are
*theoretically possible* for each amino acid:

- ``ptmpsi/residues/ptms.py::check_ptm`` — radical-attachment PTMs
  (phosphorylation, acetylation, methylations) and their allowed residues.
- ``ptmpsi/protein/mutate.py::_CYSPTMS`` + ``ptmpsi/residues::ptm2nonstandard``
  — Cys-only PTMs modelled as non-standard residue replacement
  (glutathionylation, nitrosylation, sulfenylation, ...).
- ``ptmpsi/protein/mutate.py::post_translational_modification`` — the
  N-terminal alpha-acetylation special case.

Also provides :func:`map_uniprot_ptm`, which maps UniProt feature descriptions
(e.g. "Phosphoserine", "N6-acetyllysine", "S-nitroso-Cys") onto ptmpsi PTM
names so *experimentally observed* PTMs can be fetched from UniProt and
filtered to the ones ptmpsi can actually model.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Compatibility matrix (standard 3-letter residue names)
# ---------------------------------------------------------------------------

# Cys-only PTMs (ptmpsi models them by replacing CYS with a non-standard
# residue: CGL, SNC, CSS, CSO, CSD, OCS, SMC, QCS, XCN, IYY).
CYS_ONLY_PTMS: tuple[str, ...] = (
    "carbamoylation",
    "sulfhydration",
    "sulfenylation",
    "sulfinylation",
    "sulfonylation",
    "nitrosylation",
    "glutathionylation",
    "cysteinylation",
    "cyanylation",
)

# Radical-attachment PTMs and the residues ptmpsi's check_ptm() accepts.
# (Protonation-state variants LYN/ASH/CYM/HID/HIE/HIP are collapsed onto their
# standard residue names here.)
_RADICAL_PTMS: dict[str, frozenset[str]] = {
    "phosphorylation": frozenset({"SER", "THR", "TYR", "ARG", "HIS", "LYS", "ASP", "CYS"}),
    "acetylation": frozenset({"LYS"}),
    "methylation": frozenset({"GLU", "LYS", "ARG", "HIS"}),
    "dimethylation": frozenset({"LYS", "ARG"}),
    "trimethylation": frozenset({"LYS"}),
    "symmetric dimethylation": frozenset({"ARG"}),
    "asymmetric dimethylation": frozenset({"ARG"}),
}

# Special case handled by ptmpsi: acetylation of the N-terminal alpha-amino
# group (any residue, but only at the chain N-terminus).
NTERMINAL_PTM = "alpha-acetylation"

# Every PTM name ptmpsi can model (used to validate GUI/CLI input).
ALL_PTMS: tuple[str, ...] = tuple(
    dict.fromkeys(list(CYS_ONLY_PTMS) + list(_RADICAL_PTMS) + [NTERMINAL_PTM])
)

# Residues that can carry at least one PTM (used to populate the GUI residue
# dropdown; every other position is omitted because no PTM could ever apply).
_MODIFIABLE_RESIDUES: frozenset[str] = frozenset(
    {"CYS"} | set().union(*_RADICAL_PTMS.values())
)


def compatible_ptms(aa3: str, n_terminal: bool = False) -> list[str]:
    """Return the sorted list of ptmpsi PTM names compatible with a residue.

    ``aa3``: 3-letter residue name (e.g. "CYS"). Case-insensitive.
    ``n_terminal``: True if the residue is the first of its chain (enables
    alpha-acetylation).
    """
    aa = (aa3 or "").upper().strip()
    out: set[str] = set()
    if aa == "CYS":
        out.update(CYS_ONLY_PTMS)
    for ptm, residues in _RADICAL_PTMS.items():
        if aa in residues:
            out.add(ptm)
    if n_terminal and aa:
        out.add(NTERMINAL_PTM)
    return sorted(out)


def ptm_allowed(aa3: str, ptm: str, n_terminal: bool = False) -> bool:
    """True if ``ptm`` can be applied to residue ``aa3`` by ptmpsi."""
    return (ptm or "").lower().strip() in compatible_ptms(aa3, n_terminal=n_terminal)


def is_modifiable_residue(aa3: str) -> bool:
    """True if the residue type can carry at least one modelable PTM."""
    return (aa3 or "").upper().strip() in _MODIFIABLE_RESIDUES


# ---------------------------------------------------------------------------
# PTM -> mimetic substitution map (FoldX ddG)
# ---------------------------------------------------------------------------

# FoldX parameterizes only the 20 standard amino acids and cannot read
# ptmpsi's modified residues, so PTM stability effects are approximated with
# the accepted *mimetic* substitutions used in the literature:
#
#   phosphorylation  SER/THR -> GLU   (phosphomimetic; adds the negative charge)
#   acetylation      LYS     -> GLN   (acetyl-mimetic; neutralizes the charge)
#
# PTMs absent from this map have no accepted substitution mimetic (Tyr/His/Arg/
# Lys/Asp/Cys phosphorylation, all methylations, Cys redox PTMs, N-terminal
# alpha-acetylation) and are skipped by the FoldX tier with an explicit warning
# rather than being silently approximated.
PTM_MIMETIC_MAP: dict[tuple[str, str], str] = {
    ("SER", "phosphorylation"): "E",
    ("THR", "phosphorylation"): "E",
    ("LYS", "acetylation"): "Q",
}


def ptm_mimetic(aa3: str, ptm: str) -> Optional[str]:
    """Return the 1-letter mimetic substitution for a residue/PTM pair.

    ``aa3``: 3-letter residue name (e.g. "SER"). ``ptm``: ptmpsi PTM name
    (e.g. "phosphorylation"). Returns the 1-letter code of the mimetic
    replacement (e.g. "E"), or None when no accepted mimetic exists.
    """
    return PTM_MIMETIC_MAP.get(((aa3 or "").upper().strip(),
                                (ptm or "").lower().strip()))


# ---------------------------------------------------------------------------
# UniProt feature-description -> ptmpsi PTM mapping
# ---------------------------------------------------------------------------

# UniProt feature types that describe PTMs (others, e.g. "Disulfide bond",
# are not PTMs ptmpsi can graft and are ignored).
UNIPROT_PTM_FEATURE_TYPES: tuple[str, ...] = (
    "Modified residue",
    "Cross-link",
    "Lipidation",
    "Glycosylation",
)

# Ordered (pattern, ptm_name) rules, matched case-insensitively against the
# UniProt feature description. Order matters: more specific patterns first
# (e.g. "trimethyl" before "dimethyl" before "methyl"). Patterns that map to
# PTMs ptmpsi cannot model (glycosylation, myristoylation, palmitoylation,
# hydroxylation, citrullination, nitration, prenylation, ...) are deliberately
# absent -> the entry is dropped.
_UNIPROT_DESC_RULES: tuple[tuple[str, str], ...] = (
    (r"trimethyl", "trimethylation"),
    # \b matters: "asymmetric" contains "symmetric" as a substring
    (r"\basymmetric\s+dimethyl", "asymmetric dimethylation"),
    (r"\bsymmetric\s+dimethyl", "symmetric dimethylation"),
    (r"dimethyl", "dimethylation"),
    (r"methyl", "methylation"),
    (r"phospho", "phosphorylation"),
    (r"n6-acetyl|n-?alpha-acetyl|acetyl", "acetylation"),
    (r"nitroso", "nitrosylation"),
    (r"glutathion", "glutathionylation"),
    (r"cysteinyl", "cysteinylation"),
    (r"sulfen|s-hydroxy", "sulfenylation"),
    (r"sulfin", "sulfinylation"),
    (r"sulfon|cysteic", "sulfonylation"),
    (r"sulfhydr|persulfid", "sulfhydration"),
    (r"carbamoyl", "carbamoylation"),
    (r"cyanyl", "cyanylation"),
)


def map_uniprot_ptm(description: str) -> Optional[str]:
    """Map a UniProt PTM feature description to a ptmpsi PTM name.

    Returns None when the described modification is not modelable by ptmpsi
    (e.g. glycosylation, lipidation) or is unrecognized.
    """
    desc = (description or "").strip().lower()
    if not desc:
        return None
    for pattern, ptm in _UNIPROT_DESC_RULES:
        if re.search(pattern, desc):
            return ptm
    return None


# ---------------------------------------------------------------------------
# Residue-spec parsing helpers (shared by GUI, CLI and the ptm step)
# ---------------------------------------------------------------------------

_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "E": "GLU",
    "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}


def one_to_three(aa1: str) -> Optional[str]:
    """1-letter -> 3-letter residue name (None if unknown)."""
    return _ONE_TO_THREE.get((aa1 or "").upper())


_THREE_TO_ONE = {v: k for k, v in _ONE_TO_THREE.items()}


def three_to_one(aa3: str) -> Optional[str]:
    """3-letter -> 1-letter residue code (None if unknown)."""
    return _THREE_TO_ONE.get((aa3 or "").upper().strip())


def parse_residue_spec(spec: str) -> Optional[tuple[str, int]]:
    """Parse a residue spec like 'CYS10' or 'C10' into ('CYS', 10).

    Returns None when the spec cannot be parsed.
    """
    s = (spec or "").strip().upper()
    m = re.match(r"^([A-Z]{3})(\d+)$", s)
    if m:
        return m.group(1), int(m.group(2))
    m = re.match(r"^([A-Z])(\d+)$", s)
    if m:
        aa3 = one_to_three(m.group(1))
        if aa3:
            return aa3, int(m.group(2))
    return None
