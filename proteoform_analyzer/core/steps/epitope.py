"""Pluggable epitope-predictor interface for antibody hotspot selection.

Provides a common interface so RFAntibody hotspots can be sourced from AI epitope
predictors as well as user-selected residues:

  - :class:`BCellEpitopePredictor` — B-cell (antibody) epitopes. Wires a
    freely-installable predictor at runtime if one is importable; otherwise falls
    back to a transparent, clearly-labelled propensity heuristic (Parker
    hydrophilicity), never fabricating a "real" tool's output.
  - :class:`MHCIEpitopePredictor` — MHC class I (T-cell) epitopes via
    **MHCflurry 2.0** (pip-installable; ``mhcflurry-downloads fetch
    models_class1_presentation`` once). Slides a 9-mer window, predicts
    presentation, projects to per-residue scores. Falls back cleanly (raises
    ``EpitopePredictorUnavailable``) if MHCflurry/models are absent.
  - :class:`MHCIIEpitopePredictor` — MHC class II (T-cell) epitopes via the
    **IEDB NetMHCIIpan** REST API (needs internet; no local install/license
    file). Falls back cleanly if the API is unreachable.

Design principles:
  - Never fabricate epitopes from a named tool that is not actually installed.
  - Fail gracefully and actionably; the caller decides how to handle unavailability.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("proteoform_analyzer.epitope")


class EpitopePredictorUnavailable(RuntimeError):
    """Raised when a predictor's backing tool is not configured/installed."""


@dataclass
class EpitopeResidue:
    """A predicted epitope residue.

    position: 1-based residue index in the provided sequence
    score:    predictor score (higher = more likely epitope)
    chain:    chain id the residue maps to on the target structure (set by caller)
    """
    position: int
    score: float
    chain: str = "A"

    def as_hotspot(self) -> str:
        """Format as an RFAntibody-style 'chain:resid' token (source numbering)."""
        return f"{self.chain}:{self.position}"


class EpitopePredictor:
    """Base class. Subclasses implement :meth:`predict`."""

    name = "base"
    modality = "generic"

    def predict(self, sequence: str, structure_pdb: str | None = None,
                chain: str = "A", **kwargs) -> list[EpitopeResidue]:
        raise NotImplementedError

    def top_k(self, sequence: str, k: int, structure_pdb: str | None = None,
              chain: str = "A", **kwargs) -> list[EpitopeResidue]:
        residues = self.predict(sequence, structure_pdb=structure_pdb, chain=chain, **kwargs)
        residues.sort(key=lambda r: r.score, reverse=True)
        return residues[:max(1, k)]


# ---------------------------------------------------------------------------
# B-cell predictor
# ---------------------------------------------------------------------------

# Parker hydrophilicity scale (higher = more hydrophilic/surface-exposed);
# used only as a transparent fallback propensity, clearly labelled as heuristic.
_PARKER = {
    "A": 2.1, "R": 4.2, "N": 7.0, "D": 10.0, "C": 1.4, "Q": 6.0, "E": 7.8,
    "G": 5.7, "H": 2.1, "I": -8.0, "L": -9.2, "K": 5.7, "M": -4.2, "F": -9.2,
    "P": 2.1, "S": 6.5, "T": 5.2, "W": -10.0, "Y": -1.9, "V": -3.7,
}


class BCellEpitopePredictor(EpitopePredictor):
    """Predict linear B-cell epitope residues.

    Tries a freely-installable predictor at runtime (hook: ``epitopepredict`` if
    importable). Otherwise uses a windowed Parker-hydrophilicity propensity as a
    transparent fallback (labelled ``method='parker_heuristic'`` in logs) so the
    pipeline can still run end-to-end without a license-restricted tool.
    """

    name = "bcell"
    modality = "B-cell"

    def __init__(self, window: int = 7):
        self.window = window

    def _try_external(self, sequence: str, chain: str):
        """Hook for a real installable B-cell predictor. Returns None if absent."""
        try:
            import importlib
            importlib.import_module("epitopepredict")  # optional dependency
        except Exception:
            return None
        # If present, a real integration would go here. We do not fabricate output;
        # returning None keeps behaviour honest until wired to a specific version.
        log.info("epitopepredict is importable but no version-pinned integration "
                 "is wired; using Parker heuristic fallback.")
        return None

    def predict(self, sequence: str, structure_pdb: str | None = None,
                chain: str = "A", **kwargs) -> list[EpitopeResidue]:
        ext = self._try_external(sequence, chain)
        if ext is not None:
            return ext
        # Parker windowed hydrophilicity (transparent fallback)
        w = self.window
        half = w // 2
        n = len(sequence)
        residues = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window = sequence[lo:hi]
            vals = [_PARKER.get(a, 0.0) for a in window]
            score = sum(vals) / len(vals) if vals else 0.0
            residues.append(EpitopeResidue(position=i + 1, score=score, chain=chain))
        log.info("B-cell epitope prediction via method='parker_heuristic' "
                 "(%d residues scored; NOT a validated predictor)", n)
        return residues


# ---------------------------------------------------------------------------
# MHC predictors (documented stubs)
# ---------------------------------------------------------------------------

_MHC_SETUP_HINT = (
    "MHC epitope prediction requires a license-restricted tool that is not "
    "installed in this environment (e.g. NetMHCpan for MHC-I / NetMHCIIpan for "
    "MHC-II, academic download from DTU Health Tech; or an IEDB API key). "
    "Install/configure it and wire it into {cls}._try_external(), or use "
    "hotspot_source='user' or 'bcell' instead."
)


def _peptide_scores_to_residues(sequence, peptide_scores, k, chain, invert=False):
    """Map per-peptide binder scores onto per-residue epitope scores.

    A k-mer starting at position i covers residues i..i+k-1. Each residue's
    epitope score is the MAX over all k-mers covering it (a residue in any
    strong binder is a strong epitope-residue). Scores are min-max normalized
    to 0..1 for stable ranking across sequences/alleles.

    peptide_scores: list aligned with sliding k-mers over ``sequence``.
    invert: set True when the raw score is "lower = stronger" (e.g. affinity in
            nM); the value is negated before aggregation so higher = stronger.
    """
    import numpy as np
    n = len(sequence)
    per_res = np.full(n, -np.inf)
    for i, s in enumerate(peptide_scores):
        if s is None:
            continue
        val = -float(s) if invert else float(s)
        per_res[i:i + k] = np.maximum(per_res[i:i + k], val)
    # residues never covered (shouldn't happen for n>=k) -> min
    finite = per_res[np.isfinite(per_res)]
    floor = finite.min() if finite.size else 0.0
    per_res[~np.isfinite(per_res)] = floor
    # min-max normalize
    lo, hi = per_res.min(), per_res.max()
    norm = (per_res - lo) / (hi - lo) if hi > lo else np.zeros_like(per_res)
    return [EpitopeResidue(position=i + 1, score=float(norm[i]), chain=chain)
            for i in range(n)]


class _MHCPredictor(EpitopePredictor):
    """Common base for MHC-I/II predictors.

    Subclasses wire a real tool in :meth:`_try_external`; if it is unavailable
    (not installed / no network), :meth:`predict` raises
    :class:`EpitopePredictorUnavailable` with actionable guidance rather than
    fabricating output. The caller (antibody hotspot resolution) can then fall
    back to ``bcell`` or ``user`` sources.
    """

    allele_default = "HLA-A*02:01"

    def __init__(self, allele: str | None = None, peptide_length: int | None = None):
        self.allele = allele or self.allele_default
        self.peptide_length = peptide_length

    def _try_external(self, sequence: str, chain: str):
        return None

    def predict(self, sequence: str, structure_pdb: str | None = None,
                chain: str = "A", **kwargs) -> list[EpitopeResidue]:
        ext = self._try_external(sequence, chain)
        if ext is not None:
            return ext
        raise EpitopePredictorUnavailable(
            _MHC_SETUP_HINT.format(cls=type(self).__name__))


class MHCIEpitopePredictor(_MHCPredictor):
    """MHC class I (T-cell) epitope residues via **MHCflurry 2.0**.

    MHCflurry is pip-installable (``pip install mhcflurry`` + one-time
    ``mhcflurry-downloads fetch models_class1_presentation``). We slide a k-mer
    (default 9) over the sequence, predict each peptide's presentation score,
    and project onto per-residue scores (:func:`_peptide_scores_to_residues`).
    If MHCflurry or its models are absent, returns None -> predict() raises
    a clear EpitopePredictorUnavailable.
    """

    name = "mhc_i"
    modality = "MHC-I"
    allele_default = "HLA-A*02:01"

    def __init__(self, allele: str | None = None, peptide_length: int = 9):
        super().__init__(allele=allele, peptide_length=peptide_length)

    def _try_external(self, sequence: str, chain: str):
        k = self.peptide_length or 9
        if len(sequence) < k:
            return None
        try:
            from mhcflurry import Class1PresentationPredictor
        except Exception as e:
            log.info("MHCflurry not importable (%s); MHC-I predictor unavailable.", e)
            return None
        try:
            predictor = Class1PresentationPredictor.load()
        except Exception as e:
            log.info("MHCflurry models not available (%s). Run "
                     "`mhcflurry-downloads fetch models_class1_presentation`.", e)
            return None
        try:
            peptides = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
            df = predictor.predict(peptides=peptides, alleles=[self.allele],
                                   verbose=0)
            # Align presentation_score back to k-mer order (predict preserves order).
            scores = list(df.sort_values("peptide_num")["presentation_score"]) \
                if "peptide_num" in df.columns else list(df["presentation_score"])
            if len(scores) != len(peptides):
                # fall back to affinity-based (lower nM = stronger) if lengths drift
                scores = list(df["affinity"])
                res = _peptide_scores_to_residues(sequence, scores, k, chain,
                                                  invert=True)
            else:
                res = _peptide_scores_to_residues(sequence, scores, k, chain,
                                                  invert=False)
            log.info("MHC-I epitopes via MHCflurry (allele=%s, %d %d-mers).",
                     self.allele, len(peptides), k)
            return res
        except Exception as e:
            log.warning("MHCflurry prediction failed: %s", e)
            return None


class MHCIIEpitopePredictor(_MHCPredictor):
    """MHC class II (T-cell) epitope residues via **IEDB NetMHCIIpan**.

    Calls the IEDB REST API (NetMHCIIpan 4.3 BA) at
    ``https://tools-cluster-interface.iedb.org/tools_api/mhcii/`` — this needs
    internet access (no local install / no license file required). We submit the
    sequence for the configured allele, parse the returned table's per-peptide
    scores, and project onto per-residue scores. If the network/API is
    unavailable, returns None -> predict() raises EpitopePredictorUnavailable so
    the caller can fall back.
    """

    name = "mhc_ii"
    modality = "MHC-II"
    allele_default = "HLA-DRB1*01:01"
    _IEDB_URL = "https://tools-cluster-interface.iedb.org/tools_api/mhcii/"
    _IEDB_METHOD = "netmhciipan_ba-4.3"

    def __init__(self, allele: str | None = None, peptide_length: int = 15,
                 timeout: int = 120):
        super().__init__(allele=allele, peptide_length=peptide_length)
        self.timeout = timeout

    def _try_external(self, sequence: str, chain: str):
        k = self.peptide_length or 15
        if len(sequence) < k:
            return None
        import urllib.request
        import urllib.parse
        data = urllib.parse.urlencode({
            "method": self._IEDB_METHOD,
            "sequence_text": sequence,
            "allele": self.allele,
            "length": str(k),
        }).encode()
        try:
            req = urllib.request.Request(
                self._IEDB_URL, data=data,
                headers={"User-Agent": "proteoform-analyzer"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                txt = r.read().decode()
        except Exception as e:
            log.info("IEDB NetMHCIIpan API unavailable (%s); MHC-II predictor "
                     "falling back. Needs internet access to tools-cluster-"
                     "interface.iedb.org.", e)
            return None
        return self._parse_iedb(txt, sequence, k, chain)

    @staticmethod
    def _parse_iedb(txt, sequence, k, chain):
        """Parse IEDB tab-delimited MHC-II output into per-residue scores.

        Output columns include 'start', 'end', 'peptide', 'ic50'/'affinity',
        and often a percentile-rank column. We key on start position + ic50
        (lower nM = stronger binder -> invert).
        """
        lines = [l for l in txt.splitlines() if l.strip()]
        if not lines:
            return None
        header = lines[0].split("\t")
        cols = {name.strip().lower(): i for i, name in enumerate(header)}
        # locate start + affinity/ic50 columns robustly
        start_i = next((cols[c] for c in ("start", "seq_num", "pos") if c in cols), None)
        aff_i = next((cols[c] for c in ("ic50", "affinity", "netmhciipan_ba ic50",
                                        "score") if c in cols), None)
        if start_i is None or aff_i is None:
            log.info("IEDB output columns unrecognized (%s); MHC-II fallback.",
                     header)
            return None
        n_kmers = len(sequence) - k + 1
        scores = [None] * n_kmers
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) <= max(start_i, aff_i):
                continue
            try:
                start = int(float(parts[start_i]))  # IEDB start is 1-based
                aff = float(parts[aff_i])
            except Exception:
                continue
            idx = start - 1
            if 0 <= idx < n_kmers:
                # keep strongest (lowest ic50) if multiple rows per start
                if scores[idx] is None or aff < scores[idx]:
                    scores[idx] = aff
        if all(s is None for s in scores):
            return None
        log.info("MHC-II epitopes via IEDB NetMHCIIpan (%d %d-mers scored).",
                 sum(s is not None for s in scores), k)
        return _peptide_scores_to_residues(sequence, scores, k, chain, invert=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_epitope_predictor(source: str, **kwargs) -> EpitopePredictor:
    """Return an epitope predictor for the given hotspot source.

    source: 'bcell' | 'mhc_i' | 'mhc_ii'. ('user' is handled by the caller and
    does not use a predictor.)
    """
    s = (source or "").lower()
    if s == "bcell":
        return BCellEpitopePredictor(**{k: v for k, v in kwargs.items() if k == "window"})
    if s == "mhc_i":
        return MHCIEpitopePredictor(**{k: v for k, v in kwargs.items()
                                       if k in ("allele", "peptide_length")})
    if s == "mhc_ii":
        return MHCIIEpitopePredictor(**{k: v for k, v in kwargs.items()
                                        if k in ("allele", "peptide_length")})
    raise ValueError(f"Unknown epitope source '{source}' (use bcell|mhc_i|mhc_ii)")
