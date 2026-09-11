"""Core engine: config, pipeline orchestration, and shared utilities."""
from .config import (
    AnalysisConfig, EngineChoice, ProteoformMode, BindingSiteMethod,
    PTMConfig, MDConfig,
    hemoglobin_fast_config, ttr_fast_config, p53_fast_config,
)
from .pipeline import run_analysis, StepResult, STEP_REGISTRY

__all__ = [
    "AnalysisConfig", "EngineChoice", "ProteoformMode", "BindingSiteMethod",
    "PTMConfig", "MDConfig",
    "hemoglobin_fast_config", "ttr_fast_config", "p53_fast_config",
    "run_analysis", "StepResult", "STEP_REGISTRY",
]
