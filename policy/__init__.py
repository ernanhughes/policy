"""
Policy: Model-agnostic AI governance container.

This package provides a governance container that wraps any AI system and
enforces calibrated decision boundaries based on hallucination energy or
other risk metrics.
"""

from .container import PolicyContainer
from .metrics import HallucinationMetric, RiskMetric
from .boundaries import DecisionBoundary, CalibratedBoundary
from .config import PolicyConfig

__version__ = "0.1.0"
__all__ = [
    "PolicyContainer",
    "HallucinationMetric",
    "RiskMetric",
    "DecisionBoundary",
    "CalibratedBoundary",
    "PolicyConfig",
]
