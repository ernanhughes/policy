# policy/core/policy_container.py

from __future__ import annotations

import time
from typing import Callable, Any, Dict, List, Optional

import numpy as np

from policy.protocols.calibration import CalibratorProtocol
from policy.calibrator.adaptive_calibrator import CalibrationResult
from policy.custom_types import PolicyDecision

# ============================================================
# Policy Container
# ============================================================

class PolicyContainer:
    """
    AI Governance Wrapper.

    Wraps ANY AI system and enforces calibrated
    hallucination-energy-based governance.

    Fully Z-space normalized.
    """

    def __init__(
        self,
        ai_callable: Callable[..., Any],
        energy_function: Callable[[Any, Dict], float],
        calibrator: CalibratorProtocol,
        calibration: CalibrationResult,
        review_margin: float = 0.0,
        reject_margin: float = 2.0,
        drift_threshold: float = 2.0,
    ):
        self.ai_callable = ai_callable
        self.energy_function = energy_function
        self.calibrator = calibrator
        self.calibration = calibration

        self.review_margin = review_margin
        self.reject_margin = reject_margin
        self.drift_threshold = drift_threshold

        self.recent_energies: List[float] = []

    # ------------------------------------------------------------
    # Main Execution
    # ------------------------------------------------------------

    def execute(self, *args, context: Optional[Dict] = None, **kwargs):

        context = context or {}

        # 1️⃣ Run AI
        output = self.ai_callable(*args, **kwargs)

        # 2️⃣ Compute energy externally
        energy = float(self.energy_function(output, context))

        self.recent_energies.append(energy)

        mu = self.calibration.energy_mean
        sigma = self.calibration.energy_std
        tau = self.calibration.tau_energy

        # 3️⃣ Compute Z-score
        if sigma < 1e-8:
            z_score = 0.0
        else:
            z_score = (energy - mu) / sigma

        # 4️⃣ Margin in Z-space
        tau_z = (tau - mu) / sigma if sigma > 1e-8 else 0.0
        margin = tau_z - z_score

        # 5️⃣ Drift detection
        drift_score = 0.0
        if len(self.recent_energies) >= 20 and sigma > 1e-8:
            recent = np.array(self.recent_energies[-100:])
            mean_recent = float(np.mean(recent))
            drift_score = (mean_recent - mu) / sigma

        # 6️⃣ Decision
        verdict = self._decide(z_score, tau_z)

        decision = PolicyDecision(
            verdict=verdict,
            energy=energy,
            margin=margin,
            drift_score=float(drift_score),
            timestamp=time.time(),
            metadata={
                **context,
                "z_score": z_score,
                "tau_z": tau_z,
                "energy_mean": mu,
                "energy_std": sigma,
            },
        )

        return output, decision

    # ------------------------------------------------------------
    # Decision Logic
    # ------------------------------------------------------------

    def _decide(self, z_score: float, tau_z: float) -> str:

        reject_boundary = tau_z + self.reject_margin
        review_boundary = tau_z + self.review_margin

        if z_score >= reject_boundary:
            return "REJECT"

        if z_score > review_boundary:
            return "REVIEW"

        return "ACCEPT"

    # ------------------------------------------------------------
    # Recalibration
    # ------------------------------------------------------------

    def recalibrate(
        self,
        positive_energies: List[float],
        hard_negative_energies: Optional[List[float]] = None,
    ):
        self.calibration = self.calibrator.calibrate(
            positive_energies,
            hard_negative_energies,
        )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def diagnostics(self) -> Dict[str, Any]:
        if len(self.recent_energies) == 0:
            return {
                "energies": [],
                "mean_energy": float("nan"),
                "max_energy": float("nan"),
                "final_energy": float("nan"),
                "accept_rate": 0.0,
            }
        return {
            "tau_energy": self.calibration.tau_energy,
            "energy_mean": self.calibration.energy_mean,
            "energy_std": self.calibration.energy_std,
            "hard_negative_gap_norm": self.calibration.hard_negative_gap_norm,
            "recent_mean_energy": float(np.mean(self.recent_energies[-50:]))
            if self.recent_energies
            else None,
        }
