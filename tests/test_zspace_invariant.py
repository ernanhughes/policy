import numpy as np

from policy.policy_container import PolicyContainer
from policy.calibrator.adaptive_calibrator import CalibrationResult


def dummy_ai(state):
    return {"value": state["value"]}


def dummy_energy(output, context):
    return float(output["value"])


def test_zspace_acceptance_invariant():

    calibration = CalibrationResult(
        tau_energy=0.0,
        energy_mean=1.0,
        energy_std=0.5,
        hard_negative_gap=0.0,
        hard_negative_gap_norm=0.0,
        positive_count=100,
        negative_count=0,
    )

    class StaticCalibrator:
        def update(self, energy):
            pass

    policy = PolicyContainer(
        ai_callable=dummy_ai,
        energy_function=dummy_energy,
        calibrator=StaticCalibrator(),
        calibration=calibration,
        review_margin=0.0,
        reject_margin=2.0,
    )

    tau_z = (calibration.tau_energy - calibration.energy_mean) / calibration.energy_std

    for value in np.linspace(0.0, 1.5, 50):
        energy = float(value)
        z = (energy - calibration.energy_mean) / calibration.energy_std

        _, decision = policy.execute({"value": energy})

        # Your policy accepts on equality at review boundary (z == tau_z)
        if z <= tau_z:
            assert decision.verdict == "ACCEPT"
        elif z > tau_z:
            assert decision.verdict in ("REVIEW", "REJECT")
