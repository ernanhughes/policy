import numpy as np

from policy.policy_container import PolicyContainer
from policy.calibrator.adaptive_calibrator import AdaptiveCalibrator


def dummy_ai(state):
    return {"value": state["value"]}


def dummy_energy(output, context):
    return float(output["value"])


def generate_bimodal(n=500):
    """
    Low-energy cluster around 0.2
    High-energy cluster around 2.0
    """
    low = np.random.normal(0.2, 0.05, n // 2)
    high = np.random.normal(2.0, 0.2, n // 2)
    return np.concatenate([low, high])


def test_policy_separates_modes():

    energies = generate_bimodal(400)

    calibrator = AdaptiveCalibrator(
        percentile=80.0,
        min_samples=50,
    )

    calibration = calibrator.calibrate(energies[:200].tolist())

    policy = PolicyContainer(
        ai_callable=dummy_ai,
        energy_function=dummy_energy,
        calibrator=calibrator,
        calibration=calibration,
        reject_margin=-1.5,
    )

    accepted = 0
    rejected = 0

    for e in energies:
        _, decision = policy.execute({"value": float(e)})

        if decision.verdict == "ACCEPT":
            accepted += 1
        if decision.verdict == "REJECT":
            rejected += 1

    # We expect meaningful separation
    assert accepted > 0
    assert rejected > 0
    assert accepted < len(energies)
