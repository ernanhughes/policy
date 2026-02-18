import numpy as np

from policy.policy_container import PolicyContainer
from policy.calibrator.adaptive_calibrator import AdaptiveCalibrator


def dummy_ai(state):
    return {"value": state["value"]}


def dummy_energy(output, context):
    return float(output["value"])


def test_drift_detects_shift():

    base_distribution = np.random.normal(0.5, 0.1, 300)
    shifted_distribution = np.random.normal(2.0, 0.1, 200)

    calibrator = AdaptiveCalibrator(
        percentile=80.0,
        min_samples=50,
    )

    calibration = calibrator.calibrate(base_distribution.tolist())

    policy = PolicyContainer(
        ai_callable=dummy_ai,
        energy_function=dummy_energy,
        calibrator=calibrator,
        calibration=calibration,
        drift_threshold=1.0,
    )

    # Feed baseline data
    for e in base_distribution:
        policy.execute({"value": float(e)})

    # Now feed shifted distribution
    drift_flags = []

    for e in shifted_distribution:
        _, decision = policy.execute({"value": float(e)})
        drift_flags.append(decision.drift_score)

    assert any(score >= 1.0 for score in drift_flags)
