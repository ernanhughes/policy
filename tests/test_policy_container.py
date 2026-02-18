import numpy as np
from policy.policy_container import PolicyContainer
from policy.calibrator.adaptive_calibrator import AdaptiveCalibrator


# --------------------------------------------------------
# Dummy AI
# --------------------------------------------------------

def dummy_ai(state):
    return {"value": state["value"]}


def dummy_energy(output, context):
    return float(output["value"])


# --------------------------------------------------------
# Utility: Generate Sine Wave Energies
# --------------------------------------------------------

def generate_sine_wave(n=200, mean=1.0, amplitude=0.5):
    x = np.linspace(0, 4 * np.pi, n)
    return mean + amplitude * np.sin(x)


# --------------------------------------------------------
# Test: Accept / Review / Reject
# --------------------------------------------------------

def test_policy_with_real_calibration():

    energies = generate_sine_wave(200)

    calibrator = AdaptiveCalibrator(
        percentile=80.0,
        min_samples=50,
    )

    calibration = calibrator.calibrate(energies[:100].tolist())

    policy = PolicyContainer(
        ai_callable=dummy_ai,
        energy_function=dummy_energy,
        calibrator=calibrator,
        calibration=calibration,
        review_margin=0.0,
        reject_margin=-1.5,
    )


    # Low energy point
    low_value = float(np.min(energies))
    _, decision_low = policy.execute({"value": low_value})

    assert decision_low.verdict == "ACCEPT"

    # High energy point
    high_value = float(np.max(energies))
    _, decision_high = policy.execute({"value": high_value})

    assert decision_high.verdict in ("REVIEW", "REJECT")


# --------------------------------------------------------
# Test: Drift Detection
# --------------------------------------------------------

def test_drift_detection_with_distribution_shift():

    base = generate_sine_wave(200)

    calibrator = AdaptiveCalibrator(
        percentile=80.0,
        min_samples=50,
    )

    calibration = calibrator.calibrate(base[:100].tolist())

    policy = PolicyContainer(
        ai_callable=dummy_ai,
        energy_function=dummy_energy,
        calibrator=calibrator,
        calibration=calibration,
        drift_threshold=1.0,
    )

    # Inject shifted distribution (mean + 1.0)
    shifted = base[100:] + 1.0

    for val in shifted[:50]:
        _, decision = policy.execute({"value": float(val)})

    assert decision.drift_score > 1.0
