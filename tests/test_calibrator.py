# tests/test_calibrator.py

import numpy as np
from policy.calibrator.adaptive_calibrator import AdaptiveCalibrator


def test_calibrator_updates_statistics():
    calibrator = AdaptiveCalibrator(
        percentile=95.0,
        min_samples=10,
    )

    values = np.linspace(0.0, 1.0, 20)

    result = calibrator.calibrate(values.tolist())

    assert result.energy_mean is not None
    assert result.energy_std > 0
    assert result.tau_energy is not None
    assert result.positive_count == 20


def test_tau_quantile_behavior():
    calibrator = AdaptiveCalibrator(
        percentile=20.0,  # 20th percentile
        min_samples=10,
    )

    values = list(range(50))

    result = calibrator.calibrate(values)

    expected_tau = np.percentile(values, 20.0)

    assert abs(result.tau_energy - expected_tau) < 1e-6
