# tests/test_runaway_experiment.py

import numpy as np

from policy.experiments.runaway_experiment import RunawayDeclineExperiment
from policy.policy_container import PolicyContainer
from policy.calibrator.adaptive_calibrator import CalibrationResult


def dummy_ai(state):
    return {"quality": state["quality"]}


def dummy_energy(output, context):
    return abs(output["quality"] - 1.0)


# ----------------------------------------
# Static calibration artifact (realistic)
# ----------------------------------------

class StaticCalibrator:
    def update(self, energy):
        pass  # no-op for testing


def make_static_calibration():
    return CalibrationResult(
        tau_energy=0.0,  # threshold at mean
        energy_mean=0.5,
        energy_std=0.2,
        hard_negative_gap=0.0,
        hard_negative_gap_norm=0.0,
        positive_count=100,
        negative_count=0,
    )


# ----------------------------------------
# Test
# ----------------------------------------

def test_policy_reduces_mean_energy():

    episodes = 200

    # Baseline (no policy)
    baseline_exp = RunawayDeclineExperiment(
        ai_callable=dummy_ai,
        energy_function=dummy_energy,
        policy=None,
        episodes=episodes,
    )

    baseline = baseline_exp.run()

    # Bounded (with policy)
    calibration = make_static_calibration()

    policy = PolicyContainer(
        ai_callable=dummy_ai,
        energy_function=dummy_energy,
        calibrator=StaticCalibrator(),
        calibration=calibration,
    )

    bounded_exp = RunawayDeclineExperiment(
        ai_callable=dummy_ai,
        energy_function=dummy_energy,
        policy=policy,
        episodes=episodes,
    )

    bounded = bounded_exp.run()

    assert np.mean(bounded["energies"]) <= np.mean(baseline["energies"])
