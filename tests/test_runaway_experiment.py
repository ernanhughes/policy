
from policy.experiments.runaway_experiment import RunawayDeclineExperiment
from policy.policy_container import PolicyContainer
from policy.calibrator.adaptive_calibrator import CalibrationResult


def dummy_ai(state):
    return {"quality": state["quality"]}


def dummy_energy(output, context):
    return abs(output["quality"] - 1.0)


class StaticCalibrator:
    def update(self, energy):
        pass  # no-op


def make_static_calibration():
    # Calibration values consistent with this simulation:
    # energy is typically small under noise_scale=0.05.
    return CalibrationResult(
        tau_energy=0.20,   # allow up to ~0.2 energy as "acceptable"
        energy_mean=0.08,  # rough nominal mean
        energy_std=0.06,   # rough nominal std
        hard_negative_gap=0.0,
        hard_negative_gap_norm=0.0,
        positive_count=100,
        negative_count=0,
    )


def test_policy_contains_tail_risk():

    episodes = 300

    baseline_exp = RunawayDeclineExperiment(
        ai_callable=dummy_ai,
        energy_function=dummy_energy,
        policy=None,
        episodes=episodes,
    )
    baseline = baseline_exp.run()

    policy = PolicyContainer(
        ai_callable=dummy_ai,
        energy_function=dummy_energy,
        calibrator=StaticCalibrator(),
        calibration=make_static_calibration(),
        review_margin=0.0,
        reject_margin=2.0,
    )

    bounded_exp = RunawayDeclineExperiment(
        ai_callable=dummy_ai,
        energy_function=dummy_energy,
        policy=policy,
        episodes=episodes,
    )
    bounded = bounded_exp.run()

    # Experiment must be meaningful
    assert bounded["accept_rate"] > 0.01
    assert len(bounded["energies"]) > 0

    # Policy-bounded system should reduce tail-risk
    assert bounded["max_energy"] <= baseline["max_energy"]
