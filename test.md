============================= test session starts =============================
platform win32 -- Python 3.11.4, pytest-9.0.2, pluggy-1.6.0
rootdir: C:\Projects\policy
configfile: pyproject.toml
collected 8 items

tests\test_calibrator.py ..                                              [ 25%]
tests\test_distribution_separation.py .                                  [ 37%]
tests\test_drift_detection.py .                                          [ 50%]
tests\test_policy_container.py ..                                        [ 75%]
tests\test_runaway_experiment.py F                                       [ 87%]
tests\test_zspace_invariant.py F                                         [100%]

================================== FAILURES ===================================
_______________________ test_policy_reduces_mean_energy _______________________

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
    
>       bounded = bounded_exp.run()
                  ^^^^^^^^^^^^^^^^^

tests\test_runaway_experiment.py:74: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
src\policy\experiments\runaway_experiment.py:59: in run
    "max_energy": float(np.max(energies)),
                        ^^^^^^^^^^^^^^^^
venv\Lib\site-packages\numpy\_core\fromnumeric.py:3123: in max
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

obj = [], ufunc = <ufunc 'maximum'>, method = 'max', axis = None, dtype = None
out = None
kwargs = {'initial': <no value>, 'keepdims': <no value>, 'where': <no value>}
passkwargs = {}

    def _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs):
        passkwargs = {k: v for k, v in kwargs.items()
                      if v is not np._NoValue}
    
        if type(obj) is not mu.ndarray:
            try:
                reduction = getattr(obj, method)
            except AttributeError:
                pass
            else:
                # This branch is needed for reductions like any which don't
                # support a dtype.
                if dtype is not None:
                    return reduction(axis=axis, dtype=dtype, out=out, **passkwargs)
                else:
                    return reduction(axis=axis, out=out, **passkwargs)
    
>       return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       ValueError: zero-size array to reduction operation maximum which has no identity

venv\Lib\site-packages\numpy\_core\fromnumeric.py:83: ValueError
______________________ test_zspace_acceptance_invariant _______________________

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
        )
    
        tau_z = (calibration.tau_energy - calibration.energy_mean) / calibration.energy_std
    
        for value in np.linspace(0.0, 1.5, 50):
            energy = float(value)
            z = (energy - calibration.energy_mean) / calibration.energy_std
    
            _, decision = policy.execute({"value": energy})
    
            if z < tau_z:
                assert decision.verdict == "ACCEPT"
            elif z >= tau_z:
>               assert decision.verdict in ("REVIEW", "REJECT")
E               AssertionError: assert 'ACCEPT' in ('REVIEW', 'REJECT')
E                +  where 'ACCEPT' = PolicyDecision(verdict='ACCEPT', energy=0.0, margin=0.0, drift_score=0.0, timestamp=1771417700.137981, metadata={'z_score': -2.0, 'tau_z': -2.0, 'energy_mean': 1.0, 'energy_std': 0.5}).verdict

tests\test_zspace_invariant.py:49: AssertionError
============================== warnings summary ===============================
tests/test_runaway_experiment.py::test_policy_reduces_mean_energy
  C:\Projects\policy\venv\Lib\site-packages\numpy\_core\fromnumeric.py:3824: RuntimeWarning: Mean of empty slice
    return _methods._mean(a, axis=axis, dtype=dtype,

tests/test_runaway_experiment.py::test_policy_reduces_mean_energy
  C:\Projects\policy\venv\Lib\site-packages\numpy\_core\_methods.py:142: RuntimeWarning: invalid value encountered in scalar divide
    ret = ret.dtype.type(ret / rcount)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/test_runaway_experiment.py::test_policy_reduces_mean_energy - Va...
FAILED tests/test_zspace_invariant.py::test_zspace_acceptance_invariant - Ass...
=================== 2 failed, 6 passed, 2 warnings in 0.15s ===================
