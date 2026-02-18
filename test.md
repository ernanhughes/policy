============================= test session starts =============================
platform win32 -- Python 3.11.4, pytest-9.0.2, pluggy-1.6.0
rootdir: C:\Projects\policy
configfile: pyproject.toml
collected 7 items / 1 error

=================================== ERRORS ====================================
______________ ERROR collecting tests/test_runaway_experiment.py ______________
ImportError while importing test module 'C:\Projects\policy\tests\test_runaway_experiment.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\ernan\AppData\Local\Programs\Python\Python311\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_runaway_experiment.py:5: in <module>
    from policy.calibration.adaptive_calibrator import CalibrationResult
E   ModuleNotFoundError: No module named 'policy.calibration.adaptive_calibrator'; 'policy.calibration' is not a package
=========================== short test summary info ===========================
ERROR tests/test_runaway_experiment.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.17s ===============================
