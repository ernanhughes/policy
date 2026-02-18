# Calibration

Calibration is distribution-aware threshold learning.

Inputs:

* Positive energies
* Optional hard-negative energies

Outputs:

* τ (threshold)
* μ (mean)
* σ (std)
* hard-negative separation gap

---

## Hard Negative Gap

```
gap = mean(E_neg) - mean(E_pos)
gap_norm = gap / σ
```

This quantifies adversarial separation in Z-space.

---

## Drift Detection

We detect distribution shift via:

```
z_shift = |mean_recent - μ| / σ
```

Drift occurs if:

```
z_shift > drift_threshold
```
