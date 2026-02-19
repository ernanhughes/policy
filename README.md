# 🛡️ Policy — Risk-Bounded AI Governance Framework

**Policy** is a Z-space normalized governance layer for AI systems.

It wraps any callable model and enforces calibrated, distribution-aware decision boundaries using hallucination-energy (or any risk scalar).

This is not filtering.
This is **risk containment mathematics**.

---

## 🎯 Core Idea

We treat model uncertainty as a measurable scalar: **energy**.

We then:

1. Learn the positive distribution.
2. Calibrate a threshold.
3. Normalize everything into **Z-space**.
4. Enforce bounded decision logic.

This produces:

* Scale invariance
* Distribution awareness
* Drift detection
* Risk-bounded containment

---

## 📦 Installation

```bash
pip install policy-governance
```

Or locally:

```bash
pip install -e .
```

---

## 🚀 Quick Example

```python
from policy.core.policy_container import PolicyContainer
from policy.calibration.adaptive_calibrator import AdaptiveCalibrator

def ai(x):
    return {"value": x}

def energy(output, ctx):
    return abs(output["value"] - 1.0)

# Calibrate
calibrator = AdaptiveCalibrator(percentile=95.0)
calibration = calibrator.calibrate(positive_energies)

policy = PolicyContainer(
    ai_callable=ai,
    energy_function=energy,
    calibrator=calibrator,
    calibration=calibration,
)

output, decision = policy.execute(0.8)

print(decision.verdict)
```

---

## 📐 Z-Space Decision Rule

We normalize energy:

```
z = (energy - μ) / σ
```

Decision boundary:

```css
REJECT  if z ≥ τ_z + reject_margin
REVIEW  if z >  τ_z + review_margin
ACCEPT  otherwise
```

Where:

```math
τ_z = (τ - μ) / σ
```

This guarantees:

* Scale invariance
* Distribution awareness
* Calibrated containment

---

## 🧪 Included Experiments

* Runaway self-modification simulation
* Distribution separation tests
* Z-space invariance verification
* Drift detection validation

---

## 📊 What This Framework Guarantees

✔ Decision boundaries are invariant to scaling
✔ Drift is detectable in Z-space
✔ Tail risk is constrained
✔ Calibration artifacts are portable

---

## 🧠 Intended Use

* LLM hallucination gating
* Agentic system containment
* Policy-bounded self-modification
* High-assurance deployment environments

---

## 📖 Documentation

See:

* [`docs/theory.md`](docs/theory.md)
* [`docs/calibration.md`](docs/calibration.md)
* [`docs/architecture.md`](docs/architecture.md)
* [`docs/experiments.md`](docs/experiments.md)
* [`docs/testing.md`](docs/testing.md)

---

## 📜 License

Apache 2.0 License
