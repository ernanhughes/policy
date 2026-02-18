# Policy

**Model-agnostic AI governance container with adaptive calibration and drift detection.**

---

## Motivation

Self-improving AI systems can drift, destabilize, or degrade over time.

Policy Governor provides a deterministic governance layer that:

* Wraps any AI system
* Evaluates output risk via external scoring (e.g. hallucination energy)
* Learns thresholds dynamically
* Detects distribution drift
* Enforces ACCEPT / REVIEW / REJECT decisions

It is fully decoupled from:

* Any specific model
* Any embedding backend
* Any AI architecture

This makes it deployable across research systems, production AI pipelines, and safety-critical applications.

---

## Core Architecture

```css All right
AI System (Black Box)
        ↓
Energy Function (External)
        ↓
Calibrator (Adaptive)
        ↓
Policy Container
        ↓
ACCEPT / REVIEW / REJECT
```

---

## Example Usage

```python
from policy.core.policy_container import PolicyContainer
from policy.calibration.quantile_calibrator import QuantileCalibrator

def my_ai(input_data):
    return {"text": "Generated output"}

def energy_fn(output, context):
    return 0.15  # example energy

calibrator = QuantileCalibrator(
    quantile=0.2,
    warmup=50
)

policy = PolicyContainer(
    ai_callable=my_ai,
    energy_function=energy_fn,
    calibrator=calibrator,
    calibration=None
)

output, decision = policy.execute({"prompt": "Hello"})
print(decision.verdict)
```

---

## Features

* Dynamic quantile calibration
* Z-score calibration
* Drift detection
* Margin-based decisions
* Full decision trace object
* Simulation harness for runaway degradation experiments
* Statistical reporting
* Trajectory & variance visualization

---

## Research Motivation

Policy Governor is designed to:

* Prevent runaway degradation
* Provide mathematically defensible thresholds
* Separate governance from model logic
* Enable publishable safety experiments

---

## Installation

```bash
pip install policy
```

Or locally:

```bash
pip install .
```

---

## Development

```bash
pip install .[dev]
pytest
```

---

## License

Apache 2.0

---

# 3️⃣ Paper Abstract (Clean + Academic Tone)

You can drop this into a paper draft immediately.

---

### Abstract

Self-improving AI systems are vulnerable to runaway degradation when iterative updates accumulate error or distributional drift. While model-centric approaches attempt to mitigate hallucination and instability internally, they often entangle governance logic with learning dynamics.

We introduce **Policy Governor**, a model-agnostic governance container that wraps arbitrary AI systems and enforces calibrated decision boundaries based on externally computed risk signals, such as hallucination energy. The framework separates execution from evaluation, enabling deterministic ACCEPT / REVIEW / REJECT decisions without modifying the underlying model.

Policy Governor employs adaptive calibration mechanisms—including quantile-based and z-score normalization—combined with drift detection over rolling distributions. This architecture allows thresholds to adjust dynamically while maintaining statistical interpretability.

We demonstrate, through simulated runaway degradation experiments, that policy-bounded systems exhibit significantly reduced variance and bounded energy trajectories compared to unregulated baselines. The results suggest that externalized policy enforcement provides a stable control mechanism for self-improving AI pipelines.

This separation of governance from cognition establishes a deployable, model-independent safety layer applicable across research and production environments.
