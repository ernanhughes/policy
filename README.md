# Policy: Model-Agnostic AI Governance Container

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Policy** is a production-ready Python package that implements a model-agnostic AI governance container. It wraps any AI system and enforces calibrated decision boundaries based on hallucination energy or other risk metrics.

## Features

- 🎯 **Model-Agnostic**: Works with any AI model or prediction system
- 🔒 **Risk-Based Governance**: Enforce boundaries based on hallucination energy and uncertainty
- 📊 **Calibrated Boundaries**: Automatically calibrate thresholds based on labeled data
- 📈 **Comprehensive Metrics**: Built-in hallucination and risk metrics
- 🔧 **Highly Configurable**: Flexible configuration for different use cases
- 📝 **Production-Ready**: Type hints, logging, and comprehensive testing

## Installation

```bash
pip install -e .
```

For development with testing tools:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from policy import PolicyContainer, PolicyConfig, HallucinationMetric

# Define your AI model
def my_model(input_data):
    # Your model logic here
    return [0.7, 0.2, 0.1]  # Probability distribution

# Create policy container
policy = PolicyContainer(
    model=my_model,
    metric=HallucinationMetric(),
    config=PolicyConfig(threshold=0.5)
)

# Make governed predictions
decision = policy.predict("some input")

if decision.allowed:
    print(f"Prediction: {decision.metadata['prediction']}")
    print(f"Risk score: {decision.metric_value:.4f}")
else:
    print("Prediction rejected due to high risk")
```

## Core Concepts

### 1. Hallucination Energy

Hallucination energy measures the model's uncertainty and potential for generating unreliable outputs. It combines:

- **Entropy-based uncertainty**: Higher entropy indicates more uncertain predictions
- **Consistency checks**: Compares multiple samples to detect inconsistencies
- **Temperature scaling**: Adjustable sensitivity to uncertainty

### 2. Risk Metrics

Risk metrics provide a comprehensive view of prediction reliability:

- **Uncertainty**: Model's confidence in its prediction
- **Confidence**: Maximum probability in the prediction
- **Consistency**: Agreement across multiple predictions

### 3. Decision Boundaries

Decision boundaries determine whether a prediction is acceptable:

- **Fixed Boundary**: Simple threshold-based decisions
- **Calibrated Boundary**: Automatically adjusts based on labeled data to maintain target false positive rate

## Usage Examples

### Basic Usage

```python
from policy import PolicyContainer, HallucinationMetric

def classifier(text):
    # Your classification model
    return [0.8, 0.15, 0.05]

policy = PolicyContainer(model=classifier)
decision = policy.predict("input text")

print(f"Allowed: {decision.allowed}")
print(f"Risk: {decision.metric_value:.4f}")
```

### Calibration

```python
from policy import PolicyContainer, CalibratedBoundary

# Create container with calibrated boundary
policy = PolicyContainer(
    model=my_model,
    boundary=CalibratedBoundary(
        initial_threshold=0.5,
        target_fpr=0.1,  # 10% false positive rate
        min_samples=100
    )
)

# Collect calibration data (predictions, is_acceptable)
calibration_data = [
    ([0.8, 0.1, 0.1], True),   # High confidence - acceptable
    ([0.4, 0.3, 0.3], False),  # Low confidence - unacceptable
    # ... more samples
]

# Calibrate boundary
policy.calibrate(calibration_data)
```

### Custom Configuration

```python
from policy import PolicyContainer, PolicyConfig, RiskMetric

config = PolicyConfig(
    threshold=0.3,
    risk_tolerance=0.05,
    calibration_samples=500,
    enable_logging=True,
    metadata={'project': 'my-ai-system'}
)

policy = PolicyContainer(
    model=my_model,
    metric=RiskMetric(weights={
        'uncertainty': 0.5,
        'confidence': 0.3,
        'consistency': 0.2
    }),
    config=config
)
```

### Monitoring and Statistics

```python
# Make multiple predictions
for data in dataset:
    decision = policy.predict(data)
    # ... process decision

# Get statistics
stats = policy.get_stats()
print(f"Total predictions: {stats['total_predictions']}")
print(f"Allow rate: {stats['allow_rate']:.2%}")
print(f"Current threshold: {stats['current_threshold']:.4f}")
```

## API Reference

### PolicyContainer

Main container for AI governance.

**Methods:**
- `predict(input_data, context=None, return_raw=False)`: Make a governed prediction
- `evaluate(predictions, context=None)`: Evaluate predictions without running model
- `calibrate(calibration_data, auto_apply=True)`: Calibrate decision boundary
- `get_stats()`: Get statistics about policy decisions
- `reset_history()`: Reset prediction history

### HallucinationMetric

Metric for estimating hallucination energy.

**Parameters:**
- `temperature` (float): Temperature parameter for scaling entropy (default: 1.0)

**Methods:**
- `compute(predictions, context=None)`: Compute hallucination energy (0.0 to 1.0)

### RiskMetric

General risk metric combining multiple factors.

**Parameters:**
- `weights` (dict): Weights for risk components (default: equal weighting)

**Methods:**
- `compute(predictions, context=None)`: Compute overall risk score (0.0 to 1.0)

### CalibratedBoundary

Automatically calibrated decision boundary.

**Parameters:**
- `initial_threshold` (float): Initial threshold value (default: 0.5)
- `target_fpr` (float): Target false positive rate (default: 0.1)
- `min_samples` (int): Minimum samples for calibration (default: 100)

**Methods:**
- `evaluate(metric_value)`: Check if value passes boundary
- `calibrate()`: Perform calibration
- `get_calibration_stats()`: Get calibration statistics

### PolicyConfig

Configuration for policy container.

**Parameters:**
- `threshold` (float): Decision boundary threshold (0.0 to 1.0)
- `risk_tolerance` (float): Maximum acceptable risk level (0.0 to 1.0)
- `calibration_samples` (int): Number of samples for calibration
- `enable_logging` (bool): Enable logging of decisions
- `metadata` (dict): Additional configuration metadata

## Examples

Run the included examples:

```bash
# Basic usage example
python examples/basic_example.py

# Calibration example
python examples/calibration_example.py
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=policy --cov-report=html

# Run specific test file
pytest tests/test_container.py
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/ernanhughes/policy.git
cd policy

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black policy tests examples

# Lint code
ruff check policy tests examples

# Type checking
mypy policy
```

## Use Cases

### 1. Production ML Systems

Wrap production ML models to reject predictions with high uncertainty:

```python
policy = PolicyContainer(
    model=production_model,
    config=PolicyConfig(threshold=0.3)  # Conservative threshold
)
```

### 2. Human-in-the-Loop Systems

Use calibrated boundaries to route uncertain predictions to human review:

```python
policy = PolicyContainer(
    model=ai_assistant,
    boundary=CalibratedBoundary(target_fpr=0.05)
)

decision = policy.predict(user_query)
if not decision.allowed:
    route_to_human_review(user_query)
```

### 3. A/B Testing AI Safety

Compare different governance configurations:

```python
# Conservative policy
policy_a = PolicyContainer(config=PolicyConfig(threshold=0.3))

# Permissive policy
policy_b = PolicyContainer(config=PolicyConfig(threshold=0.7))

# Compare metrics
stats_a = policy_a.get_stats()
stats_b = policy_b.get_stats()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{policy2026,
  title = {Policy: Model-Agnostic AI Governance Container},
  author = {Hughes, Ernan},
  year = {2026},
  url = {https://github.com/ernanhughes/policy}
}
```

## Support

For issues and questions:
- Create an issue on [GitHub](https://github.com/ernanhughes/policy/issues)
- Check the [documentation](https://github.com/ernanhughes/policy)

## Roadmap

- [ ] Additional metrics (perplexity, toxicity, bias)
- [ ] Integration with popular ML frameworks (TensorFlow, PyTorch, scikit-learn)
- [ ] Web UI for policy monitoring
- [ ] Advanced calibration methods
- [ ] Multi-model ensemble governance