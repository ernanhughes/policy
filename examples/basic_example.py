"""Basic example of using the policy package."""

import numpy as np
from policy import PolicyContainer, PolicyConfig, HallucinationMetric


def simple_classifier(input_text):
    """Simulated AI classifier that returns probability distribution."""
    # Simulate different confidence levels based on input
    if "certain" in input_text.lower():
        # High confidence prediction
        return np.array([0.85, 0.10, 0.05])
    elif "uncertain" in input_text.lower():
        # Low confidence prediction
        return np.array([0.40, 0.35, 0.25])
    else:
        # Medium confidence
        return np.array([0.60, 0.25, 0.15])


def main():
    """Run basic example."""
    print("=" * 60)
    print("Policy Package - Basic Example")
    print("=" * 60)
    print()
    
    # Create policy configuration
    config = PolicyConfig(
        threshold=0.5,  # Risk threshold
        risk_tolerance=0.1,
        enable_logging=True
    )
    
    # Create policy container with the model
    policy = PolicyContainer(
        model=simple_classifier,
        metric=HallucinationMetric(),
        config=config
    )
    
    # Test different inputs
    test_inputs = [
        "This is a certain classification",
        "This is an uncertain classification",
        "This is a normal classification"
    ]
    
    print("Making governed predictions:")
    print("-" * 60)
    
    for input_text in test_inputs:
        print(f"\nInput: {input_text}")
        
        # Make prediction with governance
        decision = policy.predict(input_text, return_raw=True)
        
        print(f"  Prediction: {decision.metadata['prediction']}")
        print(f"  Risk Score: {decision.metric_value:.4f}")
        print(f"  Threshold: {decision.threshold:.4f}")
        print(f"  Allowed: {decision.allowed}")
        
        if not decision.allowed:
            print("  ⚠️  REJECTED - Risk too high!")
        else:
            print("  ✓ ACCEPTED")
    
    # Display statistics
    print("\n" + "=" * 60)
    print("Policy Statistics")
    print("=" * 60)
    stats = policy.get_stats()
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Allowed: {stats['allowed']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Allow rate: {stats['allow_rate']:.2%}")
    print(f"Current threshold: {stats['current_threshold']:.4f}")


if __name__ == "__main__":
    main()
