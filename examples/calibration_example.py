"""Example demonstrating calibration of decision boundaries."""

import numpy as np
from policy import PolicyContainer, PolicyConfig, HallucinationMetric, CalibratedBoundary


def ml_model(input_data):
    """Simulated ML model with varying uncertainty."""
    # Generate random predictions with different characteristics
    if np.random.random() < 0.3:
        # High confidence (low risk)
        probs = np.random.dirichlet([10, 1, 1])
    else:
        # Low confidence (high risk)
        probs = np.random.dirichlet([3, 2, 2])
    return probs


def generate_calibration_data(n_samples=200):
    """Generate labeled calibration data.
    
    In practice, this would come from human evaluation or
    ground truth validation.
    """
    calibration_data = []
    
    for i in range(n_samples):
        predictions = ml_model(f"sample_{i}")
        
        # Simulate labeling: high confidence predictions are acceptable
        # In real scenarios, this would be manual review or validation
        max_prob = np.max(predictions)
        is_acceptable = max_prob > 0.6
        
        calibration_data.append((predictions, is_acceptable))
    
    return calibration_data


def main():
    """Run calibration example."""
    print("=" * 60)
    print("Policy Package - Calibration Example")
    print("=" * 60)
    print()
    
    # Create policy container with calibrated boundary
    config = PolicyConfig(
        threshold=0.5,
        risk_tolerance=0.1,  # Target 10% false positive rate
        calibration_samples=100,
        enable_logging=True
    )
    
    boundary = CalibratedBoundary(
        initial_threshold=0.5,
        target_fpr=0.1,
        min_samples=100
    )
    
    policy = PolicyContainer(
        model=ml_model,
        metric=HallucinationMetric(),
        boundary=boundary,
        config=config
    )
    
    print("Step 1: Making predictions before calibration")
    print("-" * 60)
    
    # Make some predictions before calibration
    for i in range(5):
        decision = policy.predict(f"test_input_{i}")
        print(f"Prediction {i+1}: Allowed={decision.allowed}, "
              f"Risk={decision.metric_value:.4f}, "
              f"Threshold={decision.threshold:.4f}")
    
    stats_before = policy.get_stats()
    print(f"\nAllow rate before calibration: {stats_before['allow_rate']:.2%}")
    
    # Reset history for clean comparison
    policy.reset_history()
    
    # Generate calibration data
    print("\n" + "=" * 60)
    print("Step 2: Generating calibration data")
    print("-" * 60)
    print("Generating 200 labeled samples...")
    
    calibration_data = generate_calibration_data(n_samples=200)
    acceptable_count = sum(1 for _, label in calibration_data if label)
    print(f"Generated {len(calibration_data)} samples")
    print(f"Acceptable: {acceptable_count} ({acceptable_count/len(calibration_data):.1%})")
    print(f"Unacceptable: {len(calibration_data) - acceptable_count}")
    
    # Calibrate the boundary
    print("\n" + "=" * 60)
    print("Step 3: Calibrating decision boundary")
    print("-" * 60)
    
    success = policy.calibrate(calibration_data, auto_apply=True)
    
    if success:
        print("✓ Calibration successful!")
        
        # Get calibration statistics
        if isinstance(policy.boundary, CalibratedBoundary):
            calib_stats = policy.boundary.get_calibration_stats()
            print(f"\nCalibration Statistics:")
            print(f"  Samples used: {calib_stats['n_samples']}")
            print(f"  New threshold: {calib_stats['threshold']:.4f}")
            if 'acceptable_mean' in calib_stats:
                print(f"  Acceptable samples mean risk: {calib_stats['acceptable_mean']:.4f}")
            if 'unacceptable_mean' in calib_stats:
                print(f"  Unacceptable samples mean risk: {calib_stats['unacceptable_mean']:.4f}")
    else:
        print("✗ Calibration failed (insufficient samples)")
    
    # Make predictions after calibration
    print("\n" + "=" * 60)
    print("Step 4: Making predictions after calibration")
    print("-" * 60)
    
    for i in range(5):
        decision = policy.predict(f"test_input_after_{i}")
        print(f"Prediction {i+1}: Allowed={decision.allowed}, "
              f"Risk={decision.metric_value:.4f}, "
              f"Threshold={decision.threshold:.4f}")
    
    stats_after = policy.get_stats()
    print(f"\nAllow rate after calibration: {stats_after['allow_rate']:.2%}")
    
    # Compare before and after
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Threshold before: 0.5000")
    print(f"Threshold after:  {policy.boundary.get_threshold():.4f}")
    print(f"\nThe boundary has been automatically adjusted based on")
    print(f"calibration data to meet the target risk tolerance.")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
