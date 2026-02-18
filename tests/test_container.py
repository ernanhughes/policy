"""Tests for policy container."""

import pytest
import numpy as np
from policy.container import PolicyContainer, PolicyDecision
from policy.config import PolicyConfig
from policy.metrics import HallucinationMetric, RiskMetric
from policy.boundaries import FixedBoundary, CalibratedBoundary


def dummy_model(input_data):
    """Dummy model for testing."""
    # Returns a probability distribution
    return [0.7, 0.2, 0.1]


def uncertain_model(input_data):
    """Model with uncertain predictions."""
    return [0.4, 0.35, 0.25]


class TestPolicyDecision:
    """Tests for PolicyDecision dataclass."""
    
    def test_creation(self):
        """Test creating a policy decision."""
        decision = PolicyDecision(
            allowed=True,
            metric_value=0.3,
            threshold=0.5,
            metadata={'test': 'data'}
        )
        assert decision.allowed is True
        assert decision.metric_value == 0.3
        assert decision.threshold == 0.5
        assert decision.metadata == {'test': 'data'}


class TestPolicyContainer:
    """Tests for PolicyContainer."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        container = PolicyContainer()
        assert container.model is None
        assert isinstance(container.metric, HallucinationMetric)
        assert isinstance(container.boundary, CalibratedBoundary)
        assert isinstance(container.config, PolicyConfig)
    
    def test_initialization_with_model(self):
        """Test initialization with model."""
        container = PolicyContainer(model=dummy_model)
        assert container.model is dummy_model
    
    def test_initialization_custom(self):
        """Test initialization with custom components."""
        config = PolicyConfig(threshold=0.3)
        metric = RiskMetric()
        boundary = FixedBoundary(threshold=0.4)
        
        container = PolicyContainer(
            model=dummy_model,
            metric=metric,
            boundary=boundary,
            config=config
        )
        
        assert container.model is dummy_model
        assert container.metric is metric
        assert container.boundary is boundary
        assert container.config is config
    
    def test_evaluate(self):
        """Test evaluation without model."""
        container = PolicyContainer()
        predictions = [0.8, 0.15, 0.05]
        
        decision = container.evaluate(predictions)
        
        assert isinstance(decision, PolicyDecision)
        assert isinstance(decision.allowed, bool)
        assert 0.0 <= decision.metric_value <= 1.0
        assert decision.threshold > 0.0
    
    def test_evaluate_with_context(self):
        """Test evaluation with context."""
        container = PolicyContainer()
        predictions = [0.7, 0.2, 0.1]
        context = {'confidence': 0.7}
        
        decision = container.evaluate(predictions, context)
        
        assert isinstance(decision, PolicyDecision)
        assert 'metric_type' in decision.metadata
        assert 'boundary_type' in decision.metadata
    
    def test_predict_without_model(self):
        """Test predict raises error without model."""
        container = PolicyContainer()
        
        with pytest.raises(ValueError, match="No model configured"):
            container.predict("test input")
    
    def test_predict_with_model(self):
        """Test prediction with model."""
        container = PolicyContainer(model=dummy_model)
        
        decision = container.predict("test input")
        
        assert isinstance(decision, PolicyDecision)
        assert 'prediction' in decision.metadata
        assert len(container.prediction_history) == 1
    
    def test_predict_return_raw(self):
        """Test prediction with return_raw flag."""
        container = PolicyContainer(model=dummy_model)
        
        decision = container.predict("test input", return_raw=True)
        
        assert 'raw_prediction' in decision.metadata
        assert 'prediction' in decision.metadata
    
    def test_allowed_prediction(self):
        """Test that low-risk prediction is allowed."""
        config = PolicyConfig(threshold=0.8)  # High threshold
        container = PolicyContainer(
            model=dummy_model,
            config=config,
            boundary=FixedBoundary(threshold=0.8)
        )
        
        decision = container.predict("test input")
        
        # Peaked distribution should have low energy and be allowed
        assert decision.allowed is True
    
    def test_rejected_prediction(self):
        """Test that high-risk prediction is rejected."""
        config = PolicyConfig(threshold=0.1)  # Low threshold
        container = PolicyContainer(
            model=uncertain_model,
            config=config,
            boundary=FixedBoundary(threshold=0.1)
        )
        
        decision = container.predict("test input")
        
        # Uncertain distribution should have high energy and be rejected
        assert decision.allowed is False
    
    def test_calibrate(self):
        """Test calibration with labeled data."""
        # Create container with lower minimum samples for faster test
        boundary = CalibratedBoundary(min_samples=50, target_fpr=0.1)
        container = PolicyContainer(boundary=boundary)
        
        # Generate calibration data
        calibration_data = [
            ([0.8, 0.15, 0.05], True),  # Acceptable
            ([0.75, 0.20, 0.05], True),  # Acceptable
            ([0.4, 0.35, 0.25], False),  # Unacceptable
            ([0.35, 0.35, 0.30], False),  # Unacceptable
        ]
        
        # Need enough samples for calibration
        calibration_data = calibration_data * 15  # 60 samples
        
        success = container.calibrate(calibration_data)
        
        assert success is True
    
    def test_calibrate_non_calibratable_boundary(self):
        """Test calibration with fixed boundary."""
        boundary = FixedBoundary(threshold=0.5)
        container = PolicyContainer(boundary=boundary)
        
        calibration_data = [([0.8, 0.1, 0.1], True)]
        success = container.calibrate(calibration_data)
        
        assert success is False
    
    def test_get_stats_empty(self):
        """Test statistics with no predictions."""
        container = PolicyContainer()
        stats = container.get_stats()
        
        assert stats['total_predictions'] == 0
        assert stats['allowed'] == 0
        assert stats['rejected'] == 0
        assert stats['allow_rate'] == 0.0
    
    def test_get_stats_with_predictions(self):
        """Test statistics with predictions."""
        container = PolicyContainer(
            model=dummy_model,
            boundary=FixedBoundary(threshold=0.8)
        )
        
        # Make several predictions
        for i in range(5):
            container.predict("test input")
        
        stats = container.get_stats()
        
        assert stats['total_predictions'] == 5
        assert stats['allowed'] + stats['rejected'] == 5
        assert 0.0 <= stats['allow_rate'] <= 1.0
        assert 'current_threshold' in stats
    
    def test_reset_history(self):
        """Test resetting prediction history."""
        container = PolicyContainer(model=dummy_model)
        
        container.predict("test input")
        assert len(container.prediction_history) == 1
        
        container.reset_history()
        assert len(container.prediction_history) == 0
    
    def test_update_config(self):
        """Test updating configuration."""
        container = PolicyContainer()
        
        new_config = PolicyConfig(
            threshold=0.3,
            enable_logging=False
        )
        container.update_config(new_config)
        
        assert container.config.threshold == 0.3
        assert container.config.enable_logging is False
    
    def test_multiple_predictions_tracking(self):
        """Test that multiple predictions are tracked."""
        container = PolicyContainer(model=dummy_model)
        
        for i in range(10):
            container.predict(f"input_{i}")
        
        assert len(container.prediction_history) == 10
        
        stats = container.get_stats()
        assert stats['total_predictions'] == 10
