"""Tests for decision boundaries."""

import pytest
from policy.boundaries import FixedBoundary, CalibratedBoundary


class TestFixedBoundary:
    """Tests for FixedBoundary."""
    
    def test_initialization(self):
        """Test boundary initialization."""
        boundary = FixedBoundary()
        assert boundary.threshold == 0.5
        
        boundary = FixedBoundary(threshold=0.3)
        assert boundary.threshold == 0.3
    
    def test_invalid_threshold(self):
        """Test invalid threshold values."""
        with pytest.raises(ValueError, match="threshold must be between"):
            FixedBoundary(threshold=-0.1)
        
        with pytest.raises(ValueError, match="threshold must be between"):
            FixedBoundary(threshold=1.5)
    
    def test_evaluate(self):
        """Test boundary evaluation."""
        boundary = FixedBoundary(threshold=0.5)
        
        assert boundary.evaluate(0.3) is True  # Below threshold
        assert boundary.evaluate(0.5) is True  # At threshold
        assert boundary.evaluate(0.7) is False  # Above threshold
    
    def test_get_threshold(self):
        """Test getting threshold."""
        boundary = FixedBoundary(threshold=0.4)
        assert boundary.get_threshold() == 0.4
    
    def test_set_threshold(self):
        """Test setting new threshold."""
        boundary = FixedBoundary(threshold=0.5)
        boundary.set_threshold(0.3)
        assert boundary.threshold == 0.3
        
        with pytest.raises(ValueError):
            boundary.set_threshold(1.5)


class TestCalibratedBoundary:
    """Tests for CalibratedBoundary."""
    
    def test_initialization(self):
        """Test boundary initialization."""
        boundary = CalibratedBoundary()
        assert boundary.threshold == 0.5
        assert boundary.target_fpr == 0.1
        assert boundary.min_samples == 100
        assert boundary.is_calibrated is False
    
    def test_custom_initialization(self):
        """Test with custom parameters."""
        boundary = CalibratedBoundary(
            initial_threshold=0.3,
            target_fpr=0.05,
            min_samples=50
        )
        assert boundary.threshold == 0.3
        assert boundary.target_fpr == 0.05
        assert boundary.min_samples == 50
    
    def test_evaluate(self):
        """Test boundary evaluation."""
        boundary = CalibratedBoundary(initial_threshold=0.5)
        
        assert boundary.evaluate(0.3) is True
        assert boundary.evaluate(0.5) is True
        assert boundary.evaluate(0.7) is False
    
    def test_add_calibration_sample(self):
        """Test adding calibration samples."""
        boundary = CalibratedBoundary()
        
        boundary.add_calibration_sample(0.3, True)
        boundary.add_calibration_sample(0.7, False)
        
        assert len(boundary.calibration_data) == 2
    
    def test_calibrate_insufficient_samples(self):
        """Test calibration with insufficient samples."""
        boundary = CalibratedBoundary(min_samples=10)
        
        for i in range(5):
            boundary.add_calibration_sample(0.3 + i * 0.1, True)
        
        success = boundary.calibrate()
        assert success is False
        assert boundary.is_calibrated is False
    
    def test_calibrate_success(self):
        """Test successful calibration."""
        boundary = CalibratedBoundary(min_samples=50, target_fpr=0.1)
        
        # Add acceptable samples (low metric values)
        for i in range(50):
            metric_value = 0.2 + (i / 100.0)  # 0.2 to 0.7
            boundary.add_calibration_sample(metric_value, True)
        
        # Add unacceptable samples (high metric values)
        for i in range(30):
            metric_value = 0.7 + (i / 50.0)  # 0.7 to 1.3
            boundary.add_calibration_sample(metric_value, False)
        
        success = boundary.calibrate()
        assert success is True
        assert boundary.is_calibrated is True
        
        # Threshold should be set to achieve target FPR
        assert 0.0 <= boundary.threshold <= 1.0
    
    def test_calibrate_all_acceptable(self):
        """Test calibration with all acceptable samples."""
        boundary = CalibratedBoundary(min_samples=10)
        
        for i in range(20):
            boundary.add_calibration_sample(0.3 + i * 0.01, True)
        
        success = boundary.calibrate()
        assert success is True
        assert boundary.threshold == 1.0  # All acceptable -> permissive threshold
    
    def test_calibrate_all_unacceptable(self):
        """Test calibration with all unacceptable samples."""
        boundary = CalibratedBoundary(min_samples=10)
        
        for i in range(20):
            boundary.add_calibration_sample(0.7 + i * 0.01, False)
        
        success = boundary.calibrate()
        assert success is True
        assert boundary.threshold == 0.0  # All unacceptable -> conservative threshold
    
    def test_get_calibration_stats(self):
        """Test getting calibration statistics."""
        boundary = CalibratedBoundary(min_samples=10)
        
        # Empty stats
        stats = boundary.get_calibration_stats()
        assert stats['n_samples'] == 0
        assert stats['is_calibrated'] is False
        
        # Add samples
        for i in range(15):
            boundary.add_calibration_sample(0.3, True)
        for i in range(5):
            boundary.add_calibration_sample(0.8, False)
        
        boundary.calibrate()
        stats = boundary.get_calibration_stats()
        
        assert stats['n_samples'] == 20
        assert stats['n_acceptable'] == 15
        assert stats['n_unacceptable'] == 5
        assert stats['is_calibrated'] is True
        assert 'acceptable_mean' in stats
        assert 'unacceptable_mean' in stats
    
    def test_reset_calibration(self):
        """Test resetting calibration."""
        boundary = CalibratedBoundary(min_samples=5)
        
        for i in range(10):
            boundary.add_calibration_sample(0.3, True)
        
        boundary.calibrate()
        assert boundary.is_calibrated is True
        assert len(boundary.calibration_data) > 0
        
        boundary.reset_calibration()
        assert boundary.is_calibrated is False
        assert len(boundary.calibration_data) == 0
