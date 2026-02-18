"""Tests for policy metrics."""

import pytest
import numpy as np
from policy.metrics import HallucinationMetric, RiskMetric


class TestHallucinationMetric:
    """Tests for HallucinationMetric."""
    
    def test_initialization(self):
        """Test metric initialization."""
        metric = HallucinationMetric()
        assert metric.temperature == 1.0
        
        metric = HallucinationMetric(temperature=0.5)
        assert metric.temperature == 0.5
    
    def test_compute_with_uniform_distribution(self):
        """Test with uniform distribution (high uncertainty)."""
        metric = HallucinationMetric()
        probs = [0.25, 0.25, 0.25, 0.25]
        energy = metric.compute(probs)
        
        # Uniform distribution should have high energy (near 1.0)
        assert 0.9 <= energy <= 1.0
    
    def test_compute_with_peaked_distribution(self):
        """Test with peaked distribution (low uncertainty)."""
        metric = HallucinationMetric()
        probs = [0.95, 0.02, 0.02, 0.01]
        energy = metric.compute(probs)
        
        # Peaked distribution should have low energy
        assert 0.0 <= energy <= 0.3
    
    def test_compute_with_context(self):
        """Test with additional context."""
        metric = HallucinationMetric()
        probs = [0.6, 0.3, 0.1]
        context = {'probabilities': probs}
        energy = metric.compute(None, context)
        
        assert 0.0 <= energy <= 1.0
    
    def test_compute_with_samples(self):
        """Test with multiple samples for consistency check."""
        metric = HallucinationMetric()
        
        # Consistent samples (low energy)
        samples = np.array([
            [0.7, 0.2, 0.1],
            [0.75, 0.15, 0.1],
            [0.72, 0.18, 0.1]
        ])
        energy = metric.compute(samples)
        assert 0.0 <= energy <= 0.5
        
        # Inconsistent samples (high energy)
        samples = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        energy = metric.compute(samples)
        assert 0.5 <= energy <= 1.0
    
    def test_compute_with_zero_sum(self):
        """Test with zero-sum probabilities."""
        metric = HallucinationMetric()
        probs = [0.0, 0.0, 0.0]
        energy = metric.compute(probs)
        
        # Zero sum should return maximum energy
        assert energy == 1.0
    
    def test_temperature_scaling(self):
        """Test temperature parameter effect."""
        probs = [0.6, 0.3, 0.1]
        
        metric_low = HallucinationMetric(temperature=0.5)
        metric_high = HallucinationMetric(temperature=2.0)
        
        energy_low = metric_low.compute(probs)
        energy_high = metric_high.compute(probs)
        
        # Higher temperature should give higher energy (more sensitive)
        assert energy_high > energy_low


class TestRiskMetric:
    """Tests for RiskMetric."""
    
    def test_initialization(self):
        """Test metric initialization."""
        metric = RiskMetric()
        assert 'uncertainty' in metric.weights
        assert 'confidence' in metric.weights
        assert 'consistency' in metric.weights
    
    def test_custom_weights(self):
        """Test with custom weights."""
        weights = {'uncertainty': 0.5, 'confidence': 0.3, 'consistency': 0.2}
        metric = RiskMetric(weights=weights)
        assert metric.weights == weights
    
    def test_compute_with_probabilities(self):
        """Test risk computation with probability distribution."""
        metric = RiskMetric()
        probs = [0.7, 0.2, 0.1]
        risk = metric.compute(probs)
        
        assert 0.0 <= risk <= 1.0
    
    def test_compute_with_context(self):
        """Test risk computation with context."""
        metric = RiskMetric()
        probs = [0.8, 0.15, 0.05]
        context = {
            'confidence': 0.8,
            'uncertainty': 0.2,
            'inconsistency': 0.1
        }
        risk = metric.compute(probs, context)
        
        assert 0.0 <= risk <= 1.0
    
    def test_high_confidence_low_risk(self):
        """Test that high confidence leads to low risk."""
        metric = RiskMetric()
        probs = [0.95, 0.03, 0.02]
        context = {'confidence': 0.95}
        risk = metric.compute(probs, context)
        
        # High confidence should give relatively low risk
        assert risk < 0.5
    
    def test_low_confidence_high_risk(self):
        """Test that low confidence leads to high risk."""
        metric = RiskMetric()
        probs = [0.4, 0.35, 0.25]
        context = {'confidence': 0.4}
        risk = metric.compute(probs, context)
        
        # Low confidence should give relatively high risk
        assert risk > 0.4
