"""Tests for policy configuration."""

import pytest
from policy.config import PolicyConfig


def test_default_config():
    """Test default configuration values."""
    config = PolicyConfig()
    assert config.threshold == 0.5
    assert config.risk_tolerance == 0.1
    assert config.calibration_samples == 1000
    assert config.enable_logging is True
    assert config.metadata == {}


def test_custom_config():
    """Test custom configuration values."""
    config = PolicyConfig(
        threshold=0.3,
        risk_tolerance=0.05,
        calibration_samples=500,
        enable_logging=False,
        metadata={'key': 'value'}
    )
    assert config.threshold == 0.3
    assert config.risk_tolerance == 0.05
    assert config.calibration_samples == 500
    assert config.enable_logging is False
    assert config.metadata == {'key': 'value'}


def test_config_validation():
    """Test configuration validation."""
    # Invalid threshold
    with pytest.raises(ValueError, match="threshold must be between"):
        PolicyConfig(threshold=-0.1)
    
    with pytest.raises(ValueError, match="threshold must be between"):
        PolicyConfig(threshold=1.5)
    
    # Invalid risk_tolerance
    with pytest.raises(ValueError, match="risk_tolerance must be between"):
        PolicyConfig(risk_tolerance=-0.1)
    
    with pytest.raises(ValueError, match="risk_tolerance must be between"):
        PolicyConfig(risk_tolerance=1.5)
    
    # Invalid calibration_samples
    with pytest.raises(ValueError, match="calibration_samples must be positive"):
        PolicyConfig(calibration_samples=0)


def test_config_to_dict():
    """Test conversion to dictionary."""
    config = PolicyConfig(
        threshold=0.3,
        risk_tolerance=0.05,
        metadata={'test': 'data'}
    )
    config_dict = config.to_dict()
    
    assert config_dict['threshold'] == 0.3
    assert config_dict['risk_tolerance'] == 0.05
    assert config_dict['metadata'] == {'test': 'data'}


def test_config_from_dict():
    """Test creation from dictionary."""
    config_dict = {
        'threshold': 0.4,
        'risk_tolerance': 0.15,
        'calibration_samples': 2000,
        'enable_logging': False,
        'metadata': {'source': 'dict'}
    }
    config = PolicyConfig.from_dict(config_dict)
    
    assert config.threshold == 0.4
    assert config.risk_tolerance == 0.15
    assert config.calibration_samples == 2000
    assert config.enable_logging is False
    assert config.metadata == {'source': 'dict'}
