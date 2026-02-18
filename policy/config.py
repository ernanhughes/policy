"""Configuration management for policy package."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class PolicyConfig:
    """Configuration for policy governance container.
    
    Attributes:
        threshold: Decision boundary threshold for accepting predictions
        risk_tolerance: Maximum acceptable risk level (0.0 to 1.0)
        calibration_samples: Number of samples used for boundary calibration
        enable_logging: Whether to log policy decisions
        metadata: Additional configuration metadata
    """
    
    threshold: float = 0.5
    risk_tolerance: float = 0.1
    calibration_samples: int = 1000
    enable_logging: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.risk_tolerance <= 1.0:
            raise ValueError("risk_tolerance must be between 0.0 and 1.0")
        if self.calibration_samples < 1:
            raise ValueError("calibration_samples must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "threshold": self.threshold,
            "risk_tolerance": self.risk_tolerance,
            "calibration_samples": self.calibration_samples,
            "enable_logging": self.enable_logging,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PolicyConfig":
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            PolicyConfig instance
        """
        return cls(**config_dict)
