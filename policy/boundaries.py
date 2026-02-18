"""Decision boundaries for policy enforcement."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class DecisionBoundary(ABC):
    """Base class for decision boundaries."""
    
    @abstractmethod
    def evaluate(self, metric_value: float) -> bool:
        """Evaluate if a metric value passes the boundary.
        
        Args:
            metric_value: Metric value to evaluate
            
        Returns:
            True if value is acceptable (below risk threshold), False otherwise
        """
        pass
    
    @abstractmethod
    def get_threshold(self) -> float:
        """Get the current threshold value.
        
        Returns:
            Threshold value
        """
        pass


class CalibratedBoundary(DecisionBoundary):
    """Calibrated decision boundary based on statistical analysis.
    
    This boundary adjusts its threshold based on calibration data to maintain
    a specified false positive rate or risk tolerance.
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.5,
        target_fpr: float = 0.1,
        min_samples: int = 100
    ) -> None:
        """Initialize calibrated boundary.
        
        Args:
            initial_threshold: Initial threshold value (default: 0.5)
            target_fpr: Target false positive rate (default: 0.1)
            min_samples: Minimum samples needed for calibration (default: 100)
        """
        self.threshold = initial_threshold
        self.target_fpr = target_fpr
        self.min_samples = min_samples
        self.calibration_data: List[Tuple[float, bool]] = []
        self.is_calibrated = False
    
    def evaluate(self, metric_value: float) -> bool:
        """Evaluate if metric value passes the boundary.
        
        Args:
            metric_value: Metric value to evaluate
            
        Returns:
            True if acceptable (metric below threshold), False otherwise
        """
        return metric_value <= self.threshold
    
    def get_threshold(self) -> float:
        """Get current threshold value.
        
        Returns:
            Current threshold
        """
        return self.threshold
    
    def add_calibration_sample(self, metric_value: float, is_acceptable: bool) -> None:
        """Add a sample for calibration.
        
        Args:
            metric_value: Observed metric value
            is_acceptable: True if this sample represents acceptable behavior
        """
        self.calibration_data.append((metric_value, is_acceptable))
    
    def calibrate(self) -> bool:
        """Calibrate the boundary based on collected samples.
        
        Returns:
            True if calibration was successful, False otherwise
        """
        if len(self.calibration_data) < self.min_samples:
            return False
        
        # Separate acceptable and unacceptable samples
        acceptable_values = [v for v, label in self.calibration_data if label]
        unacceptable_values = [v for v, label in self.calibration_data if not label]
        
        if not acceptable_values:
            # No acceptable samples - use conservative threshold
            self.threshold = 0.0
            self.is_calibrated = True
            return True
        
        if not unacceptable_values:
            # All samples acceptable - use permissive threshold
            self.threshold = 1.0
            self.is_calibrated = True
            return True
        
        # Find threshold that achieves target FPR
        # FPR = false positives / (false positives + true negatives)
        # False positive: acceptable sample with metric > threshold
        
        # Sort all acceptable values
        sorted_acceptable = sorted(acceptable_values)
        
        # Find threshold at target FPR percentile
        fpr_index = int(len(sorted_acceptable) * (1.0 - self.target_fpr))
        fpr_index = max(0, min(fpr_index, len(sorted_acceptable) - 1))
        
        self.threshold = float(sorted_acceptable[fpr_index])
        self.is_calibrated = True
        
        return True
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get statistics about calibration.
        
        Returns:
            Dictionary with calibration statistics
        """
        if not self.calibration_data:
            return {
                'n_samples': 0,
                'is_calibrated': self.is_calibrated,
                'threshold': self.threshold,
            }
        
        acceptable_values = [v for v, label in self.calibration_data if label]
        unacceptable_values = [v for v, label in self.calibration_data if not label]
        
        stats = {
            'n_samples': len(self.calibration_data),
            'n_acceptable': len(acceptable_values),
            'n_unacceptable': len(unacceptable_values),
            'is_calibrated': self.is_calibrated,
            'threshold': self.threshold,
        }
        
        if acceptable_values:
            stats['acceptable_mean'] = float(np.mean(acceptable_values))
            stats['acceptable_std'] = float(np.std(acceptable_values))
        
        if unacceptable_values:
            stats['unacceptable_mean'] = float(np.mean(unacceptable_values))
            stats['unacceptable_std'] = float(np.std(unacceptable_values))
        
        return stats
    
    def reset_calibration(self) -> None:
        """Reset calibration data and state."""
        self.calibration_data.clear()
        self.is_calibrated = False


class FixedBoundary(DecisionBoundary):
    """Fixed decision boundary with a constant threshold."""
    
    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize fixed boundary.
        
        Args:
            threshold: Fixed threshold value (default: 0.5)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold
    
    def evaluate(self, metric_value: float) -> bool:
        """Evaluate if metric value passes the boundary.
        
        Args:
            metric_value: Metric value to evaluate
            
        Returns:
            True if acceptable (metric below threshold), False otherwise
        """
        return metric_value <= self.threshold
    
    def get_threshold(self) -> float:
        """Get current threshold value.
        
        Returns:
            Current threshold
        """
        return self.threshold
    
    def set_threshold(self, threshold: float) -> None:
        """Set new threshold value.
        
        Args:
            threshold: New threshold value
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold
