"""Main policy container for AI governance."""

from typing import Any, Callable, Dict, Optional, List, Tuple
import logging
from dataclasses import dataclass

from .config import PolicyConfig
from .metrics import BaseMetric, HallucinationMetric, RiskMetric
from .boundaries import DecisionBoundary, CalibratedBoundary


@dataclass
class PolicyDecision:
    """Result of a policy evaluation.
    
    Attributes:
        allowed: Whether the prediction is allowed
        metric_value: Computed metric value
        threshold: Threshold used for decision
        metadata: Additional decision metadata
    """
    
    allowed: bool
    metric_value: float
    threshold: float
    metadata: Dict[str, Any]


class PolicyContainer:
    """Model-agnostic AI governance container.
    
    Wraps any AI system and enforces calibrated decision boundaries based on
    hallucination energy or other risk metrics.
    
    Example:
        >>> # Wrap an AI model
        >>> policy = PolicyContainer(
        ...     model=my_model,
        ...     metric=HallucinationMetric(),
        ...     config=PolicyConfig(threshold=0.3)
        ... )
        >>> 
        >>> # Make governed prediction
        >>> result = policy.predict(input_data)
        >>> if result.allowed:
        ...     print(f"Prediction: {result.metadata['prediction']}")
        ... else:
        ...     print("Prediction rejected due to high risk")
    """
    
    def __init__(
        self,
        model: Optional[Callable] = None,
        metric: Optional[BaseMetric] = None,
        boundary: Optional[DecisionBoundary] = None,
        config: Optional[PolicyConfig] = None,
    ) -> None:
        """Initialize policy container.
        
        Args:
            model: AI model or prediction function to wrap (optional)
            metric: Metric for evaluating predictions (default: HallucinationMetric)
            boundary: Decision boundary (default: CalibratedBoundary)
            config: Policy configuration (default: PolicyConfig())
        """
        self.model = model
        self.metric = metric or HallucinationMetric()
        self.config = config or PolicyConfig()
        
        # Initialize boundary
        if boundary is None:
            self.boundary = CalibratedBoundary(
                initial_threshold=self.config.threshold,
                target_fpr=self.config.risk_tolerance,
                min_samples=self.config.calibration_samples
            )
        else:
            self.boundary = boundary
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if self.config.enable_logging:
            self.logger.setLevel(logging.INFO)
        
        # Track predictions for monitoring
        self.prediction_history: List[PolicyDecision] = []
    
    def predict(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
        return_raw: bool = False
    ) -> PolicyDecision:
        """Make a governed prediction.
        
        Args:
            input_data: Input to the AI model
            context: Optional context for metric computation
            return_raw: If True, include raw model output in metadata
            
        Returns:
            PolicyDecision with allowed flag and metadata
        """
        if self.model is None:
            raise ValueError("No model configured. Set model or use evaluate() directly.")
        
        # Get model prediction
        raw_prediction = self.model(input_data)
        
        # Evaluate with policy
        decision = self.evaluate(
            predictions=raw_prediction,
            context=context
        )
        
        # Add prediction to metadata
        if return_raw:
            decision.metadata['raw_prediction'] = raw_prediction
        decision.metadata['prediction'] = raw_prediction
        
        # Log decision
        if self.config.enable_logging:
            self.logger.info(
                f"Policy decision: allowed={decision.allowed}, "
                f"metric={decision.metric_value:.4f}, "
                f"threshold={decision.threshold:.4f}"
            )
        
        # Store in history
        self.prediction_history.append(decision)
        
        return decision
    
    def evaluate(
        self,
        predictions: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> PolicyDecision:
        """Evaluate predictions against policy without running model.
        
        Args:
            predictions: Model predictions to evaluate
            context: Optional context for metric computation
            
        Returns:
            PolicyDecision with allowed flag and metadata
        """
        # Compute metric
        metric_value = self.metric.compute(predictions, context)
        
        # Evaluate against boundary
        allowed = self.boundary.evaluate(metric_value)
        threshold = self.boundary.get_threshold()
        
        # Create decision
        decision = PolicyDecision(
            allowed=allowed,
            metric_value=metric_value,
            threshold=threshold,
            metadata={
                'metric_type': type(self.metric).__name__,
                'boundary_type': type(self.boundary).__name__,
            }
        )
        
        return decision
    
    def calibrate(
        self,
        calibration_data: List[Tuple[Any, bool]],
        auto_apply: bool = True
    ) -> bool:
        """Calibrate decision boundary using labeled data.
        
        Args:
            calibration_data: List of (predictions, is_acceptable) tuples
            auto_apply: If True, automatically apply calibration (default: True)
            
        Returns:
            True if calibration was successful
        """
        if not isinstance(self.boundary, CalibratedBoundary):
            self.logger.warning("Boundary is not calibratable")
            return False
        
        # Add samples to boundary
        for predictions, is_acceptable in calibration_data:
            metric_value = self.metric.compute(predictions)
            self.boundary.add_calibration_sample(metric_value, is_acceptable)
        
        # Perform calibration
        if auto_apply:
            success = self.boundary.calibrate()
            if success and self.config.enable_logging:
                stats = self.boundary.get_calibration_stats()
                self.logger.info(f"Calibration successful: {stats}")
            return success
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about policy decisions.
        
        Returns:
            Dictionary with policy statistics
        """
        if not self.prediction_history:
            return {
                'total_predictions': 0,
                'allowed': 0,
                'rejected': 0,
                'allow_rate': 0.0,
            }
        
        allowed_count = sum(1 for d in self.prediction_history if d.allowed)
        total = len(self.prediction_history)
        
        stats = {
            'total_predictions': total,
            'allowed': allowed_count,
            'rejected': total - allowed_count,
            'allow_rate': allowed_count / total,
            'current_threshold': self.boundary.get_threshold(),
        }
        
        # Add calibration stats if available
        if isinstance(self.boundary, CalibratedBoundary):
            stats['calibration'] = self.boundary.get_calibration_stats()
        
        return stats
    
    def reset_history(self) -> None:
        """Reset prediction history."""
        self.prediction_history.clear()
    
    def update_config(self, config: PolicyConfig) -> None:
        """Update policy configuration.
        
        Args:
            config: New configuration
        """
        self.config = config
        
        # Update logging
        if self.config.enable_logging:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
