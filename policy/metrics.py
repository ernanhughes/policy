"""Metrics for AI governance including hallucination energy and risk assessment."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import entropy


class BaseMetric(ABC):
    """Base class for policy metrics."""
    
    @abstractmethod
    def compute(self, predictions: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """Compute metric value.
        
        Args:
            predictions: Model predictions to evaluate
            context: Optional context information
            
        Returns:
            Metric value (higher values indicate higher risk)
        """
        pass


class HallucinationMetric(BaseMetric):
    """Metric for estimating hallucination energy in AI predictions.
    
    Hallucination energy measures the model's uncertainty and potential
    for generating unreliable outputs. It combines entropy-based uncertainty
    with consistency checks.
    """
    
    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize hallucination metric.
        
        Args:
            temperature: Temperature parameter for scaling entropy (default: 1.0)
        """
        self.temperature = temperature
    
    def compute(self, predictions: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """Compute hallucination energy.
        
        Args:
            predictions: Model predictions (probability distribution or logits)
            context: Optional context with keys:
                - 'logits': Raw logit values
                - 'probabilities': Probability distribution
                - 'samples': Multiple prediction samples for consistency check
            
        Returns:
            Hallucination energy score (0.0 to 1.0, higher means more risk)
        """
        if context is None:
            context = {}
        
        # Handle different input formats
        if isinstance(predictions, (list, np.ndarray)):
            probs = np.array(predictions)
            if probs.ndim == 1:
                # Single prediction - normalize if needed
                if np.sum(probs) > 0:
                    probs = probs / np.sum(probs)
                else:
                    # No valid prediction
                    return 1.0
            else:
                # Multiple samples - compute consistency
                return self._compute_with_samples(probs)
        
        # Use probabilities from context if available
        probs = context.get('probabilities', predictions)
        if isinstance(probs, (list, np.ndarray)):
            probs = np.array(probs)
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
            else:
                return 1.0
        
        # Compute entropy-based hallucination energy
        h_energy = self._compute_entropy_energy(probs)
        
        # Add sample consistency if available
        samples = context.get('samples')
        if samples is not None:
            consistency = self._compute_consistency(samples)
            h_energy = 0.7 * h_energy + 0.3 * (1.0 - consistency)
        
        return float(np.clip(h_energy, 0.0, 1.0))
    
    def _compute_entropy_energy(self, probs: np.ndarray) -> float:
        """Compute entropy-based energy.
        
        Args:
            probs: Probability distribution
            
        Returns:
            Normalized entropy value
        """
        if len(probs) == 0:
            return 1.0
        
        # Compute entropy
        h = entropy(probs + 1e-10)  # Add small constant for numerical stability
        
        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log(len(probs))
        if max_entropy > 0:
            normalized_h = h / max_entropy
        else:
            normalized_h = 0.0
        
        # Apply temperature scaling
        scaled_h = normalized_h ** (1.0 / self.temperature)
        
        return float(scaled_h)
    
    def _compute_consistency(self, samples: List[Any]) -> float:
        """Compute consistency across multiple samples.
        
        Args:
            samples: List of prediction samples
            
        Returns:
            Consistency score (0.0 to 1.0, higher means more consistent)
        """
        if len(samples) < 2:
            return 1.0
        
        # Convert samples to arrays
        sample_arrays = [np.array(s) for s in samples]
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(sample_arrays)):
            for j in range(i + 1, len(sample_arrays)):
                # Cosine similarity
                s1, s2 = sample_arrays[i], sample_arrays[j]
                if len(s1) > 0 and len(s2) > 0:
                    norm1, norm2 = np.linalg.norm(s1), np.linalg.norm(s2)
                    if norm1 > 0 and norm2 > 0:
                        sim = np.dot(s1, s2) / (norm1 * norm2)
                        similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        return float(np.mean(similarities))
    
    def _compute_with_samples(self, samples: np.ndarray) -> float:
        """Compute hallucination energy from multiple samples.
        
        Args:
            samples: Array of shape (n_samples, n_classes)
            
        Returns:
            Hallucination energy score
        """
        # Compute average prediction
        avg_probs = np.mean(samples, axis=0)
        if np.sum(avg_probs) > 0:
            avg_probs = avg_probs / np.sum(avg_probs)
        else:
            return 1.0
        
        # Compute entropy of average
        h_energy = self._compute_entropy_energy(avg_probs)
        
        # Compute consistency
        consistency = self._compute_consistency(list(samples))
        
        # Combine both metrics
        combined = 0.7 * h_energy + 0.3 * (1.0 - consistency)
        
        return float(np.clip(combined, 0.0, 1.0))


class RiskMetric(BaseMetric):
    """General risk metric for AI predictions.
    
    Combines multiple risk factors into a single score.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        """Initialize risk metric.
        
        Args:
            weights: Weights for different risk components
        """
        self.weights = weights or {
            'uncertainty': 0.4,
            'confidence': 0.3,
            'consistency': 0.3,
        }
    
    def compute(self, predictions: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """Compute overall risk score.
        
        Args:
            predictions: Model predictions
            context: Optional context with risk-related information
            
        Returns:
            Risk score (0.0 to 1.0, higher means more risk)
        """
        if context is None:
            context = {}
        
        risk_components = {}
        
        # Uncertainty component
        if isinstance(predictions, (list, np.ndarray)):
            probs = np.array(predictions)
            if probs.ndim == 1 and np.sum(probs) > 0:
                probs = probs / np.sum(probs)
                # High entropy = high uncertainty = high risk
                h = entropy(probs + 1e-10)
                max_h = np.log(len(probs))
                risk_components['uncertainty'] = h / max_h if max_h > 0 else 0.0
            else:
                risk_components['uncertainty'] = 1.0
        else:
            risk_components['uncertainty'] = context.get('uncertainty', 0.5)
        
        # Confidence component (low confidence = high risk)
        confidence = context.get('confidence', np.max(predictions) if isinstance(predictions, (list, np.ndarray)) else 0.5)
        risk_components['confidence'] = 1.0 - float(confidence)
        
        # Consistency component
        risk_components['consistency'] = context.get('inconsistency', 0.0)
        
        # Weighted sum
        total_risk = sum(
            self.weights.get(k, 0.0) * v
            for k, v in risk_components.items()
        )
        
        return float(np.clip(total_risk, 0.0, 1.0))
