"""Conformal prediction module: uncertainty quantification."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional


def nonconformity_scores(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    """
    Compute nonconformity scores for split conformal prediction.
    
    Nonconformity score = |y_true - y_proba| for class 1.
    Higher score = more uncertainty about the prediction.
    """
    return np.abs(y_true - y_proba)


def conformal_confidence_interval(
    y_proba_test: np.ndarray,
    nonconformity_calib: np.ndarray,
    alpha: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute split conformal prediction intervals.
    
    Args:
        y_proba_test: predicted probabilities on test set
        nonconformity_calib: nonconformity scores on calibration set
        alpha: miscoverage level (1-alpha = confidence)
    
    Returns:
        (lower_bounds, upper_bounds) for prediction intervals
    """
    n = len(nonconformity_calib)
    # Quantile adjusted for finite sample size
    q_index = int(np.ceil((n + 1) * (1 - alpha) / n))
    q_index = min(q_index, len(nonconformity_calib) - 1)
    sorted_nc = np.sort(nonconformity_calib)
    threshold = sorted_nc[q_index]
    
    lower_bounds = np.clip(y_proba_test - threshold, 0, 1)
    upper_bounds = np.clip(y_proba_test + threshold, 0, 1)
    
    return lower_bounds, upper_bounds


def confidence_level_for_prediction(
    y_proba: float,
    y_proba_lower: float,
    y_proba_upper: float,
    risk_threshold: float = 0.5
) -> float:
    """
    Compute confidence level for a high-risk prediction.
    
    Measures how far the prediction is from the threshold within the confidence interval.
    Returns a value 0-1 indicating confidence.
    """
    interval_width = y_proba_upper - y_proba_lower
    
    if interval_width < 1e-6:
        # Degenerate interval
        return 1.0 if abs(y_proba - risk_threshold) > 0.1 else 0.5
    
    # Distance from threshold, normalized by interval width
    distance_from_threshold = abs(y_proba - risk_threshold)
    confidence = min(1.0, distance_from_threshold / interval_width)
    
    return confidence


def confidence_label(confidence: float, risk_band: str) -> str:
    """Generate a natural language confidence label."""
    if confidence >= 0.8:
        return f"{risk_band} — {confidence:.0%} confidence (strong signal)"
    elif confidence >= 0.6:
        return f"{risk_band} — {confidence:.0%} confidence (moderate signal)"
    else:
        return f"{risk_band} — {confidence:.0%} confidence (borderline, soft check recommended)"


def recommended_action_with_confidence(
    risk_score: float,
    risk_band: str,
    confidence: float,
    threshold: float
) -> str:
    """
    Generate recommended action based on both risk score and confidence.
    """
    is_high_risk = risk_score >= threshold
    
    if not is_high_risk:
        return "Proceed, monitor outcome"
    
    # High-risk with varying confidence
    if confidence >= 0.75:
        return "Manual review / verify (high confidence signal)"
    elif confidence >= 0.50:
        return "Manual review / verify (moderate confidence)"
    else:
        return "Lightweight check / manual review (borderline case — recommend OTP or address verification)"
