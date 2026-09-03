"""Classifier calibration module: improve probability estimates."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any


def calibrate_classifier(
    model: Pipeline,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    method: str = 'isotonic',
    random_state: int = 42
) -> Tuple[CalibratedClassifierCV, Dict[str, Any]]:
    """
    Fit calibration on validation set using stratified 2-fold CV.
    
    Args:
        model: fitted Pipeline with preprocessor and classifier
        x_validation: validation features (for calibration)
        y_validation: validation labels
        method: 'isotonic' or 'sigmoid'
        random_state: for reproducibility
    
    Returns:
        Tuple of (calibrated_model, metadata)
    """
    from sklearn.model_selection import StratifiedKFold
    
    # Use stratified 2-fold CV on validation set for calibration
    # This avoids the deprecated 'prefit' parameter
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state),
        n_jobs=-1
    )
    calibrated.fit(x_validation, y_validation)
    
    metadata = {
        'method': method,
        'n_samples': len(y_validation),
        'n_positive': int(y_validation.sum()),
        'n_negative': int((1 - y_validation).sum())
    }
    
    return calibrated, metadata


def reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Compute reliability diagram: predicted probability vs observed frequency.
    
    Args:
        y_true: binary labels
        y_proba: predicted probabilities for class 1
        n_bins: number of bins for binning
    
    Returns:
        DataFrame with columns: bin_start, bin_end, predicted_prob, observed_freq, count
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bins) - 1
    
    results = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_start = bins[i]
            bin_end = bins[i + 1]
            pred_prob = y_proba[mask].mean()
            obs_freq = y_true[mask].mean()
            count = int(mask.sum())
            results.append({
                'Bin': f'{bin_start:.2f}-{bin_end:.2f}',
                'Predicted_Probability': pred_prob,
                'Observed_Frequency': obs_freq,
                'Count': count
            })
    
    return pd.DataFrame(results)


def calibration_improvement_metrics(
    y_true: np.ndarray,
    y_proba_before: np.ndarray,
    y_proba_after: np.ndarray
) -> Dict[str, float]:
    """
    Compute calibration improvement metrics.
    
    Uses Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    def compute_ece(y_true, y_proba, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                conf = y_proba[mask].mean()
                acc = y_true[mask].mean()
                ece += (mask.sum() / len(y_true)) * abs(conf - acc)
        return ece
    
    def compute_mce(y_true, y_proba, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        
        mce = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                conf = y_proba[mask].mean()
                acc = y_true[mask].mean()
                mce = max(mce, abs(conf - acc))
        return mce
    
    return {
        'ECE_before': compute_ece(y_true, y_proba_before),
        'ECE_after': compute_ece(y_true, y_proba_after),
        'MCE_before': compute_mce(y_true, y_proba_before),
        'MCE_after': compute_mce(y_true, y_proba_after),
    }
