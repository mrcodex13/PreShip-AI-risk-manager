"""SHAP explainability module: feature contribution analysis."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import warnings


def _get_pipeline_parts(model):
    """Return the preprocessor and classifier from a pipeline or calibrated wrapper."""
    if hasattr(model, "named_steps"):
        pipeline = model
    else:
        calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
        if not calibrated_classifiers:
            raise TypeError("Expected a fitted model pipeline or calibrated pipeline.")
        pipeline = getattr(calibrated_classifiers[0], "estimator", None)
        if pipeline is None or not hasattr(pipeline, "named_steps"):
            raise TypeError("Calibrated model does not contain a fitted pipeline.")

    return pipeline.named_steps["preprocessor"], pipeline.named_steps["classifier"]


def safe_import_shap():
    """Safely import SHAP with graceful fallback."""
    try:
        import shap
        return shap
    except ImportError:
        return None


def compute_shap_values(
    model,
    x_sample: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    background_data: Optional[pd.DataFrame] = None
) -> Optional[Tuple[np.ndarray, pd.DataFrame]]:
    """
    Compute SHAP values for a sample.
    
    Args:
        model: fitted classifier model (Pipeline or compatible)
        x_sample: features for which to compute SHAP values
        feature_names: list of feature names
        background_data: data for SHAP background (if using TreeExplainer)
    
    Returns:
        (shap_values, feature_df) or (None, None) if SHAP unavailable
    """
    shap = safe_import_shap()
    if shap is None:
        warnings.warn("SHAP not available. Install via 'pip install shap'.")
        return None, None
    
    try:
        # Try LinearExplainer for LogisticRegression
        preprocessor, classifier = _get_pipeline_parts(model)
        transformed = preprocessor.transform(x_sample)
        explainer = shap.LinearExplainer(
            classifier,
            transformed
        )
        shap_vals = explainer.shap_values(transformed)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # Use class 1 (high risk)
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(shap_vals.shape[1])]
        
        # Convert to DataFrame for easier handling
        shap_df = pd.DataFrame(
            shap_vals,
            columns=feature_names
        )
        
        return shap_vals, shap_df
    
    except Exception as e:
        warnings.warn(f"SHAP computation failed: {str(e)}")
        return None, None


def top_shap_contributors(
    shap_values: Optional[np.ndarray],
    feature_names: List[str],
    base_value: Optional[float] = None,
    top_n: int = 5
) -> Optional[pd.DataFrame]:
    """
    Extract top N SHAP value contributors.
    
    Args:
        shap_values: array of SHAP values (1D or 2D for multiple samples)
        feature_names: feature names
        base_value: base value / expected value
        top_n: number of top contributors to return
    
    Returns:
        DataFrame with columns: Feature, SHAP_Value, Abs_SHAP_Value, Direction
    """
    if shap_values is None:
        return None
    
    # Handle 2D case (multiple samples) by taking the first
    if len(shap_values.shape) == 2:
        shap_values = shap_values[0]
    
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-top_n:][::-1]
    
    result = []
    for idx in top_indices:
        result.append({
            'Feature': feature_names[idx],
            'SHAP_Value': float(shap_values[idx]),
            'Abs_SHAP_Value': float(abs_shap[idx]),
            'Direction': 'Risk ↑' if shap_values[idx] > 0 else 'Risk ↓'
        })
    
    return pd.DataFrame(result)


def shap_summary_text(shap_df: pd.DataFrame, base_value: float = 0.5) -> str:
    """
    Generate a text summary of SHAP contributions.
    
    Useful for debugging or logging.
    """
    if shap_df is None or len(shap_df) == 0:
        return "No SHAP contributions available."
    
    lines = [f"Base value (expected model output): {base_value:.3f}"]
    lines.append("\nTop contributors:")
    
    for _, row in shap_df.head(5).iterrows():
        lines.append(
            f"  {row['Feature']}: {row['SHAP_Value']:+.3f} ({row['Direction']})"
        )
    
    return "\n".join(lines)
