"""Data diagnostics module: detect signal quality and feature importance."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from typing import Tuple, Dict, Any


def compute_class_means(data: pd.DataFrame, numeric_features: list, target_col: str = 'target') -> pd.DataFrame:
    """Compute mean values for each class on numeric features."""
    return data.groupby(target_col)[numeric_features].mean()


def compute_mutual_information(
    data: pd.DataFrame,
    numeric_features: list,
    target_col: str = 'target',
    random_state: int = 42
) -> pd.DataFrame:
    """Compute mutual information between each feature and target."""
    x = data[numeric_features].fillna(data[numeric_features].median())
    y = data[target_col]
    mi = mutual_info_classif(x, y, random_state=random_state)
    return pd.DataFrame({
        'Feature': numeric_features,
        'Mutual_Information': mi
    }).sort_values('Mutual_Information', ascending=False)


def diagnose_signal_quality(
    roc_auc: float,
    threshold: float = 0.55,
    warning_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Diagnose signal quality based on ROC-AUC.
    
    Returns:
        dict with keys: has_signal (bool), roc_auc (float), warning (str or None)
    """
    if roc_auc <= warning_threshold:
        return {
            'has_signal': False,
            'roc_auc': roc_auc,
            'warning': (
                f"This dataset shows near-random separation (ROC-AUC ≈ {roc_auc:.3f}, expected ≥ 0.55). "
                "Label provenance should be verified before treating any downstream metric as meaningful. "
                "The model may be learning only noise or random variation in the data."
            )
        }
    elif roc_auc < threshold:
        return {
            'has_signal': True,
            'roc_auc': roc_auc,
            'warning': (
                f"Signal is weak (ROC-AUC = {roc_auc:.3f}, expected ≥ {threshold}). "
                "Model performance may not generalize well. Consider verifying label quality or collecting more data."
            )
        }
    else:
        return {
            'has_signal': True,
            'roc_auc': roc_auc,
            'warning': None
        }


def check_feature_separation(class_means: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    """
    Check whether class means are meaningfully different for each feature.
    Returns a dataframe with absolute difference and percent difference.
    """
    if len(class_means) != 2:
        return pd.DataFrame()
    
    row_0, row_1 = class_means.iloc[0], class_means.iloc[1]
    abs_diff = (row_1 - row_0).abs()
    pct_diff = ((row_1 - row_0).abs() / row_0.abs().replace(0, 1)) * 100
    
    return pd.DataFrame({
        'Feature': numeric_features,
        'Class_0_Mean': row_0.values,
        'Class_1_Mean': row_1.values,
        'Absolute_Difference': abs_diff.values,
        'Percent_Difference': pct_diff.values
    }).sort_values('Absolute_Difference', ascending=False)


def label_balance_info(data: pd.DataFrame, target_col: str = 'target') -> Dict[str, Any]:
    """Return label balance statistics."""
    counts = data[target_col].value_counts().to_dict()
    total = len(data)
    return {
        'label_0_count': int(counts.get(0, 0)),
        'label_1_count': int(counts.get(1, 0)),
        'total': total,
        'label_1_rate': float(data[target_col].mean()),
        'is_balanced': 0.4 <= data[target_col].mean() <= 0.6
    }
