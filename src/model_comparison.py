"""Model comparison module: gradient boosted trees vs Logistic Regression."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from typing import Dict, Tuple, Optional, Any
import warnings


def safe_import_lightgbm():
    """Safely import LightGBM."""
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        return None


def safe_import_xgboost():
    """Safely import XGBoost."""
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        return None


def train_tree_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    model_type: str = 'lightgbm',
    random_state: int = 42,
    **kwargs
) -> Optional[Any]:
    """
    Train a tree-based classifier (LightGBM or XGBoost).
    
    Args:
        x_train: training features (preprocessed)
        y_train: training labels
        model_type: 'lightgbm' or 'xgboost'
        random_state: for reproducibility
        **kwargs: additional model parameters
    
    Returns:
        Trained model or None if library not available
    """
    if model_type == 'lightgbm':
        lgb = safe_import_lightgbm()
        if lgb is None:
            warnings.warn("LightGBM not available. Install via 'pip install lightgbm'.")
            return None
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'random_state': random_state,
            'verbose': -1,
            **kwargs
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(x_train, y_train)
        return model
    
    elif model_type == 'xgboost':
        xgb = safe_import_xgboost()
        if xgb is None:
            warnings.warn("XGBoost not available. Install via 'pip install xgboost'.")
            return None
        
        params = {
            'objective': 'binary:logistic',
            'random_state': random_state,
            'eval_metric': 'auc',
            **kwargs
        }
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train)
        return model
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compare_models(
    model_lr: Pipeline,
    model_tree: Optional[Any],
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    tree_model_type: str = 'lightgbm'
) -> pd.DataFrame:
    """
    Compare Logistic Regression vs Tree model on test set.
    
    Args:
        model_lr: trained LogisticRegression pipeline
        model_tree: trained tree model (or None if unavailable)
        x_test: test features
        y_test: test labels
        tree_model_type: for display purposes
    
    Returns:
        DataFrame with columns: Metric, LogisticRegression, TreeModel
    """
    results = []
    
    # Get LR predictions
    lr_proba = model_lr.predict_proba(x_test)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)
    
    metrics_to_compute = ['Precision', 'Recall', 'AUC-PR', 'ROC-AUC']
    
    for metric in metrics_to_compute:
        lr_value = None
        tree_value = None
        
        if metric == 'Precision':
            lr_value = precision_score(y_test, lr_pred, zero_division=0)
            if model_tree is not None:
                tree_pred = (model_tree.predict_proba(x_test)[:, 1] >= 0.5).astype(int)
                tree_value = precision_score(y_test, tree_pred, zero_division=0)
        
        elif metric == 'Recall':
            lr_value = recall_score(y_test, lr_pred, zero_division=0)
            if model_tree is not None:
                tree_pred = (model_tree.predict_proba(x_test)[:, 1] >= 0.5).astype(int)
                tree_value = recall_score(y_test, tree_pred, zero_division=0)
        
        elif metric == 'AUC-PR':
            lr_value = average_precision_score(y_test, lr_proba)
            if model_tree is not None:
                tree_proba = model_tree.predict_proba(x_test)[:, 1]
                tree_value = average_precision_score(y_test, tree_proba)
        
        elif metric == 'ROC-AUC':
            lr_value = roc_auc_score(y_test, lr_proba)
            if model_tree is not None:
                tree_proba = model_tree.predict_proba(x_test)[:, 1]
                tree_value = roc_auc_score(y_test, tree_proba)
        
        results.append({
            'Metric': metric,
            'Logistic Regression': f"{lr_value:.3f}" if lr_value is not None else "N/A",
            tree_model_type.capitalize(): f"{tree_value:.3f}" if tree_value is not None else "N/A"
        })
    
    return pd.DataFrame(results)


def get_tree_model_warning() -> str:
    """Return a disclaimer about tree model comparison."""
    return (
        "**⚠️ Model Comparison Disclaimer:**\n\n"
        "The tree model shown here is for comparative reference only. "
        "The primary deployed model remains Logistic Regression (LR) for these reasons:\n"
        "1. **Interpretability:** LR coefficients are directly interpretable; tree decisions are opaque.\n"
        "2. **Explainability:** SHAP values on LR are clearer for human verification workflows.\n"
        "3. **Consistency:** LR provides stable, consistent scoring; trees can overfit or behave unexpectedly on new data.\n\n"
        "Tree models trade interpretability for raw performance. Use the toggle below to experiment, "
        "but keep LR as your production model unless business metrics justify the trade-off."
    )
