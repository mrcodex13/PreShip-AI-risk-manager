"""Drift monitoring module: Population Stability Index (PSI) and subgroup parity."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
    epsilon: float = 1e-9
) -> float:
    """
    Compute Population Stability Index (PSI).
    
    Measures shift in distribution between expected (training) and actual (new batch).
    
    PSI interpretation:
        < 0.1: no significant drift
        0.1–0.2: small drift (minor intervention recommended)
        > 0.2: significant drift (retrain recommended)
    
    Args:
        expected: baseline distribution (from training data)
        actual: new distribution (from new batch)
        buckets: number of bins
        epsilon: small constant to avoid log(0)
    
    Returns:
        PSI value (float)
    """
    # Handle edge cases
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    
    # Compute bins using expected distribution
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    
    if min_val == max_val:
        # No variation; return 0 (perfect stability)
        return 0.0
    
    bin_edges = np.linspace(min_val, max_val, buckets + 1)
    
    # Digitize both distributions
    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]
    
    # Normalize to get proportions
    expected_prop = (expected_counts + epsilon) / (expected_counts.sum() + epsilon * len(expected_counts))
    actual_prop = (actual_counts + epsilon) / (actual_counts.sum() + epsilon * len(actual_counts))
    
    # Compute PSI
    psi = np.sum(actual_prop * np.log(actual_prop / expected_prop))
    
    return float(psi)


def compute_psi_per_feature(
    expected_data: pd.DataFrame,
    actual_data: pd.DataFrame,
    numeric_features: list,
    buckets: int = 10
) -> pd.DataFrame:
    """
    Compute PSI for each numeric feature.
    
    Args:
        expected_data: baseline dataset (training data)
        actual_data: new batch dataset
        numeric_features: list of numeric column names
        buckets: number of bins per feature
    
    Returns:
        DataFrame with columns: Feature, PSI, Status (drift/stable)
    """
    results = []
    
    for feature in numeric_features:
        if feature not in expected_data.columns or feature not in actual_data.columns:
            warnings.warn(f"Feature {feature} not found in one of the datasets.")
            continue
        
        expected_vals = expected_data[feature].dropna().values
        actual_vals = actual_data[feature].dropna().values
        
        if len(expected_vals) == 0 or len(actual_vals) == 0:
            psi = np.nan
        else:
            psi = compute_psi(expected_vals, actual_vals, buckets=buckets)
        
        status = 'DRIFT' if psi > 0.2 else ('MONITOR' if psi > 0.1 else 'STABLE')
        
        results.append({
            'Feature': feature,
            'PSI': psi,
            'Status': status
        })
    
    return pd.DataFrame(results).sort_values('PSI', ascending=False)


def flag_rate_by_subgroup(
    data: pd.DataFrame,
    predictions: np.ndarray,
    threshold: float,
    subgroup_columns: list
) -> Dict[str, pd.DataFrame]:
    """
    Compute flag rate (% of orders flagged as high-risk) by subgroup.
    
    Identifies disparities in model decisions across demographic/geographic groups.
    
    Args:
        data: order data with subgroup columns
        predictions: model risk scores (must match len(data))
        threshold: risk threshold for flagging
        subgroup_columns: list of column names to group by (e.g., ['Gender', 'State'])
    
    Returns:
        Dict mapping subgroup_column -> DataFrame with columns:
            Subgroup_Value, Count, Flagged_Count, Flag_Rate (%)
    """
    results = {}
    
    for col in subgroup_columns:
        if col not in data.columns:
            warnings.warn(f"Column {col} not found in data.")
            continue
        
        flagged = (predictions >= threshold).astype(int)
        df = pd.DataFrame({
            'subgroup': data[col],
            'flagged': flagged
        })
        
        grouped = df.groupby('subgroup').agg({
            'flagged': ['count', 'sum']
        }).reset_index()
        grouped.columns = ['Subgroup_Value', 'Total_Count', 'Flagged_Count']
        grouped['Flag_Rate_Percent'] = (grouped['Flagged_Count'] / grouped['Total_Count']) * 100
        grouped = grouped.sort_values('Flag_Rate_Percent', ascending=False)
        
        results[col] = grouped
    
    return results


def parity_summary(flag_rate_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Summarize flag-rate parity across subgroups.
    
    Returns:
        DataFrame with columns: Subgroup, Min_Rate, Max_Rate, Disparity_Ratio
    """
    summary_rows = []
    
    for subgroup_col, df in flag_rate_dict.items():
        if len(df) > 1:
            min_rate = df['Flag_Rate_Percent'].min()
            max_rate = df['Flag_Rate_Percent'].max()
            disparity_ratio = max_rate / min_rate if min_rate > 0 else np.inf
            
            summary_rows.append({
                'Subgroup': subgroup_col,
                'Min_Flag_Rate': f"{min_rate:.1f}%",
                'Max_Flag_Rate': f"{max_rate:.1f}%",
                'Disparity_Ratio': f"{disparity_ratio:.2f}x"
            })
    
    return pd.DataFrame(summary_rows)
