"""Fraud-spike monitoring using hybrid risk scores with control-chart alerts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SpikeReport:
    """Represents a detected abnormal spike in flagged-order volume."""

    severity: str
    segment: str
    segment_value: str
    time_window: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    observed_rate: float
    historical_mean: float
    historical_std: float
    threshold: float
    z_score: float
    flagged_orders: int
    total_orders: int

    @property
    def to_dict(self):
        return {
            "severity": self.severity,
            "segment": self.segment,
            "segment_value": self.segment_value,
            "time_window": self.time_window,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "observed_rate": self.observed_rate,
            "historical_mean": self.historical_mean,
            "historical_std": self.historical_std,
            "threshold": self.threshold,
            "z_score": self.z_score,
            "flagged_orders": self.flagged_orders,
            "total_orders": self.total_orders,
        }


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _normalize_window(window: str) -> str:
    """Translate UI-friendly names into pandas-compatible frequencies."""
    mapping = {
        "hour": "h",
        "hourly": "h",
        "day": "D",
        "daily": "D",
    }
    normalized = str(window).strip().lower()
    if normalized not in mapping:
        raise ValueError(f"Unsupported time window '{window}'. Use 'hour' or 'day'.")
    return mapping[normalized]


def _segment_label(row: pd.Series, segment_columns: Sequence[str]) -> str:
    values = []
    for column in segment_columns:
        value = row.get(column)
        if pd.isna(value):
            value = "Missing"
        values.append(f"{column}={value}")
    return " | ".join(values)


def _segment_key(row: pd.Series, segment_columns: Sequence[str]) -> tuple:
    key = []
    for column in segment_columns:
        value = row.get(column)
        if pd.isna(value):
            value = "Missing"
        key.append(str(value))
    return tuple(key)


def _bucketed_flagged_orders(
    frame: pd.DataFrame,
    time_col: str,
    risk_col: str,
    segment_columns: Sequence[str],
    window: str,
    threshold: float,
) -> pd.DataFrame:
    df = frame.copy()
    if df.empty:
        return pd.DataFrame(columns=[*segment_columns, "time_bucket", "flagged_orders", "total_orders", "observed_rate"])

    if time_col not in df.columns:
        df[time_col] = pd.Timestamp.utcnow()

    df[time_col] = _coerce_datetime(df[time_col])
    df = df.dropna(subset=[time_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=[*segment_columns, "time_bucket", "flagged_orders", "total_orders", "observed_rate"])

    if risk_col not in df.columns:
        raise ValueError(f"Risk column '{risk_col}' not found in input data. Use hybrid risk score as the input signal.")

    freq = _normalize_window(window)
    df["__flagged"] = df[risk_col].fillna(0.0) >= threshold
    df["time_bucket"] = df[time_col].dt.floor(freq)

    grouped = df.groupby([*segment_columns, "time_bucket"], dropna=False).agg(
        flagged_orders=("__flagged", "sum"),
        total_orders=("__flagged", "size"),
    ).reset_index()
    grouped["observed_rate"] = grouped["flagged_orders"] / grouped["total_orders"]
    grouped = grouped.sort_values([*segment_columns, "time_bucket"]).reset_index(drop=True)
    return grouped


def _severity_from_z(z_score: float) -> str:
    if np.isnan(z_score):
        return "No signal"
    if z_score >= 5:
        return "Critical"
    if z_score >= 3:
        return "High"
    if z_score >= 2:
        return "Moderate"
    return "Low"


def detect_spike(
    current_window_df: pd.DataFrame,
    history_df: pd.DataFrame,
    time_col: str = "Order_Date",
    risk_col: str = "risk_score",
    segment_columns: Optional[Sequence[str]] = None,
    window: str = "day",
    threshold: float = 0.5,
    k: float = 3.0,
) -> Optional[SpikeReport]:
    """Detect the strongest resulting fraud-spike alert in the latest time bucket."""
    if segment_columns is None:
        segment_columns = ["Category", "State", "Brand"]
    segment_columns = [col for col in segment_columns if col in current_window_df.columns or col in history_df.columns]
    if not segment_columns:
        segment_columns = ["Category"]

    if current_window_df.empty or history_df.empty:
        return None

    current_summary = _bucketed_flagged_orders(
        current_window_df, time_col, risk_col, segment_columns, window, threshold
    )
    history_summary = _bucketed_flagged_orders(
        history_df, time_col, risk_col, segment_columns, window, threshold
    )

    if current_summary.empty:
        return None

    latest_bucket = current_summary["time_bucket"].max()
    current_rows = current_summary[current_summary["time_bucket"] == latest_bucket].reset_index(drop=True)
    if current_rows.empty:
        return None

    best = None
    for _, row in current_rows.iterrows():
        segment_key = _segment_key(row, segment_columns)
        historical = history_summary.copy()
        for idx, column in enumerate(segment_columns):
            historical = historical[historical[column].astype(str) == str(segment_key[idx])]

        if historical.empty:
            continue

        rates = historical["observed_rate"].dropna()
        if rates.empty:
            continue

        historical_mean = float(rates.mean())
        historical_std = float(rates.std(ddof=0)) if len(rates) > 1 else 0.0
        observed_rate = float(row["observed_rate"])
        if historical_std == 0:
            z_score = np.inf if observed_rate > historical_mean else 0.0
        else:
            z_score = (observed_rate - historical_mean) / historical_std

        threshold_rate = historical_mean + (k * historical_std)
        if observed_rate <= threshold_rate and z_score <= k:
            continue

        time_unit = "h" if _normalize_window(window) == "h" else "d"
        candidate = SpikeReport(
            severity=_severity_from_z(z_score),
            segment=segment_columns[0] if len(segment_columns) == 1 else "multi-segment",
            segment_value=_segment_label(row, segment_columns),
            time_window=window,
            window_start=latest_bucket,
            window_end=latest_bucket + pd.Timedelta(1, unit=time_unit),
            observed_rate=observed_rate,
            historical_mean=historical_mean,
            historical_std=historical_std,
            threshold=threshold_rate,
            z_score=z_score,
            flagged_orders=int(row["flagged_orders"]),
            total_orders=int(row["total_orders"]),
        )

        if best is None or candidate.z_score > best.z_score:
            best = candidate

    return best


def spike_report_table(
    current_window_df: pd.DataFrame,
    history_df: pd.DataFrame,
    time_col: str = "Order_Date",
    risk_col: str = "risk_score",
    segment_columns: Optional[Sequence[str]] = None,
    window: str = "day",
    threshold: float = 0.5,
    k: float = 3.0,
) -> pd.DataFrame:
    """Return a summary table of all segment-window alerts for review."""
    if segment_columns is None:
        segment_columns = ["Category", "State", "Brand"]
    segment_columns = [col for col in segment_columns if col in current_window_df.columns or col in history_df.columns]
    if not segment_columns:
        segment_columns = ["Category"]

    current_summary = _bucketed_flagged_orders(
        current_window_df, time_col, risk_col, segment_columns, window, threshold
    )
    history_summary = _bucketed_flagged_orders(
        history_df, time_col, risk_col, segment_columns, window, threshold
    )

    if current_summary.empty:
        return pd.DataFrame()

    latest_bucket = current_summary["time_bucket"].max()
    time_unit = "h" if _normalize_window(window) == "h" else "d"
    rows = []
    for _, row in current_summary[current_summary["time_bucket"] == latest_bucket].iterrows():
        segment_key = _segment_key(row, segment_columns)
        historical = history_summary.copy()
        for idx, column in enumerate(segment_columns):
            historical = historical[historical[column].astype(str) == str(segment_key[idx])]

        rates = historical["observed_rate"].dropna()
        if rates.empty:
            continue

        observed_rate = float(row["observed_rate"])
        historical_mean = float(rates.mean())
        historical_std = float(rates.std(ddof=0)) if len(rates) > 1 else 0.0
        control_threshold = historical_mean + (k * historical_std)
        if historical_std == 0:
            z_score = np.inf if observed_rate > historical_mean else 0.0
        else:
            z_score = (observed_rate - historical_mean) / historical_std

        rows.append(
            {
                "segment": _segment_label(row, segment_columns),
                "segment_value": _segment_label(row, segment_columns),
                "time_window": window,
                "window_start": row["time_bucket"],
                "window_end": row["time_bucket"] + pd.Timedelta(1, unit=time_unit),
                "flagged_orders": int(row["flagged_orders"]),
                "total_orders": int(row["total_orders"]),
                "observed_rate": observed_rate,
                "historical_mean": historical_mean,
                "historical_std": historical_std,
                "control_threshold": control_threshold,
                "z_score": z_score,
                "severity": _severity_from_z(z_score),
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values("z_score", ascending=False).reset_index(drop=True)
    return result
