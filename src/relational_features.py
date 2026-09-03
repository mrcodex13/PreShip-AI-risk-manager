"""Relational features module: graph-based risk signals and abuse-ring indicators."""

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class RelationalFeatureExtractor:
    """Simplified graph-based feature extractor for customer/order relations."""

    REQUIRED_COLUMNS = {
        "velocity": ["Customer_ID", "Order_Date"],
        "duplicate_addresses": ["Customer_ID", "Address"],
        "duplicate_phones": ["Customer_ID", "Phone"],
        "first_time_buyer": ["Customer_ID", "Order_Date"],
        "high_value_cod": ["Order_Value", "Payment_Method"],
    }

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.missing_columns = self._check_required_columns()

        if self.missing_columns:
            logger.warning(
                "Relational features are partial or unavailable. "
                f"Missing identifiers: {self.missing_columns}. "
                "When Customer_ID, Address, Phone, and Order_Date are available, the graph features below become active."
            )

    def _check_required_columns(self) -> List[str]:
        all_required = set()
        for cols in self.REQUIRED_COLUMNS.values():
            all_required.update(cols)
        return [col for col in sorted(all_required) if col not in self.data.columns]

    def _as_series(self, values: Optional[pd.Series], index: pd.Index) -> Optional[pd.Series]:
        if values is None:
            return None
        series = pd.Series(values, index=index)
        return series

    def extract_all(self) -> pd.DataFrame:
        features = {
            "order_velocity": self.order_velocity(),
            "duplicate_address_count": self.duplicate_address_count(),
            "duplicate_phone_count": self.duplicate_phone_count(),
            "is_first_time_buyer": self.is_first_time_buyer(),
            "is_high_value_cod": self.is_high_value_cod(),
        }

        available = {key: value for key, value in features.items() if value is not None}
        if not available:
            return pd.DataFrame(
                {
                    "order_velocity": np.nan,
                    "duplicate_address_count": np.nan,
                    "duplicate_phone_count": np.nan,
                    "is_first_time_buyer": np.nan,
                    "is_high_value_cod": np.nan,
                    "ring_risk_score": np.nan,
                },
                index=self.data.index,
            )

        features_df = pd.DataFrame(available, index=self.data.index)
        features_df["ring_risk_score"] = self.ring_risk_score(features_df)
        return features_df

    def order_velocity(self) -> Optional[pd.Series]:
        required = ["Customer_ID", "Order_Date"]
        if not all(col in self.data.columns for col in required):
            return None

        frame = self.data.copy()
        frame["Order_Date"] = pd.to_datetime(frame["Order_Date"], errors="coerce")
        frame["Customer_ID"] = frame["Customer_ID"].fillna("UNKNOWN").astype(str)
        window_end = frame["Order_Date"].max()
        if pd.isna(window_end):
            return None
        window_start = window_end - pd.Timedelta(days=7)
        recent = frame[frame["Order_Date"] >= window_start].groupby("Customer_ID").size()
        return frame["Customer_ID"].map(recent).fillna(0).astype(float)

    def duplicate_address_count(self) -> Optional[pd.Series]:
        if "Address" not in self.data.columns:
            return None

        normalized = self.data["Address"].fillna("UNKNOWN").astype(str)
        counts = normalized.value_counts()
        return normalized.map(counts).fillna(0).astype(float)

    def duplicate_phone_count(self) -> Optional[pd.Series]:
        if "Phone" not in self.data.columns:
            return None

        normalized = self.data["Phone"].fillna("UNKNOWN").astype(str)
        counts = normalized.value_counts()
        return normalized.map(counts).fillna(0).astype(float)

    def is_first_time_buyer(self) -> Optional[pd.Series]:
        required = ["Customer_ID", "Order_Date"]
        if not all(col in self.data.columns for col in required):
            return None

        frame = self.data.copy()
        frame["Customer_ID"] = frame["Customer_ID"].fillna("UNKNOWN").astype(str)
        frame["Order_Date"] = pd.to_datetime(frame["Order_Date"], errors="coerce")
        first_order = frame.groupby("Customer_ID")["Order_Date"].transform("min")
        return (frame["Order_Date"] == first_order).astype(float)

    def is_high_value_cod(self) -> Optional[pd.Series]:
        required = ["Order_Value", "Payment_Method"]
        if not all(col in self.data.columns for col in required):
            return None

        value = pd.to_numeric(self.data["Order_Value"], errors="coerce")
        threshold = value.quantile(0.75) if value.notna().any() else 0.0
        is_cod = self.data["Payment_Method"].astype(str).str.lower().eq("cod")
        return ((value > threshold) & is_cod).astype(float)

    def ring_risk_score(self, features_df: Optional[pd.DataFrame] = None) -> pd.Series:
        if features_df is None:
            features_df = self.extract_all()

        if features_df.empty:
            return pd.Series(np.nan, index=self.data.index)

        score_parts = []
        for column in ["duplicate_address_count", "duplicate_phone_count", "order_velocity", "is_first_time_buyer"]:
            if column in features_df.columns:
                series = pd.to_numeric(features_df[column], errors="coerce").fillna(0)
                score_parts.append(series.rank(pct=True))

        if not score_parts:
            return pd.Series(np.nan, index=self.data.index)

        combined = sum(score_parts) / len(score_parts)
        return combined.clip(0, 1).fillna(0.0)


def get_relational_features_documentation() -> str:
    """Return a detailed explanation of relational features for the UI."""
    return (
        "**Relational / abuse-ring features**\n\n"
        "When customer identifiers are present, the module can compute:\n"
        "- order velocity (same customer, rolling 7-day activity)\n"
        "- address duplication count\n"
        "- phone duplication count\n"
        "- first-time buyer flag\n"
        "- high-value COD risk flag\n"
        "- ring_risk_score (blend of duplicate-account and velocity indicators)\n\n"
        "The current demo dataset does not include the customer linkage fields needed for production abuse-ring detection, "
        "so these features stay dormant until operational identity data is available."
    )
