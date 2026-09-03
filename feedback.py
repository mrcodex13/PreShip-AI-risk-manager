"""Feedback and active learning module: collect human feedback and retrain."""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List
import json
import logging


logger = logging.getLogger(__name__)


class FeedbackDatabase:
    """Manage feedback collection and storage in SQLite."""
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Path = None):
        """
        Initialize feedback database.
        
        Args:
            db_path: path to SQLite database file
        """
        self.db_path = db_path or Path(__file__).parent / 'feedback.db'
        self._create_tables()
    
    def _create_tables(self):
        """Create feedback tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT,
                    order_features TEXT,
                    model_score REAL,
                    risk_band TEXT,
                    confidence REAL,
                    human_outcome TEXT,
                    outcome_timestamp TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(order_id) ON CONFLICT REPLACE
                )
            """)
            
            # Metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Ensure version is tracked
            cursor.execute("INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)",
                          ('schema_version', str(self.SCHEMA_VERSION)))
            
            conn.commit()
    
    def record_feedback(
        self,
        order_id: str,
        order_features: Dict[str, Any],
        model_score: float,
        risk_band: str,
        confidence: float,
        human_outcome: str,
        notes: str = ""
    ) -> int:
        """
        Record human feedback for a scored order.
        
        Args:
            order_id: unique order identifier
            order_features: dict of order features
            model_score: model's risk score (0-1)
            risk_band: 'Low', 'Watch', or 'High'
            confidence: confidence level (0-1)
            human_outcome: 'confirmed_risky', 'false_alarm', or 'not_yet_reviewed'
            notes: optional reviewer notes
        
        Returns:
            Row ID of inserted/updated record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            features_json = json.dumps(order_features)
            outcome_timestamp = datetime.now(timezone.utc).isoformat()
            
            cursor.execute("""
                INSERT INTO feedback (
                    order_id, order_features, model_score, risk_band,
                    confidence, human_outcome, outcome_timestamp, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id, features_json, model_score, risk_band,
                confidence, human_outcome, outcome_timestamp, notes
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_feedback_records(self, outcome_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve all feedback records.
        
        Args:
            outcome_filter: if set, filter to this outcome type
        
        Returns:
            DataFrame with feedback records
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM feedback"
            if outcome_filter:
                query += f" WHERE human_outcome = '{outcome_filter}'"
            
            df = pd.read_sql_query(query, conn)
            
            # Parse order_features JSON
            if len(df) > 0 and 'order_features' in df.columns:
                df['order_features'] = df['order_features'].apply(json.loads)
            
            return df
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total = cursor.fetchone()[0]
            
            # By outcome
            cursor.execute("""
                SELECT human_outcome, COUNT(*) as count
                FROM feedback
                GROUP BY human_outcome
            """)
            by_outcome = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total_records': total,
                'by_outcome': by_outcome,
                'db_path': str(self.db_path)
            }


def prepare_feedback_for_retraining(
    feedback_db: FeedbackDatabase,
    original_features: list,
    min_feedback_rows: int = 10
) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """
    Prepare feedback records for retraining.
    
    Converts feedback into a dataset compatible with the original model features.
    
    Args:
        feedback_db: FeedbackDatabase instance
        original_features: list of feature names from original training
        min_feedback_rows: minimum feedback records needed to retrain
    
    Returns:
        (X_feedback, y_feedback) or None if not enough feedback
    """
    df = feedback_db.get_feedback_records()
    
    if len(df) < min_feedback_rows:
        logger.warning(
            f"Only {len(df)} feedback records available; need {min_feedback_rows} to retrain."
        )
        return None
    
    # Expand order_features JSON into columns
    features_expanded = pd.json_normalize(df['order_features'])
    
    # Map human_outcome to binary label
    outcome_mapping = {
        'confirmed_risky': 1,
        'false_alarm': 0,
        'not_yet_reviewed': None  # Exclude
    }
    df['label'] = df['human_outcome'].map(outcome_mapping)
    df_labeled = df[df['label'].notna()].copy()
    
    if len(df_labeled) < min_feedback_rows:
        logger.warning(
            f"Only {len(df_labeled)} labeled feedback records available; need {min_feedback_rows}."
        )
        return None
    
    # Select columns matching original features
    x_feedback = features_expanded[original_features].copy()
    y_feedback = df_labeled['label'].values.astype(int)
    
    return x_feedback, pd.Series(y_feedback, index=x_feedback.index)


def log_retraining_metrics(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    feedback_count: int
) -> str:
    """
    Generate a detailed log of retraining impact.
    
    Args:
        metrics_before: original test metrics
        metrics_after: metrics after retraining
        feedback_count: number of feedback records used
    
    Returns:
        Formatted log string
    """
    lines = [
        f"=== Retraining Report ===",
        f"Feedback records incorporated: {feedback_count}",
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Metric comparison (Before → After):"
    ]
    
    for metric_key in ['precision', 'recall', 'auc_pr', 'roc_auc']:
        before = metrics_before.get(metric_key, np.nan)
        after = metrics_after.get(metric_key, np.nan)
        
        if not np.isnan(before) and not np.isnan(after):
            delta = after - before
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            lines.append(
                f"  {metric_key.upper()}: {before:.3f} → {after:.3f} ({delta:+.3f}) {direction}"
            )
    
    lines.extend([
        "",
        "IMPORTANT: Monitor whether these improvements generalize to future data."
        " If metrics worsen after retraining, consider:",
        "  - Checking data quality and label consistency",
        "  - Increasing the feedback sample size",
        "  - Using only high-confidence feedback"
    ])
    
    return "\n".join(lines)
