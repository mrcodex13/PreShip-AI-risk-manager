"""Standalone chargeback-evidence responder for post-dispute workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from feedback import FeedbackDatabase


@dataclass
class RiskProfile:
    """Risk profile for the disputed order."""

    risk_score: float
    probability: float
    novelty: float
    risk_band: str
    confidence: float
    shap_drivers: Dict[str, float]


@dataclass
class ChargebackEvidencePacket:
    """Structured evidence packet to send to a payment processor or bank."""

    order_id: str
    order_record: Dict[str, Any]
    risk_profile: RiskProfile
    delivery_confirmation: str
    prior_review_outcome: str
    prior_review_notes: str
    generated_at_utc: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "order_record": self.order_record,
            "risk_profile": asdict(self.risk_profile),
            "delivery_confirmation": self.delivery_confirmation,
            "prior_review_outcome": self.prior_review_outcome,
            "prior_review_notes": self.prior_review_notes,
            "generated_at_utc": self.generated_at_utc,
        }


def build_chargeback_evidence(
    order_id: str,
    order_record: Dict[str, Any],
    risk_profile: RiskProfile,
    feedback_db_path: Optional[str | Path] = None,
    delivery_confirmation: str = "No delivery confirmation attached.",
) -> ChargebackEvidencePacket:
    """Assemble a structured evidence file for a disputed transaction."""
    if feedback_db_path is not None:
        db = FeedbackDatabase(Path(feedback_db_path))
        records = db.get_feedback_records()
        related = records[records["order_id"] == order_id] if "order_id" in records.columns else None
        if related is not None and not related.empty:
            latest = related.iloc[-1]
            prior_review_outcome = str(latest.get("human_outcome", "not_yet_reviewed"))
            prior_review_notes = str(latest.get("notes", ""))
        else:
            prior_review_outcome = "not_yet_reviewed"
            prior_review_notes = "No prior human review outcome was found for this order."
    else:
        prior_review_outcome = "not_yet_reviewed"
        prior_review_notes = "No prior human review outcome was found for this order."

    return ChargebackEvidencePacket(
        order_id=order_id,
        order_record=order_record,
        risk_profile=risk_profile,
        delivery_confirmation=delivery_confirmation,
        prior_review_outcome=prior_review_outcome,
        prior_review_notes=prior_review_notes,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def format_chargeback_response(packet: ChargebackEvidencePacket) -> str:
    """Format an evidence packet into a concise response document for a processor or bank."""
    summary_lines = [
        "PreShip AI Chargeback Evidence Response",
        "====================================",
        f"Order ID: {packet.order_id}",
        f"Generated at (UTC): {packet.generated_at_utc}",
        "",
        "1. Transaction Summary",
        json.dumps(packet.order_record, indent=2, default=str),
        "",
        "2. Risk Profile",
        f"Hybrid risk score: {packet.risk_profile.risk_score:.1%}",
        f"Calibrated probability: {packet.risk_profile.probability:.1%}",
        f"Novelty signal: {packet.risk_profile.novelty:.1%}",
        f"Risk band: {packet.risk_profile.risk_band}",
        f"Confidence: {packet.risk_profile.confidence:.1%}",
        "SHAP drivers: " + (
            ", ".join(f"{name}={value:+.3f}" for name, value in packet.risk_profile.shap_drivers.items())
            if packet.risk_profile.shap_drivers
            else "No SHAP drivers available"
        ),
        "",
        "3. Delivery / Fulfillment Evidence",
        packet.delivery_confirmation,
        "",
        "4. Prior Review Outcome",
        f"Outcome: {packet.prior_review_outcome}",
        f"Notes: {packet.prior_review_notes}",
    ]
    return "\n".join(summary_lines)


def prepare_chargeback_document(
    order_id: str,
    order_record: Dict[str, Any],
    risk_profile: RiskProfile,
    feedback_db_path: Optional[str | Path] = None,
    delivery_confirmation: str = "No delivery confirmation attached.",
) -> str:
    """Convenience wrapper that returns the formatted response document."""
    packet = build_chargeback_evidence(
        order_id=order_id,
        order_record=order_record,
        risk_profile=risk_profile,
        feedback_db_path=feedback_db_path,
        delivery_confirmation=delivery_confirmation,
    )
    return format_chargeback_response(packet)
