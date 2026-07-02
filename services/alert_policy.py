"""Risk and caregiver alert policy for VLM fall decisions."""

from __future__ import annotations

from dataclasses import dataclass


HIGH_RISK = "high_risk"
LOW_RISK = "low_risk"
NO_ALARM = "no_alarm"

ALERT_NONE = "none"
ALERT_PENDING = "pending"
ALERT_HANDLED = "handled"

DECISION_SOURCE_VLM = "vlm"
DECISION_SOURCE_YOLO_FALLBACK = "yolo_fallback"

try:
    from config import (
        DEFAULT_HIGH_RISK_REPEAT_SECONDS,
        DEFAULT_LOW_RISK_REPEAT_SECONDS,
    )
except Exception:  # pragma: no cover - standalone import fallback
    DEFAULT_HIGH_RISK_REPEAT_SECONDS = 20
    DEFAULT_LOW_RISK_REPEAT_SECONDS = 60


@dataclass(frozen=True)
class AlertDecision:
    risk_level: str
    alert_status: str
    decision_source: str
    system_degraded: bool
    vlm_status: str
    repeat_seconds: int | None


def map_vlm_decision(
    result: str | None,
    high_risk_repeat_seconds: int = DEFAULT_HIGH_RISK_REPEAT_SECONDS,
    low_risk_repeat_seconds: int = DEFAULT_LOW_RISK_REPEAT_SECONDS,
) -> AlertDecision:
    """Map a VLM outcome or failure marker to caregiver alert semantics."""
    status = str(result or "").strip().lower() or "need_human_review"
    if status == "confirmed_fall":
        return AlertDecision(
            risk_level=HIGH_RISK,
            alert_status=ALERT_PENDING,
            decision_source=DECISION_SOURCE_VLM,
            system_degraded=False,
            vlm_status=status,
            repeat_seconds=high_risk_repeat_seconds,
        )
    if status in {"need_human_review", "uncertain"}:
        return AlertDecision(
            risk_level=LOW_RISK,
            alert_status=ALERT_PENDING,
            decision_source=DECISION_SOURCE_VLM,
            system_degraded=False,
            vlm_status=status,
            repeat_seconds=low_risk_repeat_seconds,
        )
    if status == "rejected":
        return AlertDecision(
            risk_level=NO_ALARM,
            alert_status=ALERT_NONE,
            decision_source=DECISION_SOURCE_VLM,
            system_degraded=False,
            vlm_status=status,
            repeat_seconds=None,
        )
    if status in {"failed", "timeout", "vlm_failed"}:
        vlm_status = "timeout" if status == "timeout" else "failed"
        return AlertDecision(
            risk_level=LOW_RISK,
            alert_status=ALERT_PENDING,
            decision_source=DECISION_SOURCE_YOLO_FALLBACK,
            system_degraded=True,
            vlm_status=vlm_status,
            repeat_seconds=low_risk_repeat_seconds,
        )
    return AlertDecision(
        risk_level=LOW_RISK,
        alert_status=ALERT_PENDING,
        decision_source=DECISION_SOURCE_VLM,
        system_degraded=False,
        vlm_status=status,
        repeat_seconds=low_risk_repeat_seconds,
    )


def reminder_interval_seconds(
    risk_level: str | None,
    high_risk_repeat_seconds: int = DEFAULT_HIGH_RISK_REPEAT_SECONDS,
    low_risk_repeat_seconds: int = DEFAULT_LOW_RISK_REPEAT_SECONDS,
) -> int | None:
    if risk_level == HIGH_RISK:
        return high_risk_repeat_seconds
    if risk_level == LOW_RISK:
        return low_risk_repeat_seconds
    return None
