"""Immutable realized outcomes and independent native no-op verification."""

from outcomes.evidence import (
    ArrivalEvidence,
    ArrivalTargetSpec,
    TimingEvidence,
    capture_targets,
    observe_arrivals,
    timing_satisfied,
)
from outcomes.ledger import OutcomeSnapshot, ScenarioLedger
from outcomes.native_contract import NativeArrivalHistory
from outcomes.verification import (
    OutcomeClassification,
    VerificationResult,
    classify_outcome,
    verify_noop,
)

__all__ = [
    "ArrivalEvidence",
    "ArrivalTargetSpec",
    "TimingEvidence",
    "capture_targets",
    "observe_arrivals",
    "timing_satisfied",
    "OutcomeSnapshot",
    "ScenarioLedger",
    "NativeArrivalHistory",
    "OutcomeClassification",
    "VerificationResult",
    "classify_outcome",
    "verify_noop",
]
