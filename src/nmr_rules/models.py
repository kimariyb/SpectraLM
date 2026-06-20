"""Typed outputs shared by the one-dimensional NMR rule engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuleEvidence:
    """One traceable inference produced by an NMR rule.

    Parameters
    ----------
    rule_id
        Stable rule identifier.
    category
        Broad evidence category such as ``formula`` or ``h1_fragment``.
    conclusion
        Compact model-facing conclusion.
    confidence
        Rule confidence in the closed interval ``[0, 1]``.
    strength
        Qualitative evidence strength: ``hard``, ``strong``, ``moderate``, or
        ``weak``.
    human_tip
        Interpretation advice for a human spectroscopist.
    metadata
        Structured observations supporting the conclusion.
    """

    rule_id: str
    category: str
    conclusion: str
    confidence: float
    strength: str
    human_tip: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class RuleAnalysis:
    """All one-dimensional NMR rule evidence for one sample."""

    library_name: str
    molecular_formula: str | None
    dbe: float | None
    evidence: tuple[RuleEvidence, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "library_name": self.library_name,
            "molecular_formula": self.molecular_formula,
            "dbe": self.dbe,
            "evidence": [item.to_dict() for item in self.evidence],
            "warnings": list(self.warnings),
        }
