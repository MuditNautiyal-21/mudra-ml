"""Decision log used to make every automated choice auditable.

Every stage of the pipeline records what it decided and the rule that produced
the decision. The report is rendered directly from this log, so the log is the
source of truth for how a run reached its result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mudraml")


@dataclass(frozen=True)
class Decision:
    """A single recorded choice.

    Args:
        stage: Pipeline stage that made the choice (for example "profile").
        decision: Short statement of what was decided.
        rule: The named rule or statistical test that produced the decision.
        detail: Optional structured context, such as the values compared.
    """

    stage: str
    decision: str
    rule: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "decision": self.decision,
            "rule": self.rule,
            "detail": self.detail,
        }


class DecisionLog:
    """Ordered collection of decisions made during a run."""

    def __init__(self) -> None:
        self._entries: list[Decision] = []

    def record(
        self,
        stage: str,
        decision: str,
        rule: str,
        detail: dict[str, Any] | None = None,
    ) -> Decision:
        """Append a decision and emit it to the logger."""
        entry = Decision(stage=stage, decision=decision, rule=rule, detail=detail or {})
        self._entries.append(entry)
        logger.info("[%s] %s (rule: %s)", stage, decision, rule)
        return entry

    def for_stage(self, stage: str) -> list[Decision]:
        return [e for e in self._entries if e.stage == stage]

    def stages(self) -> list[str]:
        seen: list[str] = []
        for entry in self._entries:
            if entry.stage not in seen:
                seen.append(entry.stage)
        return seen

    def as_list(self) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a stream handler to the package logger if none is present."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
