"""
Causal Research Engine - Hypothesis Engine
Version: v5.0.5

Creates and validates structured causal hypotheses.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from causal_research_engine.models import (
    CausalHypothesis,
    HypothesisStatus,
)


class HypothesisValidationError(ValueError):
    """Raised when a causal hypothesis is invalid."""


class CausalHypothesisEngine:
    """Create, validate and manage causal hypotheses."""

    VALID_DIRECTIONS = {
        "positive",
        "negative",
        "bidirectional",
        "none",
    }

    def create_hypothesis(
        self,
        *,
        hypothesis_id: str,
        cause: str,
        effect: str,
        description: str = "",
        expected_direction: Optional[str] = None,
        mechanism: Optional[str] = None,
        confounders: Optional[Iterable[str]] = None,
        controls: Optional[Iterable[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CausalHypothesis:

        hypothesis_id = self._required(
            hypothesis_id,
            "hypothesis_id",
        )

        cause = self._required(
            cause,
            "cause",
        )

        effect = self._required(
            effect,
            "effect",
        )

        if cause.casefold() == effect.casefold():
            raise HypothesisValidationError(
                "Cause and effect must be different."
            )

        direction = self._normalize_direction(
            expected_direction
        )

        mechanism_value = self._optional(
            mechanism
        )

        description_value = (
            self._optional(description) or ""
        )

        normalized_confounders = self._normalize_list(
            confounders
        )

        normalized_controls = self._normalize_list(
            controls
        )

        overlap = (
            set(normalized_confounders)
            & set(normalized_controls)
        )

        if overlap:
            raise HypothesisValidationError(
                "A variable cannot be used simultaneously "
                "as a confounder and a control: "
                + ", ".join(sorted(overlap))
            )

        return CausalHypothesis(
            hypothesis_id=hypothesis_id,
            cause=cause,
            effect=effect,
            description=description_value,
            expected_direction=direction,
            mechanism=mechanism_value,
            confounders=normalized_confounders,
            controls=normalized_controls,
            status=HypothesisStatus.PROPOSED,
            metadata=dict(metadata or {}),
        )

    def validate_hypothesis(
        self,
        hypothesis: Any,
        *,
        require_mechanism: bool = False,
        require_direction: bool = False,
    ) -> list[str]:

        if not isinstance(
            hypothesis,
            CausalHypothesis,
        ):
            return [
                "Object is not a CausalHypothesis."
            ]

        issues: list[str] = []

        if not self._has_text(
            hypothesis.hypothesis_id
        ):
            issues.append(
                "Hypothesis ID is missing."
            )

        if not self._has_text(
            hypothesis.cause
        ):
            issues.append(
                "Cause is missing."
            )

        if not self._has_text(
            hypothesis.effect
        ):
            issues.append(
                "Effect is missing."
            )

        if (
            self._has_text(hypothesis.cause)
            and self._has_text(hypothesis.effect)
            and hypothesis.cause.strip().casefold()
            == hypothesis.effect.strip().casefold()
        ):
            issues.append(
                "Cause and effect must be different."
            )

        if require_direction:
            if not self._has_text(
                hypothesis.expected_direction
            ):
                issues.append(
                    "Expected causal direction is missing."
                )

        if require_mechanism:
            if not self._has_text(
                hypothesis.mechanism
            ):
                issues.append(
                    "Proposed causal mechanism is missing."
                )

        if hypothesis.expected_direction:
            direction = (
                hypothesis.expected_direction
                .strip()
                .casefold()
            )

            if direction not in self.VALID_DIRECTIONS:
                issues.append(
                    "Invalid expected_direction: "
                    f"{hypothesis.expected_direction}"
                )

        confounders = self._normalize_list(
            hypothesis.confounders
        )

        controls = self._normalize_list(
            hypothesis.controls
        )

        overlap = (
            set(confounders)
            & set(controls)
        )

        if overlap:
            issues.append(
                "A variable cannot be used simultaneously "
                "as a confounder and a control: "
                + ", ".join(sorted(overlap))
            )

        return issues

    def is_research_ready(
        self,
        hypothesis: Any,
    ) -> bool:

        return not self.validate_hypothesis(
            hypothesis,
            require_mechanism=True,
            require_direction=True,
        )

    def start_testing(
        self,
        hypothesis: CausalHypothesis,
    ) -> CausalHypothesis:

        issues = self.validate_hypothesis(
            hypothesis,
            require_mechanism=True,
            require_direction=True,
        )

        if issues:
            raise HypothesisValidationError(
                "Hypothesis is not research-ready: "
                + "; ".join(issues)
            )

        hypothesis.status = HypothesisStatus.TESTING

        return hypothesis

    def mark_supported(
        self,
        hypothesis: CausalHypothesis,
    ) -> CausalHypothesis:

        self._ensure_hypothesis(hypothesis)

        hypothesis.status = HypothesisStatus.SUPPORTED

        return hypothesis

    def mark_weak(
        self,
        hypothesis: CausalHypothesis,
    ) -> CausalHypothesis:

        self._ensure_hypothesis(hypothesis)

        hypothesis.status = HypothesisStatus.WEAK

        return hypothesis

    def mark_falsified(
        self,
        hypothesis: CausalHypothesis,
    ) -> CausalHypothesis:

        self._ensure_hypothesis(hypothesis)

        hypothesis.status = HypothesisStatus.FALSIFIED

        return hypothesis

    def mark_inconclusive(
        self,
        hypothesis: CausalHypothesis,
    ) -> CausalHypothesis:

        self._ensure_hypothesis(hypothesis)

        hypothesis.status = HypothesisStatus.INCONCLUSIVE

        return hypothesis

    @classmethod
    def _normalize_direction(
        cls,
        direction: Optional[str],
    ) -> Optional[str]:

        if direction is None:
            return None

        cleaned = direction.strip().casefold()

        aliases = {
            "up": "positive",
            "increase": "positive",
            "increasing": "positive",
            "down": "negative",
            "decrease": "negative",
            "decreasing": "negative",
            "both": "bidirectional",
            "neutral": "none",
        }

        cleaned = aliases.get(
            cleaned,
            cleaned,
        )

        if cleaned not in cls.VALID_DIRECTIONS:
            raise HypothesisValidationError(
                f"Invalid expected_direction: {direction}"
            )

        return cleaned

    @staticmethod
    def _normalize_list(
        values: Optional[Iterable[str]],
    ) -> list[str]:

        if values is None:
            return []

        result: list[str] = []

        for value in values:
            if value is None:
                continue

            cleaned = str(value).strip()

            if not cleaned:
                continue

            if cleaned not in result:
                result.append(cleaned)

        return result

    @staticmethod
    def _required(
        value: Any,
        field_name: str,
    ) -> str:

        if value is None:
            raise HypothesisValidationError(
                f"{field_name} is required."
            )

        cleaned = str(value).strip()

        if not cleaned:
            raise HypothesisValidationError(
                f"{field_name} is required."
            )

        return cleaned

    @staticmethod
    def _optional(
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        cleaned = str(value).strip()

        return cleaned or None

    @staticmethod
    def _has_text(
        value: Any,
    ) -> bool:

        return (
            value is not None
            and bool(str(value).strip())
        )

    @staticmethod
    def _ensure_hypothesis(
        hypothesis: Any,
    ) -> None:

        if not isinstance(
            hypothesis,
            CausalHypothesis,
        ):
            raise HypothesisValidationError(
                "Object is not a CausalHypothesis."
            )