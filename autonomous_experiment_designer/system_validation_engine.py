from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class ValidationStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class ValidationCriterion(str, Enum):
    REPRODUCIBLE_PROVENANCE = "reproducible_provenance"
    PREDICTIVE_CAUSAL_SEPARATION = (
        "predictive_causal_separation"
    )
    OOS_IS_SEPARATED = "oos_is_separated"
    CHANGES_MEASURABLE = "changes_measurable"
    CHANGES_REVERSIBLE = "changes_reversible"
    PORTFOLIO_RECONSTRUCTABLE = (
        "portfolio_reconstructable"
    )
    QUANTUM_CLASSICAL_BENCHMARK = (
        "quantum_classical_benchmark"
    )
    AUTONOMOUS_RESEARCH_CONTROLLED = (
        "autonomous_research_controlled"
    )
    END_TO_END_ACCEPTANCE = "end_to_end_acceptance"
    FULL_REGRESSION = "full_regression"


@dataclass(frozen=True)
class CriterionResult:
    criterion: ValidationCriterion
    status: ValidationStatus
    evidence: tuple[str, ...]
    details: str


@dataclass(frozen=True)
class SystemValidationResult:
    status: ValidationStatus
    version: str
    criteria_results: tuple[CriterionResult, ...]
    passed_count: int
    failed_count: int
    blocked_count: int
    total_count: int
    metadata: Mapping[str, object]
    error: str | None = None

    @property
    def all_criteria_passed(self) -> bool:
        return (
            self.status == ValidationStatus.PASSED
            and self.failed_count == 0
            and self.blocked_count == 0
        )


class SystemValidationEngine:
    VERSION = "v6.0.9"

    REQUIRED_CRITERIA = (
        ValidationCriterion.REPRODUCIBLE_PROVENANCE,
        ValidationCriterion.PREDICTIVE_CAUSAL_SEPARATION,
        ValidationCriterion.OOS_IS_SEPARATED,
        ValidationCriterion.CHANGES_MEASURABLE,
        ValidationCriterion.CHANGES_REVERSIBLE,
        ValidationCriterion.PORTFOLIO_RECONSTRUCTABLE,
        ValidationCriterion.QUANTUM_CLASSICAL_BENCHMARK,
        ValidationCriterion.AUTONOMOUS_RESEARCH_CONTROLLED,
        ValidationCriterion.END_TO_END_ACCEPTANCE,
        ValidationCriterion.FULL_REGRESSION,
    )

    def __init__(
        self,
        *,
        require_human_approval: bool = True,
        require_all_criteria: bool = True,
    ) -> None:
        if not isinstance(
            require_human_approval,
            bool,
        ):
            raise ValueError(
                "require_human_approval must be a boolean."
            )

        if not isinstance(
            require_all_criteria,
            bool,
        ):
            raise ValueError(
                "require_all_criteria must be a boolean."
            )

        self.require_human_approval = (
            require_human_approval
        )
        self.require_all_criteria = (
            require_all_criteria
        )

    @staticmethod
    def _validate_text(
        value: object,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{field_name} must not be empty."
            )

        return value

    @staticmethod
    def _validate_bool(
        value: object,
        field_name: str,
    ) -> bool:
        if not isinstance(value, bool):
            raise ValueError(
                f"{field_name} must be a boolean."
            )

        return value

    @staticmethod
    def _validate_evidence(
        evidence: Sequence[str],
    ) -> tuple[str, ...]:
        if isinstance(
            evidence,
            (str, bytes),
        ):
            raise ValueError(
                "evidence must be a sequence of strings."
            )

        try:
            values = tuple(evidence)
        except TypeError as exc:
            raise ValueError(
                "evidence must be a sequence of strings."
            ) from exc

        result: list[str] = []

        for item in values:
            if not isinstance(item, str):
                raise ValueError(
                    "every evidence item must be a string."
                )

            item = item.strip()

            if not item:
                raise ValueError(
                    "evidence items must not be empty."
                )

            if item not in result:
                result.append(item)

        return tuple(result)

    @staticmethod
    def _criterion_result(
        criterion: ValidationCriterion,
        passed: bool,
        evidence: Sequence[str],
        details: str,
    ) -> CriterionResult:
        return CriterionResult(
            criterion=criterion,
            status=(
                ValidationStatus.PASSED
                if passed
                else ValidationStatus.FAILED
            ),
            evidence=tuple(evidence),
            details=details,
        )

    @staticmethod
    def _validate_mapping(
        value: object,
        field_name: str,
    ) -> Mapping:
        if not isinstance(value, Mapping):
            raise ValueError(
                f"{field_name} must be a mapping."
            )

        return value

    def validate_criterion(
        self,
        criterion: ValidationCriterion,
        *,
        passed: bool,
        evidence: Sequence[str],
        details: str = "",
    ) -> CriterionResult:
        if not isinstance(
            criterion,
            ValidationCriterion,
        ):
            raise ValueError(
                "criterion must be a ValidationCriterion."
            )

        passed = self._validate_bool(
            passed,
            "passed",
        )

        evidence = self._validate_evidence(
            evidence
        )

        details = str(details).strip()

        return self._criterion_result(
            criterion,
            passed,
            evidence,
            details,
        )

    def validate(
        self,
        *,
        version: str = VERSION,
        criteria: Mapping[
            ValidationCriterion,
            bool,
        ],
        evidence: Mapping[
            ValidationCriterion,
            Sequence[str],
        ]
        | None = None,
        details: Mapping[
            ValidationCriterion,
            str,
        ]
        | None = None,
        human_approval: bool = False,
        metadata: Mapping[str, object]
        | None = None,
    ) -> SystemValidationResult:

        version = self._validate_text(
            version,
            "version",
        )

        criteria = self._validate_mapping(
            criteria,
            "criteria",
        )

        if evidence is not None:
            evidence = self._validate_mapping(
                evidence,
                "evidence",
            )
        else:
            evidence = {}

        if details is not None:
            details = self._validate_mapping(
                details,
                "details",
            )
        else:
            details = {}

        if metadata is not None:
            metadata = self._validate_mapping(
                metadata,
                "metadata",
            )
        else:
            metadata = {}

        human_approval = self._validate_bool(
            human_approval,
            "human_approval",
        )

        if (
            self.require_human_approval
            and not human_approval
        ):
            return SystemValidationResult(
                status=ValidationStatus.BLOCKED,
                version=version,
                criteria_results=(),
                passed_count=0,
                failed_count=0,
                blocked_count=1,
                total_count=len(
                    self.REQUIRED_CRITERIA
                ),
                metadata={
                    **dict(metadata),
                    "human_approval_required": True,
                    "human_approved": False,
                    "execution_started": False,
                },
                error=(
                    "System validation requires "
                    "human approval."
                ),
            )

        results: list[CriterionResult] = []

        for criterion in self.REQUIRED_CRITERIA:
            if criterion not in criteria:
                result = CriterionResult(
                    criterion=criterion,
                    status=ValidationStatus.FAILED,
                    evidence=(),
                    details=(
                        "Required criterion was not "
                        "provided."
                    ),
                )
            else:
                passed = self._validate_bool(
                    criteria[criterion],
                    f"criteria[{criterion.value}]",
                )

                criterion_evidence = evidence.get(
                    criterion,
                    (),
                )

                criterion_details = details.get(
                    criterion,
                    "",
                )

                result = self._criterion_result(
                    criterion,
                    passed,
                    self._validate_evidence(
                        criterion_evidence
                    ),
                    str(
                        criterion_details
                    ).strip(),
                )

            results.append(result)

        passed_count = sum(
            result.status
            == ValidationStatus.PASSED
            for result in results
        )

        failed_count = sum(
            result.status
            == ValidationStatus.FAILED
            for result in results
        )

        blocked_count = sum(
            result.status
            == ValidationStatus.BLOCKED
            for result in results
        )

        total_count = len(results)

        if failed_count > 0:
            overall_status = ValidationStatus.FAILED
        elif blocked_count > 0:
            overall_status = ValidationStatus.BLOCKED
        else:
            overall_status = ValidationStatus.PASSED

        if (
            not self.require_all_criteria
            and passed_count > 0
            and failed_count < total_count
        ):
            overall_status = ValidationStatus.PASSED

        combined_metadata = {
            **dict(metadata),
            "validation_version": version,
            "human_approval_required": (
                self.require_human_approval
            ),
            "human_approved": human_approval,
            "execution_started": True,
            "required_criteria": total_count,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "blocked_count": blocked_count,
            "all_criteria_evaluated": (
                total_count
                == len(self.REQUIRED_CRITERIA)
            ),
        }

        return SystemValidationResult(
            status=overall_status,
            version=version,
            criteria_results=tuple(results),
            passed_count=passed_count,
            failed_count=failed_count,
            blocked_count=blocked_count,
            total_count=total_count,
            metadata=combined_metadata,
            error=None,
        )

    def validate_full_system(
        self,
        *,
        provenance: bool,
        predictive_causal_separation: bool,
        oos_separation: bool,
        changes_measurable: bool,
        changes_reversible: bool,
        portfolio_reconstructable: bool,
        quantum_classical_benchmark: bool,
        autonomous_research_controlled: bool,
        end_to_end_acceptance: bool,
        full_regression: bool,
        evidence: Mapping[
            ValidationCriterion,
            Sequence[str],
        ]
        | None = None,
        human_approval: bool = False,
        metadata: Mapping[str, object]
        | None = None,
    ) -> SystemValidationResult:

        criteria = {
            ValidationCriterion.REPRODUCIBLE_PROVENANCE:
                provenance,
            ValidationCriterion.PREDICTIVE_CAUSAL_SEPARATION:
                predictive_causal_separation,
            ValidationCriterion.OOS_IS_SEPARATED:
                oos_separation,
            ValidationCriterion.CHANGES_MEASURABLE:
                changes_measurable,
            ValidationCriterion.CHANGES_REVERSIBLE:
                changes_reversible,
            ValidationCriterion.PORTFOLIO_RECONSTRUCTABLE:
                portfolio_reconstructable,
            ValidationCriterion.QUANTUM_CLASSICAL_BENCHMARK:
                quantum_classical_benchmark,
            ValidationCriterion.AUTONOMOUS_RESEARCH_CONTROLLED:
                autonomous_research_controlled,
            ValidationCriterion.END_TO_END_ACCEPTANCE:
                end_to_end_acceptance,
            ValidationCriterion.FULL_REGRESSION:
                full_regression,
        }

        return self.validate(
            version=self.VERSION,
            criteria=criteria,
            evidence=evidence,
            human_approval=human_approval,
            metadata=metadata,
        )


__all__ = [
    "ValidationStatus",
    "ValidationCriterion",
    "CriterionResult",
    "SystemValidationResult",
    "SystemValidationEngine",
]