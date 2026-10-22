"""
QUBO builder for the v5.0.8 Quantum Portfolio Lab.

This module converts a constrained binary portfolio-selection
problem into a Quadratic Unconstrained Binary Optimization
(QUBO) formulation.

Objective
---------

For binary asset-selection variables x_i in {0, 1}:

    risk_aversion * x^T Sigma x
    - return_weight * mu^T x

Exact-cardinality constraint
----------------------------

For target cardinality K:

    budget_penalty * (sum_i x_i - K)^2

The final QUBO has the form:

    x^T Q x + offset

The implementation uses a symmetric Q matrix. Therefore
off-diagonal interactions are represented symmetrically and
are counted twice by x^T Q x.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from quantum_portfolio_lab.qubo_models import (
    PortfolioQUBOProblem,
    QUBOVariable,
    QuantumPortfolioModelError,
)


class QUBOBuilderError(QuantumPortfolioModelError):
    """
    Raised when a portfolio QUBO request or construction
    operation is invalid.
    """


@dataclass(frozen=True)
class PortfolioSelectionQUBORequest:
    """
    Input specification for binary portfolio-selection QUBO.

    Parameters
    ----------
    asset_symbols:
        Ordered asset identifiers.

    expected_returns:
        Expected return vector mu.

    covariance_matrix:
        Covariance matrix Sigma.

    target_cardinality:
        Exact number of assets to select.

    risk_aversion:
        Weight applied to portfolio risk.

    return_weight:
        Weight applied to expected return.

    budget_penalty:
        Penalty strength for the exact-cardinality constraint.

    constraint_penalty:
        Generic constraint-penalty parameter preserved for
        later extensions of the Quantum Portfolio Lab.

    metadata:
        Optional request metadata.
    """

    asset_symbols: tuple[str, ...]
    expected_returns: np.ndarray
    covariance_matrix: np.ndarray
    target_cardinality: int

    risk_aversion: float = 1.0
    return_weight: float = 1.0
    budget_penalty: float = 1.0
    constraint_penalty: float = 1.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        # ------------------------------------------------------
        # Asset symbols
        # ------------------------------------------------------

        try:
            raw_symbols = tuple(
                self.asset_symbols
            )
        except TypeError as exc:
            raise QUBOBuilderError(
                "asset_symbols must be an iterable "
                "of non-empty strings."
            ) from exc

        if not raw_symbols:
            raise QUBOBuilderError(
                "asset_symbols cannot be empty."
            )

        cleaned_symbols: list[str] = []

        for symbol in raw_symbols:
            if not isinstance(symbol, str):
                raise QUBOBuilderError(
                    "Each asset symbol must be a "
                    "non-empty string."
                )

            cleaned = symbol.strip()

            if not cleaned:
                raise QUBOBuilderError(
                    "Each asset symbol must be a "
                    "non-empty string."
                )

            cleaned_symbols.append(
                cleaned
            )

        symbols = tuple(
            cleaned_symbols
        )

        if len(set(symbols)) != len(symbols):
            raise QUBOBuilderError(
                "asset_symbols must be unique."
            )

        asset_count = len(
            symbols
        )

        # ------------------------------------------------------
        # Expected returns
        # ------------------------------------------------------

        try:
            expected_returns = np.asarray(
                self.expected_returns,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise QUBOBuilderError(
                "expected_returns must contain "
                "numeric values."
            ) from exc

        if expected_returns.ndim != 1:
            raise QUBOBuilderError(
                "expected_returns must be "
                "one-dimensional."
            )

        if expected_returns.shape[0] != asset_count:
            raise QUBOBuilderError(
                "expected_returns length must match "
                "the number of asset_symbols."
            )

        if not np.all(
            np.isfinite(
                expected_returns
            )
        ):
            raise QUBOBuilderError(
                "expected_returns must contain only "
                "finite values."
            )

        # ------------------------------------------------------
        # Covariance matrix
        # ------------------------------------------------------

        try:
            covariance = np.asarray(
                self.covariance_matrix,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise QUBOBuilderError(
                "covariance_matrix must contain "
                "numeric values."
            ) from exc

        if covariance.ndim != 2:
            raise QUBOBuilderError(
                "covariance_matrix must be "
                "two-dimensional."
            )

        if covariance.shape != (
            asset_count,
            asset_count,
        ):
            raise QUBOBuilderError(
                "covariance_matrix dimension must match "
                "the number of asset_symbols."
            )

        if not np.all(
            np.isfinite(
                covariance
            )
        ):
            raise QUBOBuilderError(
                "covariance_matrix must contain only "
                "finite values."
            )

        if not np.allclose(
            covariance,
            covariance.T,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise QUBOBuilderError(
                "covariance_matrix must be symmetric."
            )

        # ------------------------------------------------------
        # Cardinality
        # ------------------------------------------------------

        if (
            isinstance(
                self.target_cardinality,
                bool,
            )
            or not isinstance(
                self.target_cardinality,
                (int, np.integer),
            )
        ):
            raise QUBOBuilderError(
                "target_cardinality must be a "
                "positive integer."
            )

        target_cardinality = int(
            self.target_cardinality
        )

        if target_cardinality <= 0:
            raise QUBOBuilderError(
                "target_cardinality must be positive."
            )

        if target_cardinality > asset_count:
            raise QUBOBuilderError(
                "target_cardinality cannot exceed "
                "the number of assets."
            )

        # ------------------------------------------------------
        # Numeric parameters
        # ------------------------------------------------------

        risk_aversion = self._validate_parameter(
            "risk_aversion",
            self.risk_aversion,
        )

        return_weight = self._validate_parameter(
            "return_weight",
            self.return_weight,
        )

        budget_penalty = self._validate_parameter(
            "budget_penalty",
            self.budget_penalty,
        )

        constraint_penalty = self._validate_parameter(
            "constraint_penalty",
            self.constraint_penalty,
        )

        # ------------------------------------------------------
        # Defensive copies
        # ------------------------------------------------------

        expected_returns = np.array(
            expected_returns,
            dtype=float,
            copy=True,
        )

        covariance = np.array(
            covariance,
            dtype=float,
            copy=True,
        )

        metadata = dict(
            self.metadata
        )

        object.__setattr__(
            self,
            "asset_symbols",
            symbols,
        )

        object.__setattr__(
            self,
            "expected_returns",
            expected_returns,
        )

        object.__setattr__(
            self,
            "covariance_matrix",
            covariance,
        )

        object.__setattr__(
            self,
            "target_cardinality",
            target_cardinality,
        )

        object.__setattr__(
            self,
            "risk_aversion",
            risk_aversion,
        )

        object.__setattr__(
            self,
            "return_weight",
            return_weight,
        )

        object.__setattr__(
            self,
            "budget_penalty",
            budget_penalty,
        )

        object.__setattr__(
            self,
            "constraint_penalty",
            constraint_penalty,
        )

        object.__setattr__(
            self,
            "metadata",
            metadata,
        )

    @staticmethod
    def _validate_parameter(
        field_name: str,
        value: float,
    ) -> float:
        try:
            numeric_value = float(
                value
            )
        except (TypeError, ValueError) as exc:
            raise QUBOBuilderError(
                f"{field_name} must be a finite "
                "non-negative number."
            ) from exc

        if not np.isfinite(
            numeric_value
        ):
            raise QUBOBuilderError(
                f"{field_name} must be finite."
            )

        if numeric_value < 0.0:
            raise QUBOBuilderError(
                f"{field_name} cannot be negative."
            )

        return numeric_value


class PortfolioQUBOBuilder:
    """
    Construct QUBO models for constrained binary
    portfolio-selection problems.
    """

    # ==========================================================
    # Public construction API
    # ==========================================================

    @classmethod
    def build(
        cls,
        request: PortfolioSelectionQUBORequest,
    ) -> PortfolioQUBOProblem:
        """
        Build a PortfolioQUBOProblem from a validated request.
        """

        if not isinstance(
            request,
            PortfolioSelectionQUBORequest,
        ):
            raise QUBOBuilderError(
                "request must be a "
                "PortfolioSelectionQUBORequest."
            )

        variable_count = len(
            request.asset_symbols
        )

        variables = cls._build_variables(
            request.asset_symbols
        )

        objective_matrix = (
            cls._build_objective_matrix(
                expected_returns=(
                    request.expected_returns
                ),
                covariance_matrix=(
                    request.covariance_matrix
                ),
                risk_aversion=(
                    request.risk_aversion
                ),
                return_weight=(
                    request.return_weight
                ),
            )
        )

        (
            cardinality_matrix,
            cardinality_offset,
        ) = cls._build_cardinality_penalty(
            variable_count=variable_count,
            target_cardinality=(
                request.target_cardinality
            ),
            penalty=(
                request.budget_penalty
            ),
        )

        qubo_matrix = (
            objective_matrix
            + cardinality_matrix
        )

        # Numerical symmetry guard.
        qubo_matrix = (
            qubo_matrix
            + qubo_matrix.T
        ) / 2.0

        metadata = dict(
            request.metadata
        )

        metadata.update(
            {
                "formulation": (
                    "binary_portfolio_selection"
                ),
                "objective": (
                    "risk_minus_return"
                ),
                "cardinality_constraint": (
                    "exact"
                ),
                "variable_count": (
                    variable_count
                ),
            }
        )

        return PortfolioQUBOProblem(
            variables=variables,
            qubo_matrix=qubo_matrix,
            offset=float(
                cardinality_offset
            ),
            asset_symbols=tuple(
                request.asset_symbols
            ),
            expected_returns=np.array(
                request.expected_returns,
                dtype=float,
                copy=True,
            ),
            covariance_matrix=np.array(
                request.covariance_matrix,
                dtype=float,
                copy=True,
            ),
            risk_aversion=float(
                request.risk_aversion
            ),
            return_weight=float(
                request.return_weight
            ),
            budget_penalty=float(
                request.budget_penalty
            ),
            constraint_penalty=float(
                request.constraint_penalty
            ),
            target_cardinality=int(
                request.target_cardinality
            ),
            metadata=metadata,
        )

    # ==========================================================
    # Objective evaluation
    # ==========================================================

    @classmethod
    def evaluate_base_objective(
        cls,
        *,
        request: PortfolioSelectionQUBORequest,
        binary_vector: Sequence[int | float],
    ) -> float:
        """
        Evaluate:

            risk_aversion * x^T Sigma x
            - return_weight * mu^T x
        """

        cls._validate_request(
            request
        )

        vector = cls._validate_binary_vector(
            binary_vector=binary_vector,
            variable_count=len(
                request.asset_symbols
            ),
        )

        risk = float(
            vector
            @ request.covariance_matrix
            @ vector
        )

        expected_return = float(
            request.expected_returns
            @ vector
        )

        objective = (
            request.risk_aversion
            * risk
            - request.return_weight
            * expected_return
        )

        return float(
            objective
        )

    @classmethod
    def evaluate_cardinality_violation(
        cls,
        *,
        request: PortfolioSelectionQUBORequest,
        binary_vector: Sequence[int | float],
    ) -> float:
        """
        Return absolute exact-cardinality violation:

            |sum(x) - K|
        """

        cls._validate_request(
            request
        )

        vector = cls._validate_binary_vector(
            binary_vector=binary_vector,
            variable_count=len(
                request.asset_symbols
            ),
        )

        difference = (
            float(
                np.sum(
                    vector
                )
            )
            - float(
                request.target_cardinality
            )
        )

        return float(
            abs(
                difference
            )
        )

    @classmethod
    def evaluate_cardinality_penalty(
        cls,
        *,
        request: PortfolioSelectionQUBORequest,
        binary_vector: Sequence[int | float],
    ) -> float:
        """
        Evaluate exact-cardinality squared penalty:

            penalty * (sum(x) - K)^2
        """

        cls._validate_request(
            request
        )

        vector = cls._validate_binary_vector(
            binary_vector=binary_vector,
            variable_count=len(
                request.asset_symbols
            ),
        )

        difference = (
            float(
                np.sum(
                    vector
                )
            )
            - float(
                request.target_cardinality
            )
        )

        penalty = (
            request.budget_penalty
            * difference
            * difference
        )

        return float(
            penalty
        )

    @classmethod
    def evaluate_total_objective(
        cls,
        *,
        request: PortfolioSelectionQUBORequest,
        binary_vector: Sequence[int | float],
    ) -> float:
        """
        Evaluate original constrained formulation:

            base objective
            +
            exact-cardinality penalty
        """

        base_objective = (
            cls.evaluate_base_objective(
                request=request,
                binary_vector=binary_vector,
            )
        )

        cardinality_penalty = (
            cls.evaluate_cardinality_penalty(
                request=request,
                binary_vector=binary_vector,
            )
        )

        return float(
            base_objective
            + cardinality_penalty
        )

    # ==========================================================
    # QUBO construction helpers
    # ==========================================================

    @staticmethod
    def _build_variables(
        asset_symbols: Sequence[str],
    ) -> tuple[QUBOVariable, ...]:
        """
        Create one binary QUBO variable per asset.
        """

        variables: list[
            QUBOVariable
        ] = []

        for index, symbol in enumerate(
            asset_symbols
        ):
            variables.append(
                QUBOVariable(
                    name=f"x_{symbol}",
                    index=index,
                    asset_symbol=symbol,
                    metadata={
                        "variable_type": (
                            "asset_selection"
                        ),
                    },
                )
            )

        return tuple(
            variables
        )

    @staticmethod
    def _build_objective_matrix(
        *,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float,
        return_weight: float,
    ) -> np.ndarray:
        """
        Construct the risk-return QUBO matrix.

        Original objective:

            lambda * x^T Sigma x
            - alpha * mu^T x

        Since x_i^2 = x_i for binary variables, the linear
        expected-return term is represented on the diagonal.
        """

        expected_returns = np.asarray(
            expected_returns,
            dtype=float,
        )

        covariance_matrix = np.asarray(
            covariance_matrix,
            dtype=float,
        )

        matrix = (
            float(
                risk_aversion
            )
            * covariance_matrix.copy()
        )

        diagonal_indices = np.diag_indices(
            expected_returns.shape[0]
        )

        matrix[
            diagonal_indices
        ] -= (
            float(
                return_weight
            )
            * expected_returns
        )

        return np.asarray(
            matrix,
            dtype=float,
        )

    @staticmethod
    def _build_cardinality_penalty(
        *,
        variable_count: int,
        target_cardinality: int,
        penalty: float,
    ) -> tuple[np.ndarray, float]:
        """
        Build QUBO representation of:

            penalty * (sum_i x_i - K)^2

        Expanding:

            penalty * [
                sum_i x_i
                + 2 sum_{i<j} x_i x_j
                - 2K sum_i x_i
                + K^2
            ]

        Therefore:

            diagonal coefficient:
                penalty * (1 - 2K)

            polynomial pair coefficient:
                2 * penalty

        Because x^T Q x with symmetric Q counts each
        off-diagonal interaction twice:

            Q[i, j] = penalty
            Q[j, i] = penalty

        and:

            offset = penalty * K^2
        """

        if (
            isinstance(
                variable_count,
                bool,
            )
            or not isinstance(
                variable_count,
                (int, np.integer),
            )
            or variable_count <= 0
        ):
            raise QUBOBuilderError(
                "variable_count must be a "
                "positive integer."
            )

        if (
            isinstance(
                target_cardinality,
                bool,
            )
            or not isinstance(
                target_cardinality,
                (int, np.integer),
            )
            or target_cardinality <= 0
        ):
            raise QUBOBuilderError(
                "target_cardinality must be positive."
            )

        if target_cardinality > variable_count:
            raise QUBOBuilderError(
                "target_cardinality cannot exceed "
                "variable_count."
            )

        try:
            penalty_value = float(
                penalty
            )
        except (TypeError, ValueError) as exc:
            raise QUBOBuilderError(
                "penalty must be a finite "
                "non-negative number."
            ) from exc

        if not np.isfinite(
            penalty_value
        ):
            raise QUBOBuilderError(
                "penalty must be finite."
            )

        if penalty_value < 0.0:
            raise QUBOBuilderError(
                "penalty cannot be negative."
            )

        matrix = np.zeros(
            (
                variable_count,
                variable_count,
            ),
            dtype=float,
        )

        diagonal_coefficient = (
            penalty_value
            * (
                1.0
                - 2.0
                * target_cardinality
            )
        )

        for index in range(
            variable_count
        ):
            matrix[
                index,
                index,
            ] = diagonal_coefficient

        for first in range(
            variable_count
        ):
            for second in range(
                first + 1,
                variable_count,
            ):
                matrix[
                    first,
                    second,
                ] = penalty_value

                matrix[
                    second,
                    first,
                ] = penalty_value

        offset = (
            penalty_value
            * target_cardinality**2
        )

        return (
            matrix,
            float(
                offset
            ),
        )

    # ==========================================================
    # Validation helpers
    # ==========================================================

    @staticmethod
    def _validate_request(
        request: PortfolioSelectionQUBORequest,
    ) -> None:
        if not isinstance(
            request,
            PortfolioSelectionQUBORequest,
        ):
            raise QUBOBuilderError(
                "request must be a "
                "PortfolioSelectionQUBORequest."
            )

    @staticmethod
    def _validate_binary_vector(
        *,
        binary_vector: Sequence[int | float],
        variable_count: int,
    ) -> np.ndarray:
        """
        Validate and normalize a binary decision vector.
        """

        try:
            vector = np.asarray(
                binary_vector,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise QUBOBuilderError(
                "binary_vector must contain "
                "numeric values."
            ) from exc

        if vector.ndim != 1:
            raise QUBOBuilderError(
                "binary_vector must be "
                "one-dimensional."
            )

        if vector.shape[0] != variable_count:
            raise QUBOBuilderError(
                "binary_vector length must match "
                "the number of assets."
            )

        if not np.all(
            np.isfinite(
                vector
            )
        ):
            raise QUBOBuilderError(
                "binary_vector must contain "
                "finite values."
            )

        if not np.all(
            np.logical_or(
                vector == 0.0,
                vector == 1.0,
            )
        ):
            raise QUBOBuilderError(
                "binary_vector must contain "
                "only 0 or 1."
            )

        return np.array(
            vector,
            dtype=float,
            copy=True,
        )


__all__ = [
    "PortfolioQUBOBuilder",
    "PortfolioSelectionQUBORequest",
    "QUBOBuilderError",
]