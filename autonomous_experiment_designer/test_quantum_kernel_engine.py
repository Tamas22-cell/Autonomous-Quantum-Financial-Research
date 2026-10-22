import pytest

from autonomous_experiment_designer.quantum_kernel_engine import (
    QuantumKernelEngine,
    QuantumKernelExecutionStatus,
    QuantumKernelMethod,
    QuantumKernelResult,
)


def linear_kernel(left, right):
    return (
        sum(
            a * b
            for a, b in zip(left, right)
        )
        + 1.0
    ) / 5.0


def similarity_kernel(left, right):
    distance = sum(
        (a - b) ** 2
        for a, b in zip(left, right)
    )

    return 1.0 / (1.0 + distance)


def test_default_configuration():
    engine = QuantumKernelEngine()

    assert engine.feature_dimension == 2
    assert engine.method == QuantumKernelMethod.EXACT
    assert engine.max_evaluations == 100
    assert engine.diagonal_value == 1.0


def test_custom_configuration():
    engine = QuantumKernelEngine(
        feature_dimension=4,
        method=QuantumKernelMethod.APPROXIMATE,
        max_evaluations=25,
        diagonal_value=0.95,
    )

    assert engine.feature_dimension == 4
    assert engine.method == QuantumKernelMethod.APPROXIMATE
    assert engine.max_evaluations == 25
    assert engine.diagonal_value == 0.95


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "2"],
)
def test_invalid_feature_dimension_is_rejected(value):
    with pytest.raises(ValueError):
        QuantumKernelEngine(
            feature_dimension=value
        )


@pytest.mark.parametrize(
    "value",
    [None, "EXACT", 123],
)
def test_invalid_method_is_rejected(value):
    with pytest.raises(ValueError):
        QuantumKernelEngine(
            method=value
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "10"],
)
def test_invalid_max_evaluations_is_rejected(value):
    with pytest.raises(ValueError):
        QuantumKernelEngine(
            max_evaluations=value
        )


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, float("nan"), float("inf"), True],
)
def test_invalid_diagonal_value_is_rejected(value):
    with pytest.raises(ValueError):
        QuantumKernelEngine(
            diagonal_value=value
        )


def test_exact_kernel_computation():
    engine = QuantumKernelEngine(
        feature_dimension=2,
        method=QuantumKernelMethod.EXACT,
    )

    dataset = (
        (1.0, 0.0),
        (0.0, 1.0),
    )

    result = engine.compute(
        dataset,
        similarity_kernel,
    )

    assert isinstance(
        result,
        QuantumKernelResult,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.COMPLETED
    )

    assert result.method == QuantumKernelMethod.EXACT

    assert result.kernel_matrix == (
        (1.0, 1.0 / 3.0),
        (1.0 / 3.0, 1.0),
    )

    assert result.similarity_scores == (
        1.0 / 3.0,
    )

    assert result.average_similarity == 1.0 / 3.0
    assert result.evaluations == 4


def test_approximate_kernel_computation():
    engine = QuantumKernelEngine(
        feature_dimension=2,
        method=QuantumKernelMethod.APPROXIMATE,
    )

    dataset = (
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    )

    result = engine.compute(
        dataset,
        similarity_kernel,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.COMPLETED
    )

    assert result.kernel_matrix[0][0] == 1.0
    assert result.kernel_matrix[1][1] == 1.0
    assert result.kernel_matrix[2][2] == 1.0

    assert (
        result.kernel_matrix[0][1]
        == result.kernel_matrix[1][0]
    )

    assert (
        result.kernel_matrix[0][2]
        == result.kernel_matrix[2][0]
    )

    assert (
        result.kernel_matrix[1][2]
        == result.kernel_matrix[2][1]
    )

    assert result.evaluations == 6


def test_exact_evaluates_full_matrix():
    engine = QuantumKernelEngine(
        feature_dimension=2,
        method=QuantumKernelMethod.EXACT,
    )

    dataset = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )

    result = engine.compute(
        dataset,
        similarity_kernel,
    )

    assert result.evaluations == 9


def test_approximate_evaluates_upper_triangle():
    engine = QuantumKernelEngine(
        feature_dimension=2,
        method=QuantumKernelMethod.APPROXIMATE,
    )

    dataset = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )

    result = engine.compute(
        dataset,
        similarity_kernel,
    )

    assert result.evaluations == 6


def test_kernel_matrix_is_symmetric():
    engine = QuantumKernelEngine()

    dataset = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )

    result = engine.compute(
        dataset,
        similarity_kernel,
    )

    matrix = result.kernel_matrix

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            assert matrix[i][j] == matrix[j][i]


def test_metadata_reports_symmetry():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 1.0),
        ),
        similarity_kernel,
    )

    assert result.metadata["symmetric"] is True


def test_default_diagonal_is_one():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 1.0),
        ),
        similarity_kernel,
    )

    assert result.kernel_matrix[0][0] == 1.0
    assert result.kernel_matrix[1][1] == 1.0


def test_custom_diagonal_is_preserved():
    engine = QuantumKernelEngine(
        diagonal_value=0.9
    )

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 1.0),
        ),
        similarity_kernel,
    )

    assert result.kernel_matrix[0][0] == 0.9
    assert result.kernel_matrix[1][1] == 0.9


def test_similarity_scores_exclude_diagonal():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        similarity_kernel,
    )

    assert len(result.similarity_scores) == 3


def test_average_similarity_is_computed():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 0.0),
        ),
        similarity_kernel,
    )

    assert result.average_similarity == 0.5


def test_single_sample_has_no_pairwise_similarity():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (1.0, 0.0),
        ),
        similarity_kernel,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.COMPLETED
    )

    assert result.similarity_scores == ()
    assert result.average_similarity is None
    assert result.evaluations == 1


@pytest.mark.parametrize(
    "dataset",
    [
        None,
        (),
        "invalid",
        123,
    ],
)
def test_invalid_dataset_is_captured(dataset):
    engine = QuantumKernelEngine()

    result = engine.compute(
        dataset,
        similarity_kernel,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.FAILED
    )

    assert result.kernel_matrix == ()
    assert result.similarity_scores == ()
    assert result.average_similarity is None
    assert result.error is not None


@pytest.mark.parametrize(
    "dataset",
    [
        (
            (1.0,),
            (0.0,),
        ),
        (
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 2.0),
        ),
    ],
)
def test_invalid_feature_dimension_is_captured(dataset):
    engine = QuantumKernelEngine(
        feature_dimension=2
    )

    result = engine.compute(
        dataset,
        similarity_kernel,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.FAILED
    )

    assert result.error is not None


def test_string_feature_vector_is_rejected():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            "invalid",
            (1.0, 0.0),
        ),
        similarity_kernel,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.FAILED
    )

    assert result.error is not None


def test_non_numeric_feature_is_rejected():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (1.0, "invalid"),
            (0.0, 1.0),
        ),
        similarity_kernel,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.FAILED
    )

    assert result.error is not None


def test_nonfinite_feature_is_rejected():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (float("nan"), 0.0),
            (0.0, 1.0),
        ),
        similarity_kernel,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.FAILED
    )

    assert result.error is not None


@pytest.mark.parametrize(
    "kernel_function",
    [
        None,
        "invalid",
        123,
    ],
)
def test_invalid_kernel_function_is_rejected(
    kernel_function,
):
    engine = QuantumKernelEngine()

    with pytest.raises(ValueError):
        engine.compute(
            (
                (0.0, 0.0),
                (1.0, 1.0),
            ),
            kernel_function,
        )


def test_kernel_returning_non_numeric_value_fails():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 1.0),
        ),
        lambda left, right: "invalid",
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.FAILED
    )

    assert result.error is not None


def test_kernel_returning_boolean_fails():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 1.0),
        ),
        lambda left, right: True,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.FAILED
    )

    assert result.error is not None


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_kernel_value_fails(value):
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 1.0),
        ),
        lambda left, right: value,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.FAILED
    )

    assert result.error is not None


def test_kernel_exception_is_captured():
    engine = QuantumKernelEngine()

    def failing_kernel(left, right):
        raise RuntimeError(
            "controlled kernel failure"
        )

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 1.0),
        ),
        failing_kernel,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.FAILED
    )

    assert result.kernel_matrix == ()
    assert result.evaluations == 0
    assert (
        "controlled kernel failure"
        in result.error
    )


def test_evaluation_limit_is_enforced():
    engine = QuantumKernelEngine(
        feature_dimension=2,
        max_evaluations=3,
    )

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        similarity_kernel,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.FAILED
    )

    assert result.error is not None


def test_approximate_mode_requires_fewer_evaluations():
    dataset = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    )

    exact = QuantumKernelEngine(
        method=QuantumKernelMethod.EXACT
    )

    approximate = QuantumKernelEngine(
        method=QuantumKernelMethod.APPROXIMATE
    )

    exact_result = exact.compute(
        dataset,
        similarity_kernel,
    )

    approximate_result = approximate.compute(
        dataset,
        similarity_kernel,
    )

    assert (
        approximate_result.evaluations
        < exact_result.evaluations
    )


def test_exact_and_approximate_have_same_matrix():
    dataset = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )

    exact = QuantumKernelEngine(
        method=QuantumKernelMethod.EXACT
    )

    approximate = QuantumKernelEngine(
        method=QuantumKernelMethod.APPROXIMATE
    )

    exact_result = exact.compute(
        dataset,
        similarity_kernel,
    )

    approximate_result = approximate.compute(
        dataset,
        similarity_kernel,
    )

    assert (
        exact_result.kernel_matrix
        == approximate_result.kernel_matrix
    )


def test_result_is_immutable():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 1.0),
        ),
        similarity_kernel,
    )

    with pytest.raises(AttributeError):
        result.status = (
            QuantumKernelExecutionStatus.FAILED
        )


def test_result_contains_metadata():
    engine = QuantumKernelEngine(
        feature_dimension=3,
        max_evaluations=20,
        diagonal_value=0.9,
    )

    result = engine.compute(
        (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ),
        similarity_kernel,
    )

    assert (
        result.metadata["feature_dimension"]
        == 3
    )

    assert (
        result.metadata["max_evaluations"]
        == 20
    )

    assert (
        result.metadata["diagonal_value"]
        == 0.9
    )

    assert (
        result.metadata["dataset_size"]
        == 2
    )


def test_kernel_function_receives_tuples():
    engine = QuantumKernelEngine()

    received_types = []

    def kernel(left, right):
        received_types.append(
            (type(left), type(right))
        )

        return similarity_kernel(
            left,
            right,
        )

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 1.0),
        ),
        kernel,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.COMPLETED
    )

    assert all(
        left_type is tuple
        and right_type is tuple
        for left_type, right_type
        in received_types
    )


def test_input_dataset_is_not_modified():
    dataset = [
        [0.0, 0.0],
        [1.0, 1.0],
    ]

    original = [
        [0.0, 0.0],
        [1.0, 1.0],
    ]

    engine = QuantumKernelEngine()

    engine.compute(
        dataset,
        similarity_kernel,
    )

    assert dataset == original


def test_kernel_result_is_repeatable():
    engine = QuantumKernelEngine()

    dataset = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )

    first = engine.compute(
        dataset,
        similarity_kernel,
    )

    second = engine.compute(
        dataset,
        similarity_kernel,
    )

    assert first == second


def test_linear_kernel_produces_expected_similarity():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        linear_kernel,
    )

    assert (
        result.status
        == QuantumKernelExecutionStatus.COMPLETED
    )

    assert result.kernel_matrix[0][1] == 0.2
    assert result.kernel_matrix[1][0] == 0.2


def test_average_similarity_for_multiple_samples():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        similarity_kernel,
    )

    expected = (
        0.5
        + 0.5
        + 1.0 / 3.0
    ) / 3.0

    assert (
        abs(
            result.average_similarity
            - expected
        )
        < 1e-12
    )


def test_dataset_size_is_recorded():
    engine = QuantumKernelEngine()

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ),
        similarity_kernel,
    )

    assert (
        result.metadata["dataset_size"]
        == 4
    )


def test_exact_method_is_explicitly_recorded():
    engine = QuantumKernelEngine(
        method=QuantumKernelMethod.EXACT
    )

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 1.0),
        ),
        similarity_kernel,
    )

    assert (
        result.method
        == QuantumKernelMethod.EXACT
    )


def test_approximate_method_is_explicitly_recorded():
    engine = QuantumKernelEngine(
        method=QuantumKernelMethod.APPROXIMATE
    )

    result = engine.compute(
        (
            (0.0, 0.0),
            (1.0, 1.0),
        ),
        similarity_kernel,
    )

    assert (
        result.method
        == QuantumKernelMethod.APPROXIMATE
    )