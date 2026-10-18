"""Tests for submission validation, ID alignment, and CSV export."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.submission.submission_writer import SubmissionWriter


@pytest.fixture
def sample_path(tmp_path: Path) -> Path:
    """Create a template containing string IDs with leading zeros."""
    path = tmp_path / "sample_submission.csv"

    pd.DataFrame(
        {
            "id": ["001", "002", "003"],
            "target": [0.5, 0.5, 0.5],
        }
    ).to_csv(path, index=False)

    return path


@pytest.fixture
def test_ids() -> pd.DataFrame:
    """Use a different order and index from the submission template."""
    return pd.DataFrame(
        {"id": ["003", "001", "002"]},
        index=[30, 10, 20],
    )


@pytest.fixture
def probabilities() -> np.ndarray:
    """Match probabilities positionally to test_ids, not the template."""
    return np.array([0.3, 0.1, 0.2])


@pytest.fixture
def writer(sample_path: Path) -> SubmissionWriter:
    return SubmissionWriter(sample_path)


def test_write_aligns_predictions_by_id(
    writer: SubmissionWriter,
    test_ids: pd.DataFrame,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    output = tmp_path / "submission.csv"

    returned_path = writer.write(
        test_ids,
        probabilities,
        output,
    )

    submission = pd.read_csv(output, dtype={"id": "string"})

    assert returned_path == output
    assert submission.columns.tolist() == ["id", "target"]
    assert submission["id"].tolist() == ["001", "002", "003"]

    np.testing.assert_allclose(
        submission["target"].to_numpy(),
        [0.1, 0.2, 0.3],
    )


def test_write_does_not_export_dataframe_index(
    writer: SubmissionWriter,
    test_ids: pd.DataFrame,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    output = tmp_path / "submission.csv"

    writer.write(test_ids, probabilities, output)

    header = output.read_text(encoding="utf-8").splitlines()[0]

    assert header == "id,target"

    submission = pd.read_csv(output)

    assert submission.shape == (3, 2)
    assert not any(
        column.startswith("Unnamed:")
        for column in submission.columns
    )


def test_write_creates_parent_directories(
    writer: SubmissionWriter,
    test_ids: pd.DataFrame,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "experiment" / "submission.csv"

    writer.write(test_ids, probabilities, output)

    assert output.is_file()


def test_zero_and_one_probabilities_are_allowed(
    writer: SubmissionWriter,
    test_ids: pd.DataFrame,
    tmp_path: Path,
) -> None:
    output = tmp_path / "submission.csv"

    writer.write(
        test_ids,
        np.array([0.0, 1.0, 0.5]),
        output,
    )

    submission = pd.read_csv(output)

    # test_ids order: 003, 001, 002
    # Template order: 001, 002, 003
    np.testing.assert_allclose(
        submission["target"].to_numpy(),
        [1.0, 0.5, 0.0],
    )


def test_inputs_and_template_are_not_modified(
    writer: SubmissionWriter,
    sample_path: Path,
    test_ids: pd.DataFrame,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    ids_before = test_ids.copy(deep=True)
    probabilities_before = probabilities.copy()
    template_before = sample_path.read_bytes()

    writer.write(
        test_ids,
        probabilities,
        tmp_path / "submission.csv",
    )

    pd.testing.assert_frame_equal(test_ids, ids_before)
    np.testing.assert_array_equal(
        probabilities,
        probabilities_before,
    )
    assert sample_path.read_bytes() == template_before


@pytest.mark.parametrize(
    "invalid_probabilities",
    [
        np.array(0.5),
        np.array([[0.1], [0.2], [0.3]]),
        np.array([[0.1, 0.2, 0.3]]),
    ],
)
def test_non_vector_probabilities_rejected(
    invalid_probabilities: np.ndarray,
    writer: SubmissionWriter,
    test_ids: pd.DataFrame,
    tmp_path: Path,
) -> None:
    output = tmp_path / "submission.csv"

    with pytest.raises(ValueError, match="one-dimensional"):
        writer.write(test_ids, invalid_probabilities, output)

    assert not output.exists()


@pytest.mark.parametrize(
    "invalid_probabilities",
    [
        np.array([0.1, 0.2]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([]),
    ],
)
def test_probability_count_mismatch_rejected(
    invalid_probabilities: np.ndarray,
    writer: SubmissionWriter,
    test_ids: pd.DataFrame,
    tmp_path: Path,
) -> None:
    output = tmp_path / "submission.csv"

    with pytest.raises(ValueError, match="equal lengths"):
        writer.write(test_ids, invalid_probabilities, output)

    assert not output.exists()


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_nonfinite_probabilities_rejected(
    invalid_value: float,
    writer: SubmissionWriter,
    test_ids: pd.DataFrame,
    tmp_path: Path,
) -> None:
    output = tmp_path / "submission.csv"
    scores = np.array([0.1, invalid_value, 0.3])

    with pytest.raises(ValueError, match="finite"):
        writer.write(test_ids, scores, output)

    assert not output.exists()


@pytest.mark.parametrize("invalid_value", [-0.001, 1.001])
def test_out_of_range_probabilities_rejected(
    invalid_value: float,
    writer: SubmissionWriter,
    test_ids: pd.DataFrame,
    tmp_path: Path,
) -> None:
    output = tmp_path / "submission.csv"
    scores = np.array([0.1, invalid_value, 0.3])

    with pytest.raises(ValueError, match="between 0 and 1"):
        writer.write(test_ids, scores, output)

    assert not output.exists()


def test_missing_id_column_rejected(
    writer: SubmissionWriter,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    ids = pd.DataFrame({"wrong_column": ["001", "002", "003"]})

    with pytest.raises(ValueError, match="contain an 'id' column"):
        writer.write(
            ids,
            probabilities,
            tmp_path / "submission.csv",
        )


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (["003", None, "002"], "missing values"),
        (["003", "003", "002"], "unique"),
        (["003", "999", "002"], "do not match"),
    ],
)
def test_invalid_test_ids_rejected(
    ids: list,
    message: str,
    writer: SubmissionWriter,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    output = tmp_path / "submission.csv"

    with pytest.raises(ValueError, match=message):
        writer.write(
            pd.DataFrame({"id": ids}),
            probabilities,
            output,
        )

    assert not output.exists()


def test_test_id_count_mismatch_rejected(
    writer: SubmissionWriter,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    ids = pd.DataFrame({"id": ["001", "002"]})

    with pytest.raises(ValueError, match="equal lengths"):
        writer.write(
            ids,
            probabilities,
            tmp_path / "submission.csv",
        )


@pytest.mark.parametrize(
    "columns",
    [
        ["target", "id"],
        ["id", "prediction"],
        ["id", "target", "extra"],
        ["target"],
    ],
)
def test_invalid_template_columns_rejected(
    columns: list[str],
    sample_path: Path,
    test_ids: pd.DataFrame,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    template = pd.DataFrame(
        {
            column: (
                ["001", "002", "003"]
                if column == "id"
                else [0.5, 0.5, 0.5]
            )
            for column in columns
        }
    )
    template.to_csv(sample_path, index=False)

    with pytest.raises(ValueError, match="must have columns"):
        SubmissionWriter(sample_path).write(
            test_ids,
            probabilities,
            tmp_path / "submission.csv",
        )


def test_empty_template_rejected(
    sample_path: Path,
    test_ids: pd.DataFrame,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    pd.DataFrame(columns=["id", "target"]).to_csv(
        sample_path,
        index=False,
    )

    with pytest.raises(ValueError, match="must not be empty"):
        SubmissionWriter(sample_path).write(
            test_ids,
            probabilities,
            tmp_path / "submission.csv",
        )


@pytest.mark.parametrize(
    ("template_ids", "message"),
    [
        (["001", None, "003"], "missing values"),
        (["001", "001", "003"], "unique"),
    ],
)
def test_invalid_template_ids_rejected(
    template_ids: list,
    message: str,
    sample_path: Path,
    test_ids: pd.DataFrame,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "id": template_ids,
            "target": [0.5, 0.5, 0.5],
        }
    ).to_csv(sample_path, index=False)

    with pytest.raises(ValueError, match=message):
        SubmissionWriter(sample_path).write(
            test_ids,
            probabilities,
            tmp_path / "submission.csv",
        )


def test_existing_submission_is_not_overwritten(
    writer: SubmissionWriter,
    test_ids: pd.DataFrame,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    output = tmp_path / "submission.csv"
    original_content = "existing submission\n"
    output.write_text(original_content, encoding="utf-8")

    with pytest.raises(FileExistsError):
        writer.write(test_ids, probabilities, output)

    assert output.read_text(encoding="utf-8") == original_content


def test_missing_template_file_rejected(
    test_ids: pd.DataFrame,
    probabilities: np.ndarray,
    tmp_path: Path,
) -> None:
    writer = SubmissionWriter(tmp_path / "missing_sample.csv")
    output = tmp_path / "submission.csv"

    with pytest.raises(FileNotFoundError):
        writer.write(test_ids, probabilities, output)

    assert not output.exists()