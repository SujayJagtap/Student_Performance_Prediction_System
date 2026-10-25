"""Load the selected Module 3 score model and return score and level predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MOCK_TO_DATASET = {
    "mock_math_score": "math_score",
    "mock_reading_score": "reading_score",
    "mock_writing_score": "writing_score",
    "mock_science_score": "science_score",
}


def predict_students(frame: pd.DataFrame, use_mock_marks: bool = False) -> pd.DataFrame:
    """Predict with the baseline model or the opt-in earlier-marks model."""
    metadata_file = "mock_marks_model_metadata.json" if use_mock_marks else "model_metadata.json"
    model_file = "score_model_with_mock_marks.joblib" if use_mock_marks else "score_model.joblib"
    metadata = json.loads((ARTIFACTS / metadata_file).read_text())
    model_input = frame.copy()
    if use_mock_marks:
        missing_marks = sorted(set(MOCK_TO_DATASET) - set(model_input.columns))
        if missing_marks:
            raise ValueError(f"All four earlier mock marks are required: {missing_marks}")
        marks = model_input[list(MOCK_TO_DATASET)]
        if marks.isna().any().any():
            raise ValueError("All four earlier mock marks must be filled in.")
        if not marks.apply(lambda column: pd.to_numeric(column, errors="coerce").between(0, 100)).all().all():
            raise ValueError("Earlier mock marks must be numbers between 0 and 100.")
        model_input = model_input.drop(columns=list(MOCK_TO_DATASET)).drop(
            columns=list(MOCK_TO_DATASET.values()), errors="ignore"
        ).assign(**{dataset: pd.to_numeric(frame[mock]) for mock, dataset in MOCK_TO_DATASET.items()})
    required = metadata["raw_model_features"]
    missing = sorted(set(required) - set(model_input.columns))
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    model = joblib.load(ARTIFACTS / model_file)
    raw_scores = np.asarray(model.predict(model_input[required]), dtype=float)
    # A score shown to a user must stay on the exam's 0–100 scale.
    scores = np.clip(raw_scores, 0, 100)
    displayed_scores = np.round(scores, 2)
    levels = np.select([displayed_scores < 50, displayed_scores <= 65], ["Low", "Average"], default="High")
    result = pd.DataFrame({"predicted_score": displayed_scores, "predicted_level": levels}, index=frame.index)
    if "student_id" in frame.columns:
        result.insert(0, "student_id", frame["student_id"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict final exam scores from raw student inputs.")
    parser.add_argument("input_csv", type=Path, help="CSV with the 11 Module 1 model input fields")
    parser.add_argument("output_csv", type=Path, help="Where to write predicted_score and predicted_level")
    parser.add_argument("--use-mock-marks", action="store_true", help="Use all four mock_*_score columns with the optional model")
    args = parser.parse_args()
    input_frame = pd.read_csv(args.input_csv)
    predictions = predict_students(input_frame, use_mock_marks=args.use_mock_marks)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_csv, index=False)
    print(f"Saved {len(predictions):,} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
