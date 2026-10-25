"""Module 1: validate, clean, split, encode, and scale student exam data.

Run from any directory with: python preprocess.py
The input is data/raw/student_exam_performance_dataset.csv beside this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "student_exam_performance_dataset.csv"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

SUBJECT_COLUMNS = ["math_score", "reading_score", "writing_score", "science_score"]
NUMERIC_FEATURES = [
    "study_hours_per_day",
    "attendance_rate",
    "sleep_hours",
    "social_media_hours",
    "assignment_completion_rate",
    "participation_score",
    "online_courses_completed",
    "previous_gpa",
]
CATEGORICAL_FEATURES = ["internet_access", "study_environment", "tutoring"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "final_exam_score"
DERIVED_RESULT_COLUMNS = ["pass_fail", "grade_category"]
BACKGROUND_COLUMNS = ["gender", "age", "parental_education", "family_income"]
ID_COLUMN = "student_id"
ALL_COLUMNS = [
    ID_COLUMN,
    *BACKGROUND_COLUMNS,
    *CATEGORICAL_FEATURES,
    *NUMERIC_FEATURES,
    *SUBJECT_COLUMNS,
    TARGET,
    *DERIVED_RESULT_COLUMNS,
]
NUMERIC_COLUMNS = ["age", *NUMERIC_FEATURES, *SUBJECT_COLUMNS, TARGET]
TEXT_COLUMNS = [column for column in ALL_COLUMNS if column not in NUMERIC_COLUMNS]
RANGES = {
    "age": (0, 120),
    "study_hours_per_day": (0, 24),
    "attendance_rate": (0, 100),
    "sleep_hours": (0, 24),
    "social_media_hours": (0, 24),
    "assignment_completion_rate": (0, 100),
    "participation_score": (0, 100),
    "online_courses_completed": (0, None),
    "previous_gpa": (0, 4),
    "math_score": (0, 100),
    "reading_score": (0, 100),
    "writing_score": (0, 100),
    "science_score": (0, 100),
    "final_exam_score": (0, 100),
}


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    missing_columns = sorted(set(ALL_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    source_rows = len(raw)
    duplicate_rows = int(raw.duplicated().sum())
    data = raw.drop_duplicates().copy()

    for column in TEXT_COLUMNS:
        cleaned_text = data[column].astype("string").str.strip().replace("", pd.NA)
        data[column] = cleaned_text.astype(object).where(cleaned_text.notna(), np.nan)
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data[ID_COLUMN].isna().any() or data[TARGET].isna().any():
        raise ValueError("Student ID and final exam score must be present in every row.")
    duplicate_ids = int(data[ID_COLUMN].duplicated().sum())
    if duplicate_ids:
        raise ValueError(f"Found {duplicate_ids} duplicate student IDs after exact duplicates were removed.")

    range_issues = {}
    for column, (minimum, maximum) in RANGES.items():
        invalid = data[column].notna() & (data[column] < minimum)
        if maximum is not None:
            invalid |= data[column].notna() & (data[column] > maximum)
        count = int(invalid.sum())
        if count:
            range_issues[column] = count
    if range_issues:
        raise ValueError(f"Values outside the expected ranges: {range_issues}")

    data = data.reset_index(drop=True)
    quality = {
        "source_rows": source_rows,
        "cleaned_rows": len(data),
        "source_columns": len(raw.columns),
        "exact_duplicates_removed": duplicate_rows,
        "duplicate_student_ids": duplicate_ids,
        "missing_values_by_column": {k: int(v) for k, v in data.isna().sum().items()},
        "score_min": float(data[TARGET].min()),
        "score_max": float(data[TARGET].max()),
        "pass_fail_counts": {str(k): int(v) for k, v in data["pass_fail"].value_counts().items()},
        "grade_category_counts": {
            str(k): int(v) for k, v in data["grade_category"].value_counts().items()
        },
        "pass_and_f_count": int(((data["pass_fail"] == "Pass") & (data["grade_category"] == "F")).sum()),
    }
    return data, quality


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def write_report(quality: dict, train_rows: int, test_rows: int, feature_count: int) -> None:
    missing_total = sum(quality["missing_values_by_column"].values())
    missing_lines = "\n".join(
        f"- `{column}`: {count}"
        for column, count in quality["missing_values_by_column"].items()
        if count
    ) or "- None"
    report = f"""# Module 1 data quality report

## Source and cleaning

- Source: user-provided `student_exam_performance_dataset.csv`; original collection method and whether records are synthetic have not been verified.
- Rows read: **{quality['source_rows']:,}**; rows retained: **{quality['cleaned_rows']:,}**.
- Columns: **{quality['source_columns']}**.
- Exact duplicate rows removed: **{quality['exact_duplicates_removed']}**.
- Duplicate student IDs after cleaning: **{quality['duplicate_student_ids']}**.
- Missing cells after cleaning: **{missing_total}**.
- Final exam score range: **{quality['score_min']:.1f}–{quality['score_max']:.1f}** out of 100.

Missing values by column:
{missing_lines}

## Model-ready preparation

- Prediction target: `final_exam_score`.
- Training split: **{train_rows:,}** rows; test split: **{test_rows:,}** rows (80/20, seed 42, score-decile stratification).
- Model input fields: {', '.join(f'`{column}`' for column in FEATURES)}.
- Numeric inputs use median imputation and standard scaling. Categorical inputs use most-frequent imputation and one-hot encoding.
- The preprocessing steps were fitted **only on the training split** and then applied to the test split.
- Encoded feature columns: **{feature_count}**.

`student_id` is retained in the cleaned data and split manifest but never supplied to the model. `pass_fail` and `grade_category` are derived outcome labels and are excluded. The four subject scores remain in cleaned data for descriptive analysis but are excluded from prediction because the CSV does not establish when they were recorded or whether they contribute to the final score. Gender, age, parental education, and family income remain excluded.

## Items to resolve in later modules

- Define and document the Low/Average/High thresholds before training the classification model. The supplied A/B/C/D/F labels are very uneven and should not be used directly as three performance levels.
- Verify the source definition of the subject scores before considering them as model inputs in a future version; their relationship with the target is exceptionally strong.
- The file contains no dates or term identifier, so it cannot support a true change-over-time chart.
- **{quality['pass_and_f_count']}** source rows are marked both `Pass` and grade `F`. The source labels were preserved and are excluded from model training.
"""
    (REPORTS / "data_quality.md").write_text(report, encoding="utf-8")


def main() -> None:
    for directory in (PROCESSED, ARTIFACTS, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)

    data, quality = load_and_clean(RAW)
    data.to_csv(PROCESSED / "cleaned_student_data.csv", index=False)

    # Stratification keeps the score distribution similar in both splits.
    score_deciles = pd.qcut(data[TARGET], q=10, duplicates="drop")
    indices = pd.RangeIndex(len(data))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=score_deciles
    )
    train = data.iloc[train_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)

    preprocessor = build_preprocessor()
    train_array = preprocessor.fit_transform(train[FEATURES])
    test_array = preprocessor.transform(test[FEATURES])
    feature_names = preprocessor.get_feature_names_out().tolist()
    pd.DataFrame(train_array, columns=feature_names).to_csv(PROCESSED / "train_features.csv", index=False)
    pd.DataFrame(test_array, columns=feature_names).to_csv(PROCESSED / "test_features.csv", index=False)
    train[[TARGET]].to_csv(PROCESSED / "train_target.csv", index=False)
    test[[TARGET]].to_csv(PROCESSED / "test_target.csv", index=False)

    manifest = pd.concat(
        [
            train[[ID_COLUMN]].assign(split="train", split_row=range(len(train))),
            test[[ID_COLUMN]].assign(split="test", split_row=range(len(test))),
        ],
        ignore_index=True,
    )
    manifest.to_csv(PROCESSED / "split_manifest.csv", index=False)
    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")

    metadata = {
        "quality": quality,
        "target": TARGET,
        "numeric_model_features": NUMERIC_FEATURES,
        "categorical_model_features": CATEGORICAL_FEATURES,
        "excluded_derived_labels": DERIVED_RESULT_COLUMNS,
        "subject_scores_excluded_from_prediction": SUBJECT_COLUMNS,
        "background_columns_excluded_from_initial_model": BACKGROUND_COLUMNS,
        "split": {"train_rows": len(train), "test_rows": len(test), "random_state": 42},
        "encoded_feature_names": feature_names,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    (ARTIFACTS / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_report(quality, len(train), len(test), len(feature_names))
    print(
        f"Module 1 complete: {len(data):,} clean rows, "
        f"{len(train):,} train / {len(test):,} test, {len(feature_names)} encoded features."
    )


if __name__ == "__main__":
    main()
