"""Train the optional score model using four earlier mock marks.

The user confirmed that the four subject columns represent marks available before
the final exam. The original dataset documentation has not independently verified
their timing, so the resulting metrics should be presented with that assumption.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATA = PROJECT / "module1" / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
MARK_COLUMNS = ["math_score", "reading_score", "writing_score", "science_score"]
SEED = 42


def levels(scores: np.ndarray) -> np.ndarray:
    return np.select([scores < 50, scores <= 65], ["Low", "Average"], default="High")


def make_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", Pipeline([
            ("interactions", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=20)),
        ])),
    ])


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    clipped = np.clip(predicted, 0, 100)
    # Classification follows the same displayed two-decimal score as predict.py.
    predicted_levels = levels(np.round(clipped, 2))
    true_levels = levels(actual)
    return {
        "mae": float(mean_absolute_error(actual, clipped)),
        "rmse": float(root_mean_squared_error(actual, clipped)),
        "r2": float(r2_score(actual, clipped)),
        "level_accuracy": float(accuracy_score(true_levels, predicted_levels)),
        "level_balanced_accuracy": float(balanced_accuracy_score(true_levels, predicted_levels)),
        "level_macro_f1": float(f1_score(true_levels, predicted_levels, average="macro")),
        "level_errors": int(np.count_nonzero(true_levels != predicted_levels)),
    }


def main() -> None:
    base_meta = json.loads((PROJECT / "module1" / "artifacts" / "preprocessing_metadata.json").read_text())
    numeric = base_meta["numeric_model_features"] + MARK_COLUMNS
    categorical = base_meta["categorical_model_features"]
    features = numeric + categorical
    cleaned = pd.read_csv(DATA / "cleaned_student_data.csv").set_index("student_id", verify_integrity=True)
    manifest = pd.read_csv(DATA / "split_manifest.csv")
    if len(cleaned) != 10_000 or len(manifest) != 10_000 or manifest.student_id.duplicated().any():
        raise ValueError("Expected the original 10,000-record split manifest and cleaned dataset.")
    if set(manifest.student_id) != set(cleaned.index):
        raise ValueError("Split manifest and cleaned dataset have different student IDs.")
    train_ids = manifest.loc[manifest.split.eq("train")].sort_values("split_row").student_id
    test_ids = manifest.loc[manifest.split.eq("test")].sort_values("split_row").student_id
    if len(train_ids) != 8_000 or len(test_ids) != 2_000:
        raise ValueError("Expected the original 8,000/2,000 training/test split.")
    train, test = cleaned.loc[train_ids], cleaned.loc[test_ids]
    if train[MARK_COLUMNS].isna().any().any() or test[MARK_COLUMNS].isna().any().any():
        raise ValueError("All four earlier mock marks must be present in the dataset.")
    if not all(part[MARK_COLUMNS].apply(lambda c: c.between(0, 100)).all().all() for part in (train, test)):
        raise ValueError("Earlier mock marks must be between 0 and 100.")

    x_train, y_train = train[features], train.final_exam_score.to_numpy()
    x_test, y_test = test[features], test.final_exam_score.to_numpy()
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_predictions = cross_val_predict(make_model(numeric, categorical), x_train, y_train, cv=cv, n_jobs=1)
    cv_metrics = evaluate(y_train, cv_predictions)

    model = make_model(numeric, categorical)
    model.fit(x_train, y_train)
    test_predictions = model.predict(x_test)
    test_metrics = evaluate(y_test, test_predictions)

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "score_model_with_mock_marks.joblib")
    metadata = {
        "selected_score_model": "Polynomial Ridge + earlier mock marks",
        "raw_model_features": features,
        "mock_mark_columns": MARK_COLUMNS,
        "train_rows": 8_000,
        "test_rows": 2_000,
        "random_state": SEED,
        "cv_folds": 5,
        "mock_marks_assumption": "User confirmed these subject columns are earlier mock marks known before the final exam; original dataset provenance and timing are not independently verified.",
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
    }
    (ARTIFACTS / "mock_marks_model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    REPORTS.mkdir(exist_ok=True)
    report = f"""# Optional earlier-mock-marks model

This is a **second** Polynomial Ridge model (degree 2, alpha 20) trained on the same 8,000 students and evaluated on the same 2,000 held-out students as the 11-input model. It adds four subject marks. The baseline model and its saved artifacts are unchanged.

The user confirmed that `math_score`, `reading_score`, `writing_score`, and `science_score` represent earlier mock marks known before the final exam. The dataset's original provenance and timing remain unverified. The mean of those marks is very strongly correlated with the final score, so these figures should **not** be interpreted as independent proof of real-world forecasting performance. This test set has also been used in previous project iterations and is not fresh external validation.

| Metric | Five-fold training CV | Held-out test |
| --- | ---: | ---: |
| Low/Average/High accuracy | {cv_metrics['level_accuracy']:.2%} | {test_metrics['level_accuracy']:.2%} |
| Balanced accuracy | {cv_metrics['level_balanced_accuracy']:.2%} | {test_metrics['level_balanced_accuracy']:.2%} |
| Macro F1 | {cv_metrics['level_macro_f1']:.3f} | {test_metrics['level_macro_f1']:.3f} |
| Final-score MAE | {cv_metrics['mae']:.3f} | {test_metrics['mae']:.3f} |
| Final-score RMSE | {cv_metrics['rmse']:.3f} | {test_metrics['rmse']:.3f} |
| Final-score R² | {cv_metrics['r2']:.3f} | {test_metrics['r2']:.3f} |

The dashboard asks whether to include earlier mock marks. “No” uses the original 11-input model; “Yes” requires all four marks and uses this 15-input model. In both cases, the displayed level comes from the displayed numeric score. The four marks can also guide subject-focus study advice.
"""
    (REPORTS / "mock_marks_model_report.md").write_text(report)
    print(json.dumps({"cv_metrics": cv_metrics, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
