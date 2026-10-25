"""Module 3: compare, select, evaluate, and save student performance models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MODULE1 = PROJECT / "module1"
DATA = MODULE1 / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
TARGET = "final_exam_score"
LEVELS = ["Low", "Average", "High"]
SEED = 42


def level_from_score(values: np.ndarray | pd.Series) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    return np.select([scores < 50, scores <= 65], ["Low", "Average"], default="High")


def read_split() -> tuple[pd.DataFrame, pd.DataFrame, list[str], object]:
    metadata = json.loads((MODULE1 / "artifacts" / "preprocessing_metadata.json").read_text())
    features = metadata["numeric_model_features"] + metadata["categorical_model_features"]
    forbidden = {TARGET, "student_id", "pass_fail", "grade_category"}
    if forbidden.intersection(features):
        raise ValueError("Target, identifier, or derived outcome labels entered model inputs.")
    subject_features = metadata.get("subject_scores_excluded_from_prediction", [])
    if len(subject_features) != 4 or set(subject_features).intersection(features):
        raise ValueError("The four subject scores must be excluded from model inputs.")

    cleaned = pd.read_csv(DATA / "cleaned_student_data.csv").set_index("student_id", verify_integrity=True)
    manifest = pd.read_csv(DATA / "split_manifest.csv")
    if len(manifest) != len(cleaned) or manifest.student_id.duplicated().any():
        raise ValueError("The Module 1 split manifest must contain every student exactly once.")
    if set(manifest.student_id) != set(cleaned.index):
        raise ValueError("The split manifest and cleaned data have different student IDs.")

    splits = {}
    for name in ("train", "test"):
        part = manifest.loc[manifest.split.eq(name)].sort_values("split_row")
        if part.split_row.tolist() != list(range(len(part))):
            raise ValueError(f"{name} split positions are not contiguous.")
        splits[name] = cleaned.loc[part.student_id].reset_index()
    if len(splits["train"]) != 8000 or len(splits["test"]) != 2000:
        raise ValueError("Expected the Module 1 8,000/2,000 split.")

    for name, target_file in (("train", "train_target.csv"), ("test", "test_target.csv")):
        module1_targets = pd.read_csv(DATA / target_file)[TARGET].to_numpy()
        if not np.allclose(splits[name][TARGET].to_numpy(), module1_targets):
            raise ValueError(f"{name} scores do not align with Module 1 targets.")
    preprocessor = joblib.load(MODULE1 / "artifacts" / "preprocessor.joblib")
    return splits["train"], splits["test"], features, preprocessor


def make_pipeline(preprocessor: object, estimator: object) -> Pipeline:
    # Clone removes Module 1's fitted state, so each CV fold fits imputation,
    # scaling, and one-hot encoding on that fold's training records only.
    return Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])


def score_derived_level_accuracy(estimator: object, x: pd.DataFrame, y: pd.Series) -> float:
    return float(accuracy_score(level_from_score(y), level_from_score(estimator.predict(x))))


def score_derived_level_macro_f1(estimator: object, x: pd.DataFrame, y: pd.Series) -> float:
    return float(f1_score(level_from_score(y), level_from_score(estimator.predict(x)), labels=LEVELS, average="macro"))


def compare_regressors(x: pd.DataFrame, y: pd.Series, preprocessor: object) -> tuple[pd.DataFrame, dict]:
    models = {
        "Dummy mean": DummyRegressor(strategy="mean"),
        "Ridge linear": Ridge(alpha=10),
        "Polynomial Ridge": Pipeline([
            ("interactions", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=20)),
        ]),
        "Decision tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=SEED),
        "Random forest": RandomForestRegressor(
            n_estimators=120, min_samples_leaf=3, max_features=0.8, random_state=SEED, n_jobs=1
        ),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        results = cross_validate(
            make_pipeline(preprocessor, model), x, y, cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2",
                "level_accuracy": score_derived_level_accuracy,
                "level_macro_f1": score_derived_level_macro_f1,
            },
            n_jobs=1, error_score="raise",
        )
        rows.append({
            "model": name,
            "cv_rmse_mean": float(-results["test_rmse"].mean()),
            "cv_rmse_sd": float(results["test_rmse"].std()),
            "cv_mae_mean": float(-results["test_mae"].mean()),
            "cv_r2_mean": float(results["test_r2"].mean()),
            "cv_level_accuracy_mean": float(results["test_level_accuracy"].mean()),
            "cv_level_macro_f1_mean": float(results["test_level_macro_f1"].mean()),
        })
    comparison = pd.DataFrame(rows).sort_values(
        ["cv_level_accuracy_mean", "cv_level_macro_f1_mean"], ascending=False
    ).reset_index(drop=True)
    return comparison, models


def compare_classifiers(x: pd.DataFrame, y: np.ndarray, preprocessor: object) -> tuple[pd.DataFrame, dict]:
    models = {
        "Dummy most frequent": DummyClassifier(strategy="most_frequent"),
        "Decision tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=SEED
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=3, max_features=0.8,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=1,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        results = cross_validate(
            make_pipeline(preprocessor, model), x, y, cv=cv,
            scoring={"macro_f1": "f1_macro", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
            n_jobs=1, error_score="raise",
        )
        rows.append({
            "model": name,
            "cv_macro_f1_mean": float(results["test_macro_f1"].mean()),
            "cv_macro_f1_sd": float(results["test_macro_f1"].std()),
            "cv_accuracy_mean": float(results["test_accuracy"].mean()),
            "cv_balanced_accuracy_mean": float(results["test_balanced_accuracy"].mean()),
        })
    comparison = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    return comparison, models


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, labels=LEVELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LEVELS, average="macro", zero_division=0)),
        "per_class": {
            name: {"precision": float(report[name]["precision"]), "recall": float(report[name]["recall"]),
                   "f1": float(report[name]["f1-score"]), "support": int(report[name]["support"])}
            for name in LEVELS
        },
    }


def plot_diagnostics(actual: np.ndarray, predicted: np.ndarray, true_level: np.ndarray,
                     score_level: np.ndarray, direct_level: np.ndarray) -> None:
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#f8fbfd", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(actual, predicted, s=15, alpha=0.35, color="#008f93", edgecolor="none")
    axes[0].plot([0, 100], [0, 100], color="#20394b", linestyle="--", linewidth=1.5)
    axes[0].set(xlabel="Actual final score", ylabel="Predicted final score", title="Held-out test scores", xlim=(0, 100), ylim=(0, 100))
    axes[1].hist(actual - predicted, bins=30, color="#4ab9bb", edgecolor="white")
    axes[1].axvline(0, color="#20394b", linestyle="--", linewidth=1.5)
    axes[1].set(xlabel="Actual minus predicted score", ylabel="Students", title="Test prediction errors")
    fig.savefig(FIGURES / "01_score_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, pred, title in zip(
        axes, (score_level, direct_level),
        ("Levels from predicted score", "Direct classifier (research comparison)"),
    ):
        ConfusionMatrixDisplay.from_predictions(
            true_level, pred, labels=LEVELS, ax=ax, cmap="PuBuGn", colorbar=False, values_format="d"
        )
        ax.set_title(title)
    fig.savefig(FIGURES / "02_level_confusion_matrices.png", dpi=180)
    plt.close(fig)


def md_table(frame: pd.DataFrame, columns: list[str], precision: int = 3) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame[columns].iterrows():
        rows.append("| " + " | ".join(f"{row[c]:.{precision}f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns) + " |")
    return "\n".join([header, divider, *rows])


def write_report(metadata: dict, regression_cv: pd.DataFrame, classification_cv: pd.DataFrame) -> None:
    score = metadata["test_score_metrics"]
    derived = metadata["test_score_derived_level_metrics"]
    direct = metadata["test_direct_classifier_metrics"]
    levels = metadata["class_counts"]
    train_counts = ", ".join(f"{name} {levels['train'][name]:,}" for name in LEVELS)
    test_counts = ", ".join(f"{name} {levels['test'][name]:,}" for name in LEVELS)
    feature_count = len(metadata["raw_model_features"])
    error_count = round((1 - derived["accuracy"]) * metadata["test_rows"])
    report = f"""# Module 3 — Performance prediction report

## Goal and data boundary

Predict `final_exam_score` from **{feature_count}** input fields selected in Module 1. The training set has **8,000** students; the test set has **2,000** students held out from model fitting. This same test set was evaluated in earlier project versions, so it is not fresh external validation. The dataset's original collection method and synthetic status are unverified.

The maths, reading, writing, and science scores are **excluded from prediction**. Their mean correlates **{metadata['subject_mean_target_correlation_train']:.3f}** with the final score in the training set, but the CSV does not establish when they were recorded or whether they contribute to the final score. They remain available for descriptive charts and student comparisons.

The level rule is **Low: score < 50; Average: 50 ≤ score ≤ 65; High: score > 65**. It is a project rule based on numeric final scores, not the source's inconsistent `pass_fail`/`grade_category` labels. Train counts: {train_counts}; test counts: {test_counts}.

## Training-only model comparison

Each candidate used the same five folds and the same {feature_count} raw inputs. Preprocessing was refitted inside each fold. Because this project emphasizes Low/Average/High accuracy, the score model was selected by cross-validation accuracy of the levels derived from its predicted score, with macro F1 as the tie-breaker. The test set was used only afterward. RMSE/MAE are score points; lower is better. R², accuracy, and macro F1 are higher-is-better.

**Final score regressors (ranked by cross-validation level accuracy):**

{md_table(regression_cv, ['model', 'cv_level_accuracy_mean', 'cv_level_macro_f1_mean', 'cv_rmse_mean', 'cv_mae_mean', 'cv_r2_mean'])}

**Direct level classifiers (ranked by cross-validation macro F1):**

{md_table(classification_cv, ['model', 'cv_macro_f1_mean', 'cv_macro_f1_sd', 'cv_accuracy_mean', 'cv_balanced_accuracy_mean'])}

The selected score model is **{metadata['selected_score_model']}**. The selected direct classifier is **{metadata['selected_direct_classifier']}** and is saved as a comparison artifact.

## Held-out test evaluation

- Predicted final score: **MAE {score['mae']:.2f}**, **RMSE {score['rmse']:.2f}**, **R² {score['r2']:.3f}**.
- Levels derived from the predicted score: **accuracy {derived['accuracy']:.3f}**, **balanced accuracy {derived['balanced_accuracy']:.3f}**, **macro F1 {derived['macro_f1']:.3f}**.
- Direct level classifier: **accuracy {direct['accuracy']:.3f}**, **balanced accuracy {direct['balanced_accuracy']:.3f}**, **macro F1 {direct['macro_f1']:.3f}**.

The score-derived level model makes **{error_count:,}** Low/Average/High errors among the {metadata['test_rows']:,} test students.

This model was selected from training-only cross-validation. Its test accuracy is below the requested 82–83% target. The gain over the previous linear model's 80.4% test accuracy is only 0.2 percentage points, too small to claim a reliable improvement.

**Per-level recall for score-derived predictions:** Low {derived['per_class']['Low']['recall']:.3f}, Average {derived['per_class']['Average']['recall']:.3f}, High {derived['per_class']['High']['recall']:.3f}. See `02_level_confusion_matrices.png` and `tables/test_predictions.csv` for individual errors.

The dashboard should use `artifacts/score_model.joblib` through `predict.py` so its score and level always agree. The direct classifier is available to compare classification performance, not to show a second potentially conflicting level.

## Limits and next step

The chosen model uses previous GPA, study behavior, attendance, and learning-environment inputs. Confirm that every predictor is available before the final exam before deploying it. Subject marks, result labels, student ID, and demographic/background fields remain excluded. Levels near the 50 and 65 cutoffs are sensitive to small score errors. The three classes are uneven, so macro F1 and per-class recall matter alongside overall accuracy.

The dataset has no dates. The dashboard can show score distributions and factor relationships, but no genuine time trend. Confirm the dataset publisher, collection method, and whether it is synthetic before the final capstone presentation.
"""
    (REPORTS / "model_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    for folder in (ARTIFACTS, TABLES, FIGURES, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    train, test, features, preprocessor = read_split()
    x_train, y_train = train[features], train[TARGET]
    x_test, y_test = test[features], test[TARGET]
    train_level, test_level = level_from_score(y_train), level_from_score(y_test)
    regression_cv, regressors = compare_regressors(x_train, y_train, preprocessor)
    regression_cv.to_csv(TABLES / "regression_cv_comparison.csv", index=False)
    selected_score_name = str(regression_cv.iloc[0]["model"])
    score_pipeline = make_pipeline(preprocessor, regressors[selected_score_name]).fit(x_train, y_train)
    score_predictions = score_pipeline.predict(x_test)
    score_derived_level = level_from_score(score_predictions)

    classification_cv, classifiers = compare_classifiers(x_train, train_level, preprocessor)
    classification_cv.to_csv(TABLES / "classification_cv_comparison.csv", index=False)
    selected_direct_name = str(classification_cv.iloc[0]["model"])
    direct_pipeline = make_pipeline(preprocessor, classifiers[selected_direct_name]).fit(x_train, train_level)
    direct_predictions = direct_pipeline.predict(x_test)

    score_metrics = {
        "mae": float(mean_absolute_error(y_test, score_predictions)),
        "rmse": float(root_mean_squared_error(y_test, score_predictions)),
        "r2": float(r2_score(y_test, score_predictions)),
    }
    derived_metrics = classification_metrics(test_level, score_derived_level)
    direct_metrics = classification_metrics(test_level, direct_predictions)
    predictions = pd.DataFrame({
        "student_id": test.student_id,
        "actual_score": y_test,
        "predicted_score": score_predictions,
        "actual_level": test_level,
        "score_derived_predicted_level": score_derived_level,
        "direct_classifier_predicted_level": direct_predictions,
    })
    predictions.to_csv(TABLES / "test_predictions.csv", index=False)

    joblib.dump(score_pipeline, ARTIFACTS / "score_model.joblib")
    joblib.dump(direct_pipeline, ARTIFACTS / "level_classifier_research.joblib")
    if not np.allclose(joblib.load(ARTIFACTS / "score_model.joblib").predict(x_test), score_predictions):
        raise AssertionError("Saved score model changed its predictions after reloading.")

    metadata = {
        "target": TARGET,
        "level_rule": {"Low": "score < 50", "Average": "50 <= score <= 65", "High": "score > 65"},
        "raw_model_features": features,
        "train_rows": len(train), "test_rows": len(test), "random_state": SEED, "cv_folds": 5,
        "selected_score_model": selected_score_name,
        "score_model_selection_metric": "five-fold CV score-derived level accuracy; macro F1 tie-breaker",
        "selected_direct_classifier": selected_direct_name,
        "subject_scores_excluded_from_prediction": ["math_score", "reading_score", "writing_score", "science_score"],
        "subject_mean_target_correlation_train": float(train[["math_score", "reading_score", "writing_score", "science_score"]].mean(axis=1).corr(y_train)),
        "class_counts": {
            "train": {label: int((train_level == label).sum()) for label in LEVELS},
            "test": {label: int((test_level == label).sum()) for label in LEVELS},
        },
        "test_score_metrics": score_metrics,
        "test_score_derived_level_metrics": derived_metrics,
        "test_direct_classifier_metrics": direct_metrics,
        "prediction_score_range": [float(score_predictions.min()), float(score_predictions.max())],
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_diagnostics(y_test.to_numpy(), score_predictions, test_level, score_derived_level, direct_predictions)
    write_report(metadata, regression_cv, classification_cv)
    print(json.dumps({
        "selected_score_model": selected_score_name,
        "selected_direct_classifier": selected_direct_name,
        "test_score_metrics": score_metrics,
        "score_derived_macro_f1": derived_metrics["macro_f1"],
        "direct_classifier_macro_f1": direct_metrics["macro_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
