# Module 3 — Student performance prediction

This module trains a default final exam score model from 11 non-subject inputs and converts its predicted score to **Low**, **Average**, or **High**. It also trains a direct level classifier for a research comparison. A separate optional Polynomial Ridge model adds four earlier mock marks, based on the user's confirmation that these marks are known before the final exam. The dashboard uses `predict.py`, which keeps the shown score and level consistent.

The score model is selected by five-fold training accuracy of the score-derived levels, with macro F1 as the tie-breaker. Score MAE and RMSE are reported alongside classification metrics.

## Run

Place this `module3` folder beside the completed `module1` folder, then run:

```bash
python3 -m pip install -r requirements.txt
python3 train_models.py
```

Module 3 reads Module 1's cleaned data, saved split manifest, feature metadata, and preprocessing pipeline. It writes model files to `artifacts/`, five-fold comparison tables and test predictions to `tables/`, charts to `figures/`, and a detailed result to `reports/model_report.md`.

To reproduce the optional earlier-marks model with the same 8,000/2,000 split, run `python3 train_mock_marks_model.py`. It saves `artifacts/score_model_with_mock_marks.joblib`, separate metadata, `tables/test_predictions_with_mock_marks.csv`, and `reports/mock_marks_model_report.md`. It does not overwrite the default model.

## Level definitions

- **Low:** final score below 50.
- **Average:** final score from 50 through 65, inclusive.
- **High:** final score above 65.

These thresholds are a documented project rule. The source's `pass_fail` and `grade_category` fields are excluded because they reflect the result and contain some contradictory entries.

## Example inference

Pass a CSV containing the 11 raw input columns named in `artifacts/model_metadata.json`:

```bash
python3 predict.py new_students.csv predicted_students.csv
```

The model pipeline performs imputation, scaling, and encoding itself. Do not pass the already encoded `train_features.csv` file to `predict.py`. A dashboard can import `predict_students` from this file and pass it a pandas DataFrame.

For the optional model, add all four `mock_math_score`, `mock_reading_score`, `mock_writing_score`, and `mock_science_score` columns to the same 11 inputs and pass `--use-mock-marks` on the CLI, or call `predict_students(frame, use_mock_marks=True)` in Python.

## Evaluation boundary

Only the 8,000 training records are used for the default model comparison and selection. The 2,000 test records are evaluated afterward. Each cross-validation fold fits its own preprocessing steps. The optional model uses the same split and algorithm with four added marks; its test set is not fresh external validation. The CSV does not independently verify when the subject marks were recorded or whether they are components of the final score. The dataset's publisher, collection method, and synthetic status also need verification for the final capstone report.
