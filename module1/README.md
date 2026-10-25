# Module 1 — Data collection and preprocessing

This module uses the selected student exam CSV to prepare consistent inputs for the prediction model. It keeps a raw copy, checks data quality, removes exact duplicate rows if present, validates IDs and score ranges, encodes categorical fields, scales numeric fields, and writes train/test data. The original CSV has no missing values; the saved preprocessing pipeline also handles missing predictor values if later data contains them.

## Run

From this folder:

```bash
python3 -m pip install -r requirements.txt
python3 preprocess.py
```

The script reads `data/raw/student_exam_performance_dataset.csv` and creates:

- `data/processed/cleaned_student_data.csv` — full cleaned records for analysis and dashboard work.
- `data/processed/train_features.csv` and `test_features.csv` — encoded, scaled model inputs.
- `data/processed/train_target.csv` and `test_target.csv` — corresponding final exam scores in the same row order.
- `data/processed/split_manifest.csv` — student IDs and split row positions for tracing predictions.
- `artifacts/preprocessor.joblib` — preprocessing fitted on training data only.
- `artifacts/preprocessing_metadata.json` and `reports/data_quality.md` — choices and quality findings.

## Modeling boundary

The default model uses 11 inputs: study hours, attendance, sleep, social-media hours, assignment completion, participation, online-course count, previous GPA, internet access, study environment, and tutoring. It predicts `final_exam_score`.

`pass_fail` and `grade_category` are result labels, so they must not be fed into a prediction model. The subject scores are kept for analysis and student comparisons and remain excluded from the default model because the CSV does not verify when they were recorded or whether they help determine the final score. Module 3 also has an optional separately trained model that uses these four columns under the user's confirmation that they are earlier mock marks. Student ID and background fields remain excluded. Low/Average/High cutoffs are documented in Module 3.

The source data's original collection method has not been verified. Cite its original publisher and identify whether it is synthetic before the final report or presentation.
