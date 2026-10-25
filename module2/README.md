# Module 2 — Exploratory data analysis

This module analyzes the **training split** produced in Module 1. It creates five charts, summary tables, and a written report. The 2,000 test records remain untouched for model evaluation in Module 3.

## Run

From this folder:

```bash
python3 -m pip install -r requirements.txt
python3 eda.py
```

The script reads `../module1/data/processed/cleaned_student_data.csv` and `../module1/data/processed/split_manifest.csv`. It writes figures to `figures/`, CSV and JSON summaries to `tables/`, and the narrative to `reports/eda_report.md`.

## Charts

1. Final exam score distribution.
2. Study time, attendance, participation, and previous GPA versus final score.
3. Average final score by attendance and study-time bands.
4. Subject score distributions and each student's lowest-scoring subject.
5. Correlations between selected numeric factors and final score.

All findings are descriptive. Correlation and group differences do not establish that changing one factor will change a student's score. The dataset has no date field, so this module does not create a chronological trend. Subject marks are shown for analysis and remain excluded from Module 3's default prediction model. Its optional second model uses them under the user's confirmation that they represent earlier mock marks; the original data source has not independently verified their timing.
