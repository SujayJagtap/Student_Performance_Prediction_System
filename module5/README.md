# Module 5 — Recommendation system

Module 5 adds practical, transparent study suggestions to the Module 4 dashboard. The rules use the current Module 3 predicted level and training-cohort reference thresholds. Four user-supplied earlier mock marks can identify a subject to review. The dashboard now lets a user choose whether those same marks also enter a separately trained prediction model; the recommendation rules themselves remain unchanged.

## Run the dashboard

Keep `module5` beside `module1`, `module3`, and `module4` in the same project folder. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r module4/requirements.txt
python -m streamlit run module4/app.py
```

After selecting **Generate prediction**, the **Study plan** section shows recommendations. If the user chose **No** for earlier-marks prediction, optional marks entered in the Study plan affect only the subject-focus suggestion. If the user chose **Yes**, the four marks entered in the prediction form affect the prediction and may also add subject-focus advice.

## Command-line example

From the project root:

```bash
python module5/recommend.py module5/examples/sample_students.csv module5/examples/sample_recommendations.csv --use-mock-marks
```

This Module 5 CLI continues to use the default 11-input prediction model; `--use-mock-marks` here applies marks to study advice only. For optional earlier-marks **prediction** on the CLI, use `python module3/predict.py input.csv output.csv --use-mock-marks`. The CSV still needs the 11 standard input columns; `student_id` is optional. To regenerate the training-cohort reference thresholds after changing Module 1 data, run `python module5/build_rules.py` before generating recommendations.

## How the rules work

The saved thresholds in `artifacts/rule_thresholds.json` are the lower or upper quartiles of the **training** cohort, rounded to two decimals. A value beyond one of these cutoffs flags an area worth discussing; it is **not** a clinical, school, or causal standard. The weak-subject rule requires all four optional mock marks and a gap of at least 8 points between the lowest mark and the mean of the other three.

The recommendations cover study routine, attendance, assignments, foundational topics, practice sessions, focus time, study space, and weak subjects. At most four are displayed, ordered by transparent rule priority. They do not estimate how many marks a student would gain. A teacher or student should decide which advice is relevant and feasible. See `reports/recommendation_design.md` for the full logic and limits.
