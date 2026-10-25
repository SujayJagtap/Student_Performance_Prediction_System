# Module 4 — Performance dashboard

This Streamlit dashboard presents the Module 1 training cohort, the Module 3 held-out model evaluation, student comparisons, subject scores, and a live prediction form. It uses a soft blue, white-card, teal-accent layout inspired by the provided design reference.

## Run

Keep `module4` beside `module1`, `module3`, and `module5` in the same project folder. From `module4`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the Local URL printed by Streamlit, usually `http://localhost:8501`.

## Pages

- **Overview:** Filtered training-cohort metrics, score distribution, performance levels, study-hour pattern, subject means, and example test predictions.
- **Student Explorer:** Search or compare up to four student IDs using actual scores, study/attendance fields, and four subject marks.
- **Predict Performance:** Enter the 11 standard input fields, then choose whether to include four earlier mock marks. “No” uses the original model; “Yes” requires all four marks and uses a second trained model. Module 5 shows a rule-based study plan in either mode. If “No” was selected, marks may still be added for subject advice only.
- **Model Insights:** Default-model held-out metrics, actual-versus-predicted chart, confusion matrix, per-level recall, downloadable test predictions, and a summary of the optional earlier-marks model.

The sidebar performance and attendance filters apply to **Overview** training-cohort charts. Test metrics remain fixed so filtering cannot silently change the reported evaluation.

## Interpretation

The dataset has no date or term field. The study-hour chart is a pattern across groups, not a chronological trend. The default model excludes subject marks. The optional second model uses them under the user's confirmation that they are earlier mock marks; the CSV itself does not independently verify their timing. Module 5 recommendations are transparent rules for discussion, not causal claims or guaranteed score gains. The dataset publisher, collection method, and synthetic status should also be verified.
