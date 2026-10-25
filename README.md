# Student Performance Prediction System

This capstone project contains five modules:

1. `module1` — clean, validate, split, and preprocess the supplied student CSV.
2. `module2` — exploratory charts and data summaries.
3. `module3` — predict final exam scores and Low/Average/High levels from 11 inputs.
4. `module4` — local Streamlit dashboard.
5. `module5` — transparent study suggestions integrated into the dashboard.

## Run the dashboard

From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r module4/requirements.txt
python -m streamlit run module4/app.py
```

Open the Local URL shown in Terminal, usually `http://localhost:8501`. The saved data, model, and rule thresholds are included; no retraining is needed just to view the dashboard.

The Predict Performance page asks whether to include earlier mock marks. **No** uses the original 11-input model; optional marks can still guide subject-focus advice without changing its prediction. **Yes** requires all four marks and uses a separately trained 15-input Polynomial Ridge model; those marks can also guide subject-focus advice. The user confirmed that the CSV's four subject columns represent marks known before the final exam, but the dataset's original provenance and timing remain independently unverified.

The original model's held-out level accuracy is **80.6%** on 2,000 test records; macro F1 is **77.63%**. The optional earlier-marks model reached **93.95%** accuracy and **93.39%** macro F1 on the same test set, under the earlier-marks assumption. See `module3/reports/model_report.md` and `module3/reports/mock_marks_model_report.md` for evaluation details. The test set has been used in earlier project iterations, so these are not fresh external validation results. The dataset's original source and whether it is synthetic remain unverified; recommendations are discussion prompts, not promised score gains.
