# Optional earlier-mock-marks model

This is a **second** Polynomial Ridge model (degree 2, alpha 20) trained on the same 8,000 students and evaluated on the same 2,000 held-out students as the 11-input model. It adds four subject marks. The baseline model and its saved artifacts are unchanged.

The user confirmed that `math_score`, `reading_score`, `writing_score`, and `science_score` represent earlier mock marks known before the final exam. The dataset's original provenance and timing remain unverified. The mean of those marks is very strongly correlated with the final score, so these figures should **not** be interpreted as independent proof of real-world forecasting performance. This test set has also been used in previous project iterations and is not fresh external validation.

| Metric | Five-fold training CV | Held-out test |
| --- | ---: | ---: |
| Low/Average/High accuracy | 92.64% | 93.95% |
| Balanced accuracy | 91.49% | 93.13% |
| Macro F1 | 0.916 | 0.934 |
| Final-score MAE | 1.552 | 1.522 |
| Final-score RMSE | 1.945 | 1.907 |
| Final-score R² | 0.974 | 0.976 |

The dashboard asks whether to include earlier mock marks. “No” uses the original 11-input model; “Yes” requires all four marks and uses this 15-input model. In both cases, the displayed level comes from the displayed numeric score. The four marks can also guide subject-focus study advice.
