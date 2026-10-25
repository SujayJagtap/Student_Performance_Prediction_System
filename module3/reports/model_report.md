# Module 3 — Performance prediction report

## Goal and data boundary

Predict `final_exam_score` from **11** input fields selected in Module 1. The training set has **8,000** students; the test set has **2,000** students held out from model fitting. This same test set was evaluated in earlier project versions, so it is not fresh external validation. The dataset's original collection method and synthetic status are unverified.

The maths, reading, writing, and science scores are **excluded from prediction**. Their mean correlates **0.986** with the final score in the training set, but the CSV does not establish when they were recorded or whether they contribute to the final score. They remain available for descriptive charts and student comparisons.

The level rule is **Low: score < 50; Average: 50 ≤ score ≤ 65; High: score > 65**. It is a project rule based on numeric final scores, not the source's inconsistent `pass_fail`/`grade_category` labels. Train counts: Low 4,118, Average 3,083, High 799; test counts: Low 1,024, Average 776, High 200.

## Training-only model comparison

Each candidate used the same five folds and the same 11 raw inputs. Preprocessing was refitted inside each fold. Because this project emphasizes Low/Average/High accuracy, the score model was selected by cross-validation accuracy of the levels derived from its predicted score, with macro F1 as the tie-breaker. The test set was used only afterward. RMSE/MAE are score points; lower is better. R², accuracy, and macro F1 are higher-is-better.

**Final score regressors (ranked by cross-validation level accuracy):**

| model | cv_level_accuracy_mean | cv_level_macro_f1_mean | cv_rmse_mean | cv_mae_mean | cv_r2_mean |
| --- | --- | --- | --- | --- | --- |
| Polynomial Ridge | 0.807 | 0.770 | 5.079 | 4.040 | 0.825 |
| Ridge linear | 0.805 | 0.769 | 5.061 | 4.019 | 0.826 |
| Random forest | 0.799 | 0.763 | 5.203 | 4.146 | 0.816 |
| Decision tree | 0.787 | 0.748 | 5.556 | 4.415 | 0.790 |
| Dummy mean | 0.515 | 0.227 | 12.135 | 9.620 | -0.001 |

**Direct level classifiers (ranked by cross-validation macro F1):**

| model | cv_macro_f1_mean | cv_macro_f1_sd | cv_accuracy_mean | cv_balanced_accuracy_mean |
| --- | --- | --- | --- | --- |
| Random forest | 0.758 | 0.012 | 0.791 | 0.752 |
| Decision tree | 0.714 | 0.011 | 0.750 | 0.751 |
| Dummy most frequent | 0.227 | 0.000 | 0.515 | 0.333 |

The selected score model is **Polynomial Ridge**. The selected direct classifier is **Random forest** and is saved as a comparison artifact.

## Held-out test evaluation

- Predicted final score: **MAE 4.15**, **RMSE 5.21**, **R² 0.818**.
- Levels derived from the predicted score: **accuracy 0.806**, **balanced accuracy 0.764**, **macro F1 0.776**.
- Direct level classifier: **accuracy 0.798**, **balanced accuracy 0.768**, **macro F1 0.771**.

The score-derived level model makes **388** Low/Average/High errors among the 2,000 test students.

This model was selected from training-only cross-validation. Its test accuracy is below the requested 82–83% target. The gain over the previous linear model's 80.4% test accuracy is only 0.2 percentage points, too small to claim a reliable improvement.

**Per-level recall for score-derived predictions:** Low 0.870, Average 0.758, High 0.665. See `02_level_confusion_matrices.png` and `tables/test_predictions.csv` for individual errors.

The dashboard should use `artifacts/score_model.joblib` through `predict.py` so its score and level always agree. The direct classifier is available to compare classification performance, not to show a second potentially conflicting level.

## Limits and next step

The chosen model uses previous GPA, study behavior, attendance, and learning-environment inputs. Confirm that every predictor is available before the final exam before deploying it. Subject marks, result labels, student ID, and demographic/background fields remain excluded. Levels near the 50 and 65 cutoffs are sensitive to small score errors. The three classes are uneven, so macro F1 and per-class recall matter alongside overall accuracy.

The dataset has no dates. The dashboard can show score distributions and factor relationships, but no genuine time trend. Confirm the dataset publisher, collection method, and whether it is synthetic before the final capstone presentation.
