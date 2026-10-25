# Accuracy improvement search — 11 inputs

The project keeps the four subject scores out of prediction. The Low/Average/High rule remains Low < 50, Average 50–65, and High > 65. All candidates used the same 8,000 training students and five-fold validation. The 2,000-student test set is the same one used in earlier project versions, so these results are not fresh external validation.

## Training-only comparison

The figures below are accuracy from predictions made for validation folds, before looking at each candidate's test performance. Regression candidates turn their predicted final score into a level using the fixed cutoffs. Direct classifiers predict the level separately and were not used in the dashboard because that could conflict with the displayed numeric score.

| Candidate | Five-fold validation level accuracy |
| --- | ---: |
| Polynomial Ridge (degree 2, alpha 20) | **80.66%** |
| Polynomial Ridge (degree 2, alpha 100) | 80.64% |
| Ridge linear | 80.49% |
| Linear regression | 80.41% |
| Gradient boosting regressor | 80.30% |
| Histogram gradient boosting regressor | 80.21% |
| Extra Trees regressor | 79.90% |
| Linear discriminant analysis classifier | 80.46% |
| Logistic regression classifier | 80.44% |
| Linear SVC classifier | 80.40% |
| Histogram gradient boosting classifier | 80.04% |
| RBF SVC classifier (C=1) | 79.79% |

The selected Polynomial Ridge model was chosen by validation level accuracy while keeping the score and level linked. The reproducible training script compares it with the main baseline models and saves the selected pipeline.

## Same held-out test set

| Metric | Earlier linear model | Selected Polynomial Ridge |
| --- | ---: | ---: |
| Low/Average/High accuracy | 80.4% | **80.6%** |
| Classification errors (2,000 students) | 392 | **388** |
| Balanced accuracy | 75.93% | **76.43%** |
| Macro F1 | 77.22% | **77.63%** |
| Final-score MAE (points) | **4.141** | 4.145 |

The accuracy change is four students, or 0.2 percentage points. It is too small to claim a reliable improvement. No tested candidate reached 82–83% with the 11 permitted inputs. Reaching that target credibly will likely require additional **verified pre-exam** information or a better documented dataset; changing the level cutoffs or using outcome-derived fields would change the task rather than improve this model.
