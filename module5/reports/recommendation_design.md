# Module 5 recommendation design

## Purpose

Give a student or teacher a short list of areas to review after a Module 3 prediction. Suggestions use the standard student inputs, the predicted Low/Average/High level, and optional user-supplied mock subject marks. The final exam score, `pass_fail`, and `grade_category` are never recommendation inputs. The dashboard's optional second prediction model can use earlier mock marks when the user chooses it; this does not change the recommendation rules.

## Transparent rules

The training cohort's 25th percentile flags relatively low study hours, attendance, assignment completion, participation, or previous GPA. The 75th percentile flags relatively high social-media time. A noisy study environment triggers a study-space suggestion. Low predicted performance or low participation triggers practice-session advice. The precise saved cutoffs are in `../artifacts/rule_thresholds.json`.

The weak-subject suggestion needs all four optional mock marks. It appears only when the lowest mark is at least **8 points** below the mean of the other three. This rule compares a student's own marks; it does not infer a learning disability or diagnose a cause. The marks must be genuine results available before the final exam if used for a pre-exam plan.

Rules are prioritized and capped at four cards so the dashboard stays readable. When no flag is triggered, the system suggests a simple weekly review and asking a teacher where to focus. Every card states what triggered it and proposes one concrete action.

## Interpretation limits

- These are **rule-based suggestions**, not model explanations, causal effects, or promises of higher scores.
- A cohort quartile is a descriptive reference, not a recommended minimum or maximum for an individual student.
- The dataset has no dates or verified source documentation; its generalizability and synthetic status are unknown.
- In the dashboard's default prediction mode, marks entered later for subject advice do not change the predicted score or level. Choosing the optional earlier-marks prediction mode uses the marks for both prediction and subject advice.
- The dashboard should be used for discussion with a teacher, not as an automated academic decision.
