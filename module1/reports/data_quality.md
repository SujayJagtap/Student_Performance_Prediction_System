# Module 1 data quality report

## Source and cleaning

- Source: user-provided `student_exam_performance_dataset.csv`; original collection method and whether records are synthetic have not been verified.
- Rows read: **10,000**; rows retained: **10,000**.
- Columns: **23**.
- Exact duplicate rows removed: **0**.
- Duplicate student IDs after cleaning: **0**.
- Missing cells after cleaning: **0**.
- Final exam score range: **4.4–97.8** out of 100.

Missing values by column:
- None

## Model-ready preparation

- Prediction target: `final_exam_score`.
- Training split: **8,000** rows; test split: **2,000** rows (80/20, seed 42, score-decile stratification).
- Model input fields: `study_hours_per_day`, `attendance_rate`, `sleep_hours`, `social_media_hours`, `assignment_completion_rate`, `participation_score`, `online_courses_completed`, `previous_gpa`, `internet_access`, `study_environment`, `tutoring`.
- Numeric inputs use median imputation and standard scaling. Categorical inputs use most-frequent imputation and one-hot encoding.
- The preprocessing steps were fitted **only on the training split** and then applied to the test split.
- Encoded feature columns: **15**.

`student_id` is retained in the cleaned data and split manifest but never supplied to the model. `pass_fail` and `grade_category` are derived outcome labels and are excluded. The four subject scores remain in cleaned data for descriptive analysis but are excluded from prediction because the CSV does not establish when they were recorded or whether they contribute to the final score. Gender, age, parental education, and family income remain excluded.

## Items to resolve in later modules

- Define and document the Low/Average/High thresholds before training the classification model. The supplied A/B/C/D/F labels are very uneven and should not be used directly as three performance levels.
- Verify the source definition of the subject scores before considering them as model inputs in a future version; their relationship with the target is exceptionally strong.
- The file contains no dates or term identifier, so it cannot support a true change-over-time chart.
- **30** source rows are marked both `Pass` and grade `F`. The source labels were preserved and are excluded from model training.
