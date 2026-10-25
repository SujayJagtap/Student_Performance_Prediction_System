"""Build descriptive rule thresholds from Module 1's training cohort only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PROCESSED = PROJECT / "module1" / "data" / "processed"
OUTPUT = ROOT / "artifacts" / "rule_thresholds.json"


def main() -> None:
    cleaned = pd.read_csv(PROCESSED / "cleaned_student_data.csv")
    manifest = pd.read_csv(PROCESSED / "split_manifest.csv")
    train_ids = set(manifest.loc[manifest.split.eq("train"), "student_id"])
    train = cleaned.loc[cleaned.student_id.isin(train_ids)]
    if len(train) != 8_000:
        raise ValueError("Expected 8,000 training records for Module 5 rule thresholds.")

    lower_quartile_fields = [
        "study_hours_per_day", "attendance_rate", "assignment_completion_rate",
        "participation_score", "previous_gpa",
    ]
    thresholds = {
        "reference": "25th/75th percentiles of the Module 1 training cohort; descriptive flags, not recommended targets",
        "train_rows": len(train),
        "lower_quartile": {field: round(float(train[field].quantile(.25)), 2) for field in lower_quartile_fields},
        "upper_quartile": {"social_media_hours": round(float(train.social_media_hours.quantile(.75)), 2)},
        "subject_gap_points": 8.0,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    print(f"Saved training-cohort rule thresholds to {OUTPUT}")


if __name__ == "__main__":
    main()
