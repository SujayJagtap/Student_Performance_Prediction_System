"""Transparent study suggestions based on pre-exam inputs and optional mock marks.

Rules describe areas to review. They do not estimate the effect of an action on a
student's final score, and subject marks are never passed to the prediction model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
THRESHOLDS_FILE = ROOT / "artifacts" / "rule_thresholds.json"
SUBJECTS = {"math": "Maths", "reading": "Reading", "writing": "Writing", "science": "Science"}
MOCK_COLUMNS = {name: f"mock_{name}_score" for name in SUBJECTS}


def load_thresholds() -> dict:
    if not THRESHOLDS_FILE.is_file():
        raise FileNotFoundError("Run module5/build_rules.py to create rule_thresholds.json.")
    return json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))


def _number(profile: Mapping, field: str, lower: float, upper: float) -> float | None:
    value = profile.get(field)
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{field} must be between {lower} and {upper}.")
    return number


def _item(rule_id: str, title: str, reason: str, action: str, priority: int, severity: float = 0.0) -> dict:
    return {
        "id": rule_id, "title": title, "reason": reason, "action": action,
        "priority": priority, "severity": severity,
    }


def generate_recommendations(
    profile: Mapping,
    predicted_level: str | None = None,
    subject_marks: Mapping[str, float] | None = None,
    limit: int = 4,
    thresholds: dict | None = None,
) -> list[dict]:
    """Return prioritized, rule-based actions without using the final exam result."""
    if predicted_level not in {None, "Low", "Average", "High"}:
        raise ValueError("predicted_level must be Low, Average, High, or None.")
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10.")
    rules = thresholds or load_thresholds()
    low = rules["lower_quartile"]
    high = rules["upper_quartile"]
    items: list[dict] = []

    study = _number(profile, "study_hours_per_day", 0, 24)
    attendance = _number(profile, "attendance_rate", 0, 100)
    assignments = _number(profile, "assignment_completion_rate", 0, 100)
    participation = _number(profile, "participation_score", 0, 100)
    social = _number(profile, "social_media_hours", 0, 24)
    gpa = _number(profile, "previous_gpa", 0, 4)

    if study is not None and study < low["study_hours_per_day"]:
        items.append(_item(
            "study_routine", "Build a steady study routine",
            f"Study time is {study:.1f} h/day, below the training cohort's lower quartile of {low['study_hours_per_day']:.1f} h/day.",
            "Try one extra focused 25-minute block on four days this week, then review what you covered.",
            3, low["study_hours_per_day"] - study,
        ))
    if attendance is not None and attendance < low["attendance_rate"]:
        items.append(_item(
            "attendance", "Catch up on missed classes",
            f"Attendance is {attendance:.1f}%, below the training cohort's lower quartile of {low['attendance_rate']:.1f}%.",
            "Ask for notes on missed topics, list questions for your teacher, and plan to attend upcoming sessions.",
            3, (low["attendance_rate"] - attendance) / 10,
        ))
    if assignments is not None and assignments < low["assignment_completion_rate"]:
        items.append(_item(
            "assignments", "Finish the next outstanding assignment",
            f"Assignment completion is {assignments:.1f}%, below the training cohort's lower quartile of {low['assignment_completion_rate']:.1f}%.",
            "List incomplete work, choose one item to finish first, and ask for help on anything blocking it.",
            2, (low["assignment_completion_rate"] - assignments) / 10,
        ))
    if gpa is not None and gpa < low["previous_gpa"]:
        items.append(_item(
            "foundations", "Review earlier topics",
            f"Previous GPA is {gpa:.2f}, below the training cohort's lower quartile of {low['previous_gpa']:.2f}.",
            "Pick one topic you found difficult in earlier work and review it with a teacher or a short practice set.",
            2, low["previous_gpa"] - gpa,
        ))
    if (participation is not None and participation < low["participation_score"]) or predicted_level == "Low":
        reason = (
            f"Participation is {participation:.1f}, below the training cohort's lower quartile of {low['participation_score']:.1f}."
            if participation is not None and participation < low["participation_score"]
            else "The current model estimates a Low performance level."
        )
        items.append(_item(
            "practice", "Attend a practice session",
            reason,
            "Join a teacher-led practice session or complete an exam-style question set, then discuss mistakes.",
            2, 1.0,
        ))
    if social is not None and social > high["social_media_hours"]:
        items.append(_item(
            "focus_time", "Protect a focused study slot",
            f"Social-media time is {social:.1f} h/day, above the training cohort's upper quartile of {high['social_media_hours']:.1f} h/day.",
            "Try a phone-free 25-minute study block and decide whether it helps you stay on task.",
            1, social - high["social_media_hours"],
        ))
    if str(profile.get("study_environment", "")).strip().lower() == "noisy":
        items.append(_item(
            "study_space", "Find a quieter study slot",
            "The study environment is recorded as noisy.",
            "Try a library, a quieter room, or a time with fewer interruptions for your next revision block.",
            1,
        ))

    if subject_marks is not None:
        if set(subject_marks) != set(SUBJECTS):
            raise ValueError("Provide all four mock marks: math, reading, writing, and science.")
        marks = {name: _number(subject_marks, name, 0, 100) for name in SUBJECTS}
        if any(value is None for value in marks.values()):
            raise ValueError("All four mock marks must be filled in.")
        weakest = min(marks, key=marks.get)
        other_mean = sum(value for name, value in marks.items() if name != weakest) / 3
        gap = other_mean - marks[weakest]
        if gap >= float(rules["subject_gap_points"]):
            items.append(_item(
                "subject_focus", f"Focus on {SUBJECTS[weakest]}",
                f"Your supplied mock mark is {marks[weakest]:.1f}/100, {gap:.1f} points below the average of your other three marks.",
                f"Choose two {SUBJECTS[weakest]} topics to revise, try exam-style questions, and review errors with your teacher.",
                4, gap / 10,
            ))

    if not items:
        items.append(_item(
            "weekly_review", "Keep a weekly review habit",
            "None of the selected profile fields falls below the training cohort's lower-quartile flags.",
            "Review one topic and one practice set each week, and ask your teacher where to focus next.",
            1,
        ))
    items.sort(key=lambda item: (-item["priority"], -item["severity"], item["id"]))
    return [{key: value for key, value in item.items() if key != "severity"} for item in items[:limit]]


def recommend_students(frame: pd.DataFrame, use_mock_marks: bool = False) -> pd.DataFrame:
    """Predict with Module 3, then add suggestions; mock marks never enter Module 3."""
    sys.path.insert(0, str(PROJECT / "module3"))
    from predict import predict_students

    if use_mock_marks:
        missing = sorted(set(MOCK_COLUMNS.values()) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing optional mock-mark columns: {missing}")
    predictions = predict_students(frame)
    rows = []
    for index, (_, row) in enumerate(frame.iterrows()):
        marks = {name: row[column] for name, column in MOCK_COLUMNS.items()} if use_mock_marks else None
        result = predictions.iloc[index]
        suggestions = generate_recommendations(row, predicted_level=result.predicted_level, subject_marks=marks)
        record = {"predicted_score": result.predicted_score, "predicted_level": result.predicted_level}
        if "student_id" in frame.columns:
            record["student_id"] = row.student_id
        for position, suggestion in enumerate(suggestions, 1):
            record[f"recommendation_{position}"] = f"{suggestion['title']}: {suggestion['action']}"
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Module 5 study suggestions from Module 3 predictions.")
    parser.add_argument("input_csv", type=Path, help="CSV with the 11 model inputs; student_id is optional")
    parser.add_argument("output_csv", type=Path, help="Where to save scores, levels, and recommendations")
    parser.add_argument("--use-mock-marks", action="store_true", help="Use all four mock_*_score fields for subject advice only; they never enter the model")
    args = parser.parse_args()
    output = recommend_students(pd.read_csv(args.input_csv), use_mock_marks=args.use_mock_marks)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Saved recommendations for {len(output):,} students to {args.output_csv}")


if __name__ == "__main__":
    main()
