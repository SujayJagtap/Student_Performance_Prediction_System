"""Check recommendation behavior at the model and rule boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE5))
from recommend import generate_recommendations, recommend_students  # noqa: E402


class RecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = pd.read_csv(MODULE5 / "examples" / "sample_students.csv")

    def test_mock_marks_change_advice_but_not_prediction(self) -> None:
        without_marks = recommend_students(self.sample, use_mock_marks=False)
        with_marks = recommend_students(self.sample, use_mock_marks=True)
        pd.testing.assert_frame_equal(
            without_marks[["predicted_score", "predicted_level"]],
            with_marks[["predicted_score", "predicted_level"]],
        )
        self.assertIn("Focus on Science", with_marks.loc[0, "recommendation_1"])
        self.assertNotIn("Focus on", " ".join(str(value) for value in without_marks.filter(like="recommendation").iloc[0]))

    def test_outcome_columns_do_not_change_advice(self) -> None:
        expected = recommend_students(self.sample, use_mock_marks=True)
        altered = self.sample.copy()
        altered["final_exam_score"] = 0
        altered["pass_fail"] = "Fail"
        altered["grade_category"] = "F"
        actual = recommend_students(altered, use_mock_marks=True)
        pd.testing.assert_frame_equal(expected, actual)

    def test_incomplete_mock_marks_are_rejected_and_plan_is_short(self) -> None:
        profile = self.sample.iloc[0]
        with self.assertRaisesRegex(ValueError, "all four"):
            generate_recommendations(profile, subject_marks={"math": 50})
        marks = {"math": 42, "reading": 65, "writing": 58, "science": 40}
        items = generate_recommendations(profile, predicted_level="Low", subject_marks=marks)
        self.assertLessEqual(len(items), 4)
        self.assertEqual(len({item["id"] for item in items}), len(items))


if __name__ == "__main__":
    unittest.main()
