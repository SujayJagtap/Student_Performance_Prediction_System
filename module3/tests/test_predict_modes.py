"""Check the dashboard's two prediction paths and their input boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "module3"))
from predict import predict_students  # noqa: E402


class PredictionModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = pd.read_csv(PROJECT / "module5" / "examples" / "sample_students.csv")

    def test_mock_marks_do_not_change_default_prediction(self) -> None:
        baseline = predict_students(self.sample)
        changed_marks = self.sample.copy()
        for column in ("mock_math_score", "mock_reading_score", "mock_writing_score", "mock_science_score"):
            changed_marks[column] = 0
        pd.testing.assert_frame_equal(baseline, predict_students(changed_marks))

    def test_opt_in_model_requires_all_four_marks(self) -> None:
        with self.assertRaisesRegex(ValueError, "All four earlier mock marks"):
            predict_students(self.sample.drop(columns="mock_math_score"), use_mock_marks=True)
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            invalid = self.sample.copy()
            invalid["mock_math_score"] = 101
            predict_students(invalid, use_mock_marks=True)

    def test_opt_in_changes_score_and_level_matches_displayed_score(self) -> None:
        standard = predict_students(self.sample)
        with_marks = predict_students(self.sample, use_mock_marks=True)
        self.assertFalse(standard.predicted_score.equals(with_marks.predicted_score))
        for score, level in zip(with_marks.predicted_score, with_marks.predicted_level):
            expected = "Low" if score < 50 else "Average" if score <= 65 else "High"
            self.assertEqual(level, expected)


if __name__ == "__main__":
    unittest.main()
