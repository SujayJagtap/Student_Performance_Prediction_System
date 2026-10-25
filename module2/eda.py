"""Module 2: reproducible exploratory analysis of the Module 1 training split."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MODULE1 = PROJECT / "module1" / "data" / "processed"
CLEANED = MODULE1 / "cleaned_student_data.csv"
MANIFEST = MODULE1 / "split_manifest.csv"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

SUBJECTS = {
    "math_score": "Maths",
    "reading_score": "Reading",
    "writing_score": "Writing",
    "science_score": "Science",
}
NUMERIC_FACTORS = {
    "previous_gpa": "Previous GPA",
    "study_hours_per_day": "Study hours per day",
    "attendance_rate": "Attendance rate",
    "participation_score": "Participation score",
    "assignment_completion_rate": "Assignment completion",
    "social_media_hours": "Social media hours",
    "sleep_hours": "Sleep hours",
    "online_courses_completed": "Online courses completed",
}
COLORS = {
    "ink": "#173049",
    "muted": "#607789",
    "teal": "#149f9a",
    "mint": "#7ad8d1",
    "blue": "#5c8fe8",
    "purple": "#8c78d7",
    "orange": "#ed9b60",
    "bg": "#f7fbfc",
    "grid": "#dce8ed",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "savefig.facecolor": COLORS["bg"],
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CLEANED.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError("Run Module 1 first; its cleaned data and split manifest are required.")
    clean = pd.read_csv(CLEANED)
    manifest = pd.read_csv(MANIFEST)
    if clean["student_id"].duplicated().any() or manifest["student_id"].duplicated().any():
        raise ValueError("Student IDs must be unique in the cleaned data and split manifest.")
    if set(clean["student_id"]) != set(manifest["student_id"]):
        raise ValueError("The split manifest does not match the cleaned data.")
    if set(manifest["split"]) != {"train", "test"}:
        raise ValueError("The split manifest must have train and test rows.")
    train_ids = set(manifest.loc[manifest["split"].eq("train"), "student_id"])
    train = clean.loc[clean["student_id"].isin(train_ids)].copy()
    if len(train) != len(train_ids):
        raise ValueError("Training rows could not be matched by student ID.")
    return clean, train


def factor_table(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, label in NUMERIC_FACTORS.items():
        rows.append(
            {
                "factor": label,
                "column": column,
                "pearson_r": train[column].corr(train["final_exam_score"], method="pearson"),
                "spearman_r": train[column].corr(train["final_exam_score"], method="spearman"),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r", key=lambda values: values.abs(), ascending=False)


def make_bins(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    attendance = train.assign(
        band=pd.cut(
            train["attendance_rate"],
            bins=[-np.inf, 70, 80, 90, np.inf],
            labels=["<70%", "70–<80%", "80–<90%", "90–100%"],
            right=False,
        )
    )
    study = train.assign(
        band=pd.cut(
            train["study_hours_per_day"],
            bins=[-np.inf, 2, 3, 4, 5, np.inf],
            labels=["<2 h", "2–<3 h", "3–<4 h", "4–<5 h", "5+ h"],
            right=False,
        )
    )

    def summarize(data: pd.DataFrame) -> pd.DataFrame:
        result = (
            data.groupby("band", observed=True)["final_exam_score"]
            .agg(student_count="size", mean_score="mean", std_score="std")
            .reset_index()
        )
        result["mean_score"] = result["mean_score"].round(2)
        result["std_score"] = result["std_score"].round(2)
        result["band"] = result["band"].astype(str)
        return result

    return summarize(attendance), summarize(study)


def subject_tables(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    summary = pd.DataFrame(
        [
            {
                "subject": label,
                "mean_score": round(float(train[column].mean()), 2),
                "median_score": round(float(train[column].median()), 2),
            }
            for column, label in SUBJECTS.items()
        ]
    )
    scores = train[list(SUBJECTS)]
    minimum = scores.min(axis=1)
    tied = scores.eq(minimum, axis=0).sum(axis=1).gt(1)
    weakest = scores.idxmin(axis=1).map(SUBJECTS).where(~tied, "Tie")
    counts = weakest.value_counts().reindex([*SUBJECTS.values(), "Tie"], fill_value=0)
    weakest_summary = counts.rename_axis("weakest_subject").reset_index(name="student_count")
    return summary, weakest_summary, int(tied.sum())


def chart_score_distribution(train: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    sns.histplot(train["final_exam_score"], bins=np.arange(0, 102, 4), color=COLORS["teal"], ax=ax)
    median = float(train["final_exam_score"].median())
    ax.axvline(median, color=COLORS["purple"], linewidth=2, linestyle="--", label=f"Median: {median:.1f}")
    ax.set(title="Final exam score distribution", xlabel="Final exam score (out of 100)", ylabel="Students")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.text(0.06, 0.01, f"Training records: {len(train):,} · Descriptive distribution", color=COLORS["muted"], fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "01_score_distribution.png")


def chart_factor_relationships(train: pd.DataFrame, factors: pd.DataFrame) -> None:
    chosen = [
        ("study_hours_per_day", "Study hours per day"),
        ("attendance_rate", "Attendance rate (%)"),
        ("participation_score", "Participation score"),
        ("previous_gpa", "Previous GPA (0–4)"),
    ]
    lookup = factors.set_index("column")["pearson_r"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    y = train["final_exam_score"].to_numpy()
    for ax, (column, xlabel) in zip(axes.flat, chosen):
        x = train[column].to_numpy()
        ax.scatter(x, y, s=9, alpha=0.12, color=COLORS["teal"], rasterized=True)
        slope, intercept = np.polyfit(x, y, deg=1)
        domain = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        ax.plot(domain, slope * domain + intercept, color=COLORS["purple"], linewidth=2.2)
        ax.set(title=f"{xlabel} · r = {lookup[column]:+.2f}", xlabel=xlabel, ylabel="Final exam score")
        ax.grid(axis="x", visible=False)
    fig.suptitle("How measured factors relate to final score", fontsize=17, fontweight="bold", y=0.995)
    fig.text(0.06, 0.01, "Each dot is a training record. Lines show simple linear association; they do not establish cause.", color=COLORS["muted"], fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, "02_factor_relationships.png")


def chart_binned_scores(attendance: pd.DataFrame, study: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for ax, table, title, color in [
        (axes[0], attendance, "By attendance rate", COLORS["teal"]),
        (axes[1], study, "By study hours per day", COLORS["blue"]),
    ]:
        bars = ax.bar(table["band"], table["mean_score"], color=color, alpha=0.92, width=0.7)
        ax.set(title=title, ylabel="Average final exam score", ylim=(0, max(85, float(table["mean_score"].max()) + 12)))
        ax.grid(axis="x", visible=False)
        for bar, (_, row) in zip(bars, table.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{row['mean_score']:.1f}\nn={int(row['student_count']):,}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLORS["ink"],
            )
    fig.suptitle("Average score across attendance and study bands", fontsize=17, fontweight="bold", y=0.99)
    fig.text(0.06, 0.01, "Groups are descriptive. Differences do not prove that changing a factor changes a score.", color=COLORS["muted"], fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, "03_grouped_scores.png")


def chart_subjects(train: pd.DataFrame, weakest: pd.DataFrame) -> None:
    long = train[list(SUBJECTS)].rename(columns=SUBJECTS).melt(var_name="subject", value_name="score")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), gridspec_kw={"width_ratios": [1.15, 0.85]})
    sns.boxplot(
        data=long,
        x="subject",
        y="score",
        hue="subject",
        palette=[COLORS["teal"], COLORS["blue"], COLORS["purple"], COLORS["mint"]],
        legend=False,
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set(title="Subject score distributions", xlabel="", ylabel="Score (out of 100)", ylim=(0, 105))
    bars = axes[1].barh(weakest["weakest_subject"], weakest["student_count"], color=[COLORS["teal"], COLORS["blue"], COLORS["purple"], COLORS["mint"], COLORS["orange"]])
    axes[1].invert_yaxis()
    axes[1].set(title="Lowest-scoring subject per student", xlabel="Students", ylabel="")
    axes[1].bar_label(bars, padding=4, fontsize=9, color=COLORS["ink"])
    axes[1].set_xlim(0, float(weakest["student_count"].max()) * 1.2)
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("Subject-wise performance", fontsize=17, fontweight="bold", y=0.99)
    fig.text(0.06, 0.01, "'Lowest' means lowest among the four recorded subjects for that student; ties are counted separately.", color=COLORS["muted"], fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, "04_subject_analysis.png")


def chart_correlations(factors: pd.DataFrame) -> None:
    ordered = factors.sort_values("pearson_r")
    colors = [COLORS["purple"] if value < 0 else COLORS["teal"] for value in ordered["pearson_r"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    bars = ax.barh(ordered["factor"], ordered["pearson_r"], color=colors, height=0.65)
    ax.axvline(0, color=COLORS["ink"], linewidth=1)
    ax.set(title="Correlation with final exam score", xlabel="Pearson correlation (r)", xlim=(-1, 1))
    ax.grid(axis="y", visible=False)
    for bar, value in zip(bars, ordered["pearson_r"]):
        ax.text(
            value + (0.02 if value >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9,
            color=COLORS["ink"],
        )
    fig.text(0.06, 0.01, "Correlation describes association, not the effect of changing a factor.", color=COLORS["muted"], fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "05_factor_correlations.png")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    clean: pd.DataFrame,
    train: pd.DataFrame,
    factors: pd.DataFrame,
    attendance: pd.DataFrame,
    study: pd.DataFrame,
    subjects: pd.DataFrame,
    weakest: pd.DataFrame,
    tie_count: int,
) -> None:
    target = train["final_exam_score"]
    subject_mean = train[list(SUBJECTS)].mean(axis=1)
    grade_counts = train["grade_category"].value_counts().reindex(["A", "B", "C", "D", "F"], fill_value=0)
    score_ge_50 = int(target.ge(50).sum())
    corr_rows = [
        [row.factor, f"{row.pearson_r:+.3f}", f"{row.spearman_r:+.3f}"]
        for row in factors.itertuples(index=False)
    ]
    subject_rows = [
        [row.subject, f"{row.mean_score:.2f}", f"{row.median_score:.2f}"]
        for row in subjects.itertuples(index=False)
    ]
    binned_rows = [
        ["Attendance", row.band, f"{row.mean_score:.2f}", f"{int(row.student_count):,}"]
        for row in attendance.itertuples(index=False)
    ] + [
        ["Study time", row.band, f"{row.mean_score:.2f}", f"{int(row.student_count):,}"]
        for row in study.itertuples(index=False)
    ]
    lowest_rows = [
        [row.weakest_subject, f"{int(row.student_count):,}"]
        for row in weakest.itertuples(index=False)
    ]
    report = f"""# Module 2 — Exploratory data analysis

## Data used

This report analyzes the **{len(train):,} training records** selected in Module 1 from the **{len(clean):,} cleaned records**. The {len(clean) - len(train):,} test records remain reserved for later model evaluation. Scores and factors are displayed in their original units. The data's original collection method and whether records are synthetic have not been verified.

## Overview

- Final exam score: mean **{target.mean():.2f}**, median **{target.median():.2f}**, standard deviation **{target.std():.2f}**, range **{target.min():.1f}–{target.max():.1f}** out of 100.
- Students with a final score of at least 50: **{score_ge_50:,} ({score_ge_50 / len(train):.1%})**. This uses the numeric score rather than the supplied `pass_fail` label.
- Supplied grade counts in the training split: A **{grade_counts['A']:,}**, B **{grade_counts['B']:,}**, C **{grade_counts['C']:,}**, D **{grade_counts['D']:,}**, F **{grade_counts['F']:,}**. The top letter grades are rare.

![Final score distribution](../figures/01_score_distribution.png)

## Relationships with final score

The table ranks the behavioral and previous-performance numeric fields by absolute Pearson correlation. Subject marks are analyzed separately below. A positive value means higher values tend to occur with higher final scores in this dataset; a negative value means the opposite. These are **associations, not causal effects**.

{markdown_table(['Field', 'Pearson r', 'Spearman r'], corr_rows)}

Previous GPA and study hours have the strongest associations among the fields in this table. The four subject scores are shown separately because they are academic marks rather than behavioral factors. Their within-student mean correlates **{subject_mean.corr(target):.3f}** with the final score. Module 3 excludes them from prediction because the CSV does not verify their timing or whether they contribute directly to the final score.

![Factor relationships](../figures/02_factor_relationships.png)

![Factor correlations](../figures/05_factor_correlations.png)

## Attendance and study bands

{markdown_table(['Factor', 'Band', 'Average final score', 'Students'], binned_rows)}

![Grouped average scores](../figures/03_grouped_scores.png)

These are group averages, so differences may reflect other variables and should not be presented as the effect of changing attendance or study time.

## Subject-wise analysis

{markdown_table(['Subject', 'Mean score', 'Median score'], subject_rows)}

Lowest-scoring subject within each student's four recorded scores:

{markdown_table(['Subject', 'Students'], lowest_rows)}

There are **{tie_count:,}** students with a tie for the lowest score. The dashboard can show a selected student's four subject scores and identify the lowest recorded one. That comparison does not establish the reason for the score.

![Subject analysis](../figures/04_subject_analysis.png)

## Implications for the next modules

1. Model development can use `final_exam_score` as the numeric target. Low/Average/High thresholds need to be defined and checked for class balance before training a classification model.
2. Keep `pass_fail` and `grade_category` out of predictor inputs; they describe the outcome. The source has 30 records marked both `Pass` and grade `F`, as documented in Module 1.
3. Keep subject scores in the dashboard for descriptive comparison, but exclude them from model inputs. Verify the original field definitions before considering them for prediction.
4. The CSV contains no dates or term identifiers. A chronological performance-trend chart would require additional data; grouped and subject comparisons are supported.
5. The original data source and any synthetic-data status should be verified and cited in the final project report.
"""
    (REPORTS / "eda_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    for directory in (FIGURES, TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    setup_style()
    clean, train = load_training_data()
    factors = factor_table(train)
    attendance, study = make_bins(train)
    subjects, weakest, tie_count = subject_tables(train)

    factors.round(4).to_csv(TABLES / "factor_correlations.csv", index=False)
    attendance.to_csv(TABLES / "attendance_bands.csv", index=False)
    study.to_csv(TABLES / "study_bands.csv", index=False)
    subjects.to_csv(TABLES / "subject_summary.csv", index=False)
    weakest.to_csv(TABLES / "weakest_subject_counts.csv", index=False)

    chart_score_distribution(train)
    chart_factor_relationships(train, factors)
    chart_binned_scores(attendance, study)
    chart_subjects(train, weakest)
    chart_correlations(factors)
    write_report(clean, train, factors, attendance, study, subjects, weakest, tie_count)
    summary = {
        "cleaned_records": len(clean),
        "analysis_records": len(train),
        "test_records_held_out": len(clean) - len(train),
        "final_score_mean": round(float(train["final_exam_score"].mean()), 4),
        "final_score_median": round(float(train["final_exam_score"].median()), 4),
        "subject_score_mean_correlation_with_final": round(
            float(train[list(SUBJECTS)].mean(axis=1).corr(train["final_exam_score"])), 4
        ),
        "weak_subject_ties": tie_count,
    }
    (TABLES / "eda_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Module 2 complete: {len(train):,} training records, 5 charts, 5 summary tables.")


if __name__ == "__main__":
    main()
