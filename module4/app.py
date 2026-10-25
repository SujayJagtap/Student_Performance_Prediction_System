"""Module 4: local student performance dashboard for the SkillOrbit capstone."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MODULE1 = PROJECT / "module1"
MODULE3 = PROJECT / "module3"
MODULE5 = PROJECT / "module5"
sys.path.insert(0, str(MODULE3))
from predict import predict_students  # noqa: E402
sys.path.insert(0, str(MODULE5))
from recommend import generate_recommendations  # noqa: E402


st.set_page_config(
    page_title="SkillOrbit | Student Performance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto",
)

TEAL = "#0b9a9b"
DARK = "#19323d"
MUTED = "#71858f"
BLUE = "#7295e8"
PALE = "#eef8fa"
LEVEL_COLORS = {"Low": "#7ccbd0", "Average": "#5f9eec", "High": "#099b93"}
SUBJECTS = {"math_score": "Maths", "reading_score": "Reading", "writing_score": "Writing", "science_score": "Science"}


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cleaned = pd.read_csv(MODULE1 / "data/processed/cleaned_student_data.csv")
    manifest = pd.read_csv(MODULE1 / "data/processed/split_manifest.csv")
    predictions = pd.read_csv(MODULE3 / "tables/test_predictions.csv")
    metadata = json.loads((MODULE3 / "artifacts/model_metadata.json").read_text())
    if cleaned.student_id.duplicated().any() or len(cleaned) != 10_000:
        raise ValueError("Module 1 cleaned student data is missing or inconsistent.")
    if len(manifest) != 10_000 or manifest.student_id.duplicated().any():
        raise ValueError("Module 1 split manifest is inconsistent.")
    train_ids = set(manifest.loc[manifest.split.eq("train"), "student_id"])
    train = cleaned.loc[cleaned.student_id.isin(train_ids)].copy()
    if len(train) != 8_000 or len(predictions) != 2_000:
        raise ValueError("Expected 8,000 training students and 2,000 held-out predictions.")
    cleaned["performance_level"] = level_from_score(cleaned.final_exam_score)
    train["performance_level"] = level_from_score(train.final_exam_score)
    return cleaned, train, predictions, metadata


def level_from_score(values: pd.Series | np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    return np.select([scores < 50, scores <= 65], ["Low", "Average"], default="High")


def apply_style() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;}
[data-testid="stApp"] {background: radial-gradient(circle at 91% 7%, #d7f3f5 0, transparent 25%), radial-gradient(circle at 16% 83%, #d7f0f5 0, transparent 27%), #eaf4f7; color: #19323d;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stAppDeployButton"] {display:none;}
.block-container {max-width: 1550px; padding-top: 2.1rem; padding-bottom: 4rem;}
[data-testid="stSidebar"] {background: rgba(255,255,255,.89); border-right: 1px solid #dbeaed;}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding: 1.2rem .95rem;}
[data-testid="stSidebar"] .stRadio label {font-weight: 600; color: #344c56;}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {padding: .58rem .68rem; border-radius: 12px; margin: .12rem 0;}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) {background: #e3f4f5; color: #087f83;}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) p {color: #087f83;}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {font-size: .91rem;}
h1,h2,h3 {letter-spacing: -.035em; color: #19323d;}
h1 {font-size: 2.05rem !important; margin-bottom: .05rem !important;}
h2 {font-size: 1.33rem !important;}
h3 {font-size: 1.03rem !important;}
p {color: #617783;}
.brand {display:flex;align-items:center;gap:.62rem;font-size:1.27rem;font-weight:800;color:#19323d;margin: .2rem .35rem 1.8rem;letter-spacing:-.04em;}
.brand-icon {width:34px;height:34px;border-radius:10px;background:linear-gradient(145deg,#55d0ca,#007c85);display:grid;place-items:center;color:white;font-weight:900;box-shadow:0 6px 14px #0b9a9b36;}
.eyebrow {font-size:.72rem;text-transform:uppercase;letter-spacing:.15em;color:#0b9295;font-weight:800;margin-bottom:.22rem;}
.subheading {color:#728791;margin:0 0 1.45rem;font-size:.94rem;}
.metric-card {background:rgba(255,255,255,.84);border:1px solid rgba(255,255,255,.95);border-radius:18px;box-shadow:0 11px 28px #4b8b9a13;padding:1.08rem 1.02rem;min-height:145px;}
.metric-top {display:flex;justify-content:space-between;align-items:center;color:#71858f;font-size:.72rem;font-weight:700;min-height:28px;}
.metric-icon {display:grid;place-items:center;width:35px;height:35px;border-radius:11px;background:#e6f6f5;color:#078d8d;font-size:1.08rem;}
.metric-value {font-size:1.48rem;line-height:1.25;color:#19323d;font-weight:800;letter-spacing:-.045em;margin:.49rem 0 .1rem;}
.metric-note {color:#759099;font-size:.71rem;line-height:1.4;}
.section-title {font-size:1.03rem;font-weight:800;letter-spacing:-.025em;color:#203943;margin:.1rem 0 .25rem;}
.section-subtitle {font-size:.74rem;color:#7c929b;margin-bottom:.65rem;}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .section-title) {background:rgba(255,255,255,.86);border:1px solid rgba(255,255,255,.96);border-radius:18px;box-shadow:0 11px 28px #4b8b9a12;}
.pill {display:inline-block;padding:.37rem .7rem;border-radius:999px;background:#e5f6f3;color:#008f8b;font-size:.73rem;font-weight:800;}
.small-note {color:#82959e;font-size:.74rem;line-height:1.5;}
.profile-line {display:flex;justify-content:space-between;padding:.48rem 0;border-bottom:1px solid #eaf2f4;color:#617783;font-size:.82rem;}
.profile-line b {color:#203943;}
.prediction-hero {background:linear-gradient(125deg,#087e87,#0db1a9);color:white;border-radius:20px;padding:1.4rem 1.6rem;box-shadow:0 14px 30px #07888b32;}
.prediction-hero .label {color:#ccfbf7;font-size:.8rem;font-weight:700;}
.prediction-hero .number {font-size:3.55rem;line-height:1.1;font-weight:800;letter-spacing:-.06em;color:white;margin:.15rem 0;}
.prediction-hero .level {display:inline-block;background:#ffffff24;padding:.35rem .72rem;border-radius:999px;color:white;font-size:.83rem;font-weight:800;}
.prediction-hero .fine {color:#d7f8f4;font-size:.77rem;margin-top:.65rem;}
.divider-label {color:#96a7ad;font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;font-weight:800;margin:1.35rem .1rem .75rem;}
div.stButton > button[kind="primary"], div.stDownloadButton > button {border-radius:10px;font-weight:700;}
div.stButton > button[kind="primary"] {background:#0a9998;border-color:#0a9998;}
div[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
@media(max-width:850px){.block-container{padding-top:1.15rem}.metric-card{min-height:112px}.metric-value{font-size:1.3rem}[data-testid="stHorizontalBlock"]{flex-wrap:wrap}[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{min-width:220px;flex:1 1 220px}}
</style>
""",
        unsafe_allow_html=True,
    )


def card(label: str, value: str, note: str, icon: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-top"><span>{label}</span><span class="metric-icon">{icon}</span></div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def chart_layout(height: int = 290, **extra: object) -> dict:
    return dict(
        height=height, margin=dict(l=16, r=16, t=16, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Arial, sans-serif", color=DARK, size=11),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#dbe8eb", tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor="#e9f0f2", zeroline=False, tickfont=dict(color=MUTED)),
        showlegend=False,
        **extra,
    )


def panel(title: str, subtitle: str):
    box = st.container(border=True)
    with box:
        st.markdown(f'<div class="section-title">{title}</div><div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    return box


def score_pattern(data: pd.DataFrame) -> go.Figure:
    bands = pd.cut(
        data.study_hours_per_day, bins=[0, 2, 3, 4, 5, 24],
        labels=["<2 h", "2–<3 h", "3–<4 h", "4–<5 h", "5+ h"],
        include_lowest=True, right=False,
    )
    grouped = data.groupby(bands, observed=False).final_exam_score.agg(["mean", "count"]).dropna()
    fig = go.Figure(go.Scatter(
        x=grouped.index.astype(str), y=grouped["mean"], mode="lines+markers",
        line=dict(color=TEAL, width=3, shape="spline"), marker=dict(size=9, color=TEAL, line=dict(color="white", width=2)),
        fill="tozeroy", fillcolor="rgba(11,154,155,.12)",
        customdata=grouped["count"],
        hovertemplate="Study time: %{x}<br>Average score: %{y:.1f}<br>Students: %{customdata:,}<extra></extra>",
    ))
    fig.update_layout(**chart_layout(285))
    fig.update_yaxes(range=[0, max(75, float(grouped["mean"].max()) + 8)], title="Average final score")
    return fig


def level_donut(data: pd.DataFrame) -> go.Figure:
    counts = data.performance_level.value_counts().reindex(["Low", "Average", "High"], fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=.72,
        marker=dict(colors=[LEVEL_COLORS[x] for x in counts.index], line=dict(color="white", width=4)),
        textinfo="none", hovertemplate="%{label}: %{value:,} students (%{percent})<extra></extra>",
    ))
    layout = chart_layout(230)
    layout.update(showlegend=True, legend=dict(orientation="h", y=-.13, x=.5, xanchor="center", font=dict(size=10)),
                  annotations=[dict(text=f"<b>{len(data):,}</b><br><span style='font-size:10px'>students</span>", x=.5, y=.5, showarrow=False, font=dict(color=DARK, size=17))])
    fig.update_layout(**layout)
    return fig


def score_distribution(data: pd.DataFrame) -> go.Figure:
    counts, edges = np.histogram(data.final_exam_score, bins=np.arange(0, 105, 5))
    fig = go.Figure(go.Bar(x=edges[:-1] + 2.5, y=counts, width=4.3, marker=dict(color=TEAL, line=dict(width=0)),
                           hovertemplate="Score band: %{x:.0f} ± 2.5<br>Students: %{y:,}<extra></extra>"))
    fig.update_layout(**chart_layout(255), bargap=.13)
    fig.update_xaxes(title="Final exam score", range=[0, 100])
    fig.update_yaxes(title="Students")
    return fig


def subject_means(data: pd.DataFrame) -> go.Figure:
    means = data[list(SUBJECTS)].mean().rename(index=SUBJECTS).sort_values()
    fig = go.Figure(go.Bar(
        x=means.values, y=means.index, orientation="h",
        marker=dict(color=["#83cbd0", "#68b8c4", "#39aab2", TEAL], line=dict(width=0)),
        text=[f"{v:.1f}" for v in means], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(**chart_layout(255))
    fig.update_xaxes(range=[0, 65], title="Average subject score")
    return fig


def header(title: str, subtitle: str, cohort: pd.DataFrame) -> None:
    left, right = st.columns([2.15, 1], vertical_alignment="bottom")
    with left:
        st.markdown(f'<div class="eyebrow">Academic analytics</div><h1>{title}</h1><p class="subheading">{subtitle}</p>', unsafe_allow_html=True)
    with right:
        search = st.text_input("Quick student lookup", placeholder="Search ID, e.g. S00001", key=f"search_{title}")
    if search.strip():
        match = cohort.loc[cohort.student_id.str.upper().eq(search.strip().upper())]
        if match.empty:
            st.info("No student found for that ID. Try a value such as S00001.")
        else:
            row = match.iloc[0]
            st.success(f"{row.student_id} · final score {row.final_exam_score:.1f} · {row.performance_level} level · study time {row.study_hours_per_day:.1f} h/day")


def overview(cleaned: pd.DataFrame, predictions: pd.DataFrame, metadata: dict, filtered: pd.DataFrame) -> None:
    header("Overview", "A clear view of the student performance dataset and model results.", cleaned)
    if filtered.empty:
        st.warning("No training students match these filters. Adjust the sidebar filters.")
        return

    high = int(filtered.performance_level.eq("High").sum())
    metrics = [
        ("Students analyzed", f"{len(filtered):,}", "Training cohort · filtered view", "♧"),
        ("Average final score", f"{filtered.final_exam_score.mean():.1f}", "Out of 100", "◫"),
        ("High performers", f"{high:,}", f"{high / len(filtered):.1%} of current view", "✦"),
        ("Prediction error", f"{metadata['test_score_metrics']['mae']:.2f}", "Default model · test MAE · points", "↗"),
        ("Level accuracy", f"{metadata['test_score_derived_level_metrics']['accuracy']:.1%}", "Default model · held-out test", "◉"),
    ]
    cols = st.columns(5, gap="small")
    for col, item in zip(cols, metrics):
        with col:
            card(*item)

    st.markdown('<div class="divider-label">Performance overview</div>', unsafe_allow_html=True)
    left, middle, right = st.columns([1.9, 1.05, 1.12], gap="medium")
    with left:
        with panel("Score by study time", "Average final score across study-hour bands · not a time series"):
            st.plotly_chart(score_pattern(filtered), use_container_width=True, config={"displayModeBar": False})
    with middle:
        with panel("Performance levels", "Current training cohort view"):
            st.plotly_chart(level_donut(filtered), use_container_width=True, config={"displayModeBar": False})
    with right:
        with panel("Held-out predictions", "Examples from the 2,000-student test set"):
            sample = predictions.iloc[:5][["student_id", "actual_score", "predicted_score"]].copy()
            sample.columns = ["Student", "Actual", "Predicted"]
            sample[["Actual", "Predicted"]] = sample[["Actual", "Predicted"]].round(1)
            st.dataframe(sample, hide_index=True, width="stretch", height=232)
            st.caption("Examples are in saved split order; the dataset has no dates.")

    bottom_left, bottom_right = st.columns([1.36, 1], gap="medium")
    with bottom_left:
        with panel("Final score distribution", "How scores are spread across the current training view"):
            st.plotly_chart(score_distribution(filtered), use_container_width=True, config={"displayModeBar": False})
    with bottom_right:
        with panel("Subject-wise comparison", "Mean of recorded subject scores"):
            st.plotly_chart(subject_means(filtered), use_container_width=True, config={"displayModeBar": False})

    st.caption("Cohort charts use Module 1 training records. Test metrics use the held-out set. Associations do not prove that changing an input will change a score.")


def student_explorer(cleaned: pd.DataFrame) -> None:
    header("Student Explorer", "Compare individual records with the cohort and across subjects.", cleaned)
    default_ids = "S00001, S00002, S00003"
    entered = st.text_input("Student IDs to compare", value=default_ids, help="Enter up to four IDs separated by commas.")
    ids = list(dict.fromkeys(x.strip().upper() for x in entered.split(",") if x.strip()))
    if len(ids) > 4:
        st.warning("Showing the first four IDs. Enter up to four students for a readable comparison.")
        ids = ids[:4]
    chosen = cleaned.set_index("student_id").reindex(ids).dropna(subset=["final_exam_score"]).reset_index()
    missing = [x for x in ids if x not in set(chosen.student_id)]
    if missing:
        st.info(f"No record found for: {', '.join(missing)}")
    if chosen.empty:
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Students selected", str(len(chosen)), "Up to four side by side", "♧")
    with c2:
        card("Selected average", f"{chosen.final_exam_score.mean():.1f}", "Final exam score", "◫")
    with c3:
        card("Full cohort average", f"{cleaned.final_exam_score.mean():.1f}", "Reference · all 10,000 records", "↗")

    st.markdown('<div class="divider-label">Student comparison</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        with panel("Final score comparison", "Selected students and full cohort average"):
            fig = go.Figure(go.Bar(x=chosen.student_id, y=chosen.final_exam_score, marker_color=TEAL,
                                   text=chosen.final_exam_score.round(1), textposition="outside"))
            fig.add_hline(y=cleaned.final_exam_score.mean(), line_dash="dash", line_color=BLUE,
                          annotation_text="Cohort mean", annotation_position="top right")
            fig.update_layout(**chart_layout(300))
            fig.update_yaxes(range=[0, 105], title="Final exam score")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        with panel("Study and attendance", "Inputs recorded for selected students"):
            display = chosen[["student_id", "study_hours_per_day", "attendance_rate", "previous_gpa", "performance_level"]].copy()
            display.columns = ["Student", "Study h/day", "Attendance %", "Previous GPA", "Level"]
            display[["Study h/day", "Attendance %"]] = display[["Study h/day", "Attendance %"]].round(1)
            display["Previous GPA"] = display["Previous GPA"].round(2)
            st.dataframe(display,
                         width="stretch", hide_index=True, height=295)

    with panel("Subject profile", "Recorded marks by student · descriptive only"):
        fig = go.Figure()
        colors = [TEAL, BLUE, "#6bc6c0", "#8a90df"]
        for (column, subject), color in zip(SUBJECTS.items(), colors):
            fig.add_bar(name=subject, x=chosen.student_id, y=chosen[column], marker_color=color,
                        hovertemplate="%{x}<br>" + subject + ": %{y:.1f}<extra></extra>")
        layout = chart_layout(300)
        layout.update(barmode="group", showlegend=True, legend=dict(orientation="h", y=1.17, x=0))
        fig.update_layout(**layout)
        fig.update_yaxes(range=[0, 105], title="Subject score")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("These are recorded outcomes, not a timeline. Student comparisons are descriptive and do not establish cause and effect.")


def prediction_studio(cleaned: pd.DataFrame, train: pd.DataFrame, metadata: dict) -> None:
    header("Predict Performance", "Enter information available before the final exam.", cleaned)
    mock_metadata = json.loads((MODULE3 / "artifacts/mock_marks_model_metadata.json").read_text())
    model_signature = (
        f"{metadata['selected_score_model']}:{','.join(metadata['raw_model_features'])}:"
        f"{mock_metadata['selected_score_model']}:{','.join(mock_metadata['raw_model_features'])}"
    )
    if (st.session_state.get("prediction_model_signature") != model_signature
            or ("last_prediction" in st.session_state and "last_inputs" not in st.session_state)):
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("last_inputs", None)
        st.session_state.pop("mock_subject_marks", None)
        st.session_state.pop("last_prediction_include_mock_marks", None)
        st.session_state["prediction_model_signature"] = model_signature
    include_mock_marks = st.radio(
        "Include earlier mock marks in this prediction?", ["No", "Yes"], horizontal=True,
        key="include_mock_marks_choice",
    ) == "Yes"
    if ("last_prediction_include_mock_marks" in st.session_state
            and st.session_state["last_prediction_include_mock_marks"] != include_mock_marks):
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("last_inputs", None)
        st.session_state.pop("mock_subject_marks", None)
        st.session_state.pop("last_prediction_include_mock_marks", None)
    st.markdown('<div class="divider-label">Student inputs</div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        a, b, c = st.columns(3, gap="large")
        with a:
            study = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=3.2, step=.1)
            attendance = st.number_input("Attendance rate (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
            gpa = st.number_input("Previous GPA (0–4)", min_value=0.0, max_value=4.0, value=2.5, step=.1)
            assignment = st.number_input("Assignment completion (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        with b:
            sleep = st.number_input("Sleep hours per day", min_value=0.0, max_value=24.0, value=7.0, step=.1)
            social = st.number_input("Social media hours per day", min_value=0.0, max_value=24.0, value=2.0, step=.1)
            participation = st.number_input("Participation score (0–100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            courses = st.number_input("Online courses completed", min_value=0, max_value=100, value=1, step=1)
        with c:
            internet = st.selectbox("Internet access", ["Yes", "No"])
            environment = st.selectbox("Study environment", ["Quiet", "Moderate", "Noisy"])
            tutoring = st.selectbox("Tutoring", ["No", "Yes"])
            if include_mock_marks:
                st.markdown('<p class="small-note" style="margin-top:1.2rem">This model uses the 11 standard fields plus four earlier mock marks. The final exam score and outcome labels are not predictors.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="small-note" style="margin-top:1.2rem">The model uses 11 fields. Subject marks, the final exam score, and outcome labels are not predictors.</p>', unsafe_allow_html=True)
        mock_marks = {}
        if include_mock_marks:
            st.markdown("**Earlier mock marks (out of 100)**")
            for column, (name, label) in zip(
                st.columns(4, gap="small"),
                [("math", "Maths"), ("reading", "Reading"), ("writing", "Writing"), ("science", "Science")],
            ):
                with column:
                    mock_marks[name] = st.number_input(
                        f"{label} earlier mock mark", min_value=0.0, max_value=100.0,
                        value=None, step=1.0,
                    )
        submitted = st.form_submit_button("Generate prediction", type="primary", width="stretch")

    if submitted:
        if include_mock_marks and any(mark is None for mark in mock_marks.values()):
            st.warning("Enter all four earlier mock marks to use them in the prediction.")
        else:
            inputs = pd.DataFrame([{
                "study_hours_per_day": study, "attendance_rate": attendance, "sleep_hours": sleep,
                "social_media_hours": social, "assignment_completion_rate": assignment,
                "participation_score": participation, "online_courses_completed": courses,
                "previous_gpa": gpa, "internet_access": internet,
                "study_environment": environment, "tutoring": tutoring,
            }])
            if include_mock_marks:
                for name, mark in mock_marks.items():
                    inputs[f"mock_{name}_score"] = mark
            st.session_state["last_prediction"] = predict_students(
                inputs, use_mock_marks=include_mock_marks
            ).iloc[0].to_dict()
            st.session_state["last_inputs"] = inputs.iloc[0].to_dict()
            st.session_state["last_prediction_include_mock_marks"] = include_mock_marks
            if include_mock_marks:
                st.session_state["mock_subject_marks"] = mock_marks
            else:
                st.session_state.pop("mock_subject_marks", None)
    result = st.session_state.get("last_prediction")
    if result:
        st.markdown('<div class="divider-label">Prediction result</div>', unsafe_allow_html=True)
        left, right = st.columns([1, 1.5], gap="medium")
        with left:
            st.markdown(
                f'<div class="prediction-hero"><div class="label">Predicted final exam score</div>'
                f'<div class="number">{result["predicted_score"]:.2f}<span style="font-size:1.2rem"> / 100</span></div>'
                f'<span class="level">{result["predicted_level"]} performance</span>'
                f'<div class="fine">Low &lt; 50 · Average 50–65 · High &gt; 65</div></div>', unsafe_allow_html=True,
            )
        with right:
            with panel("How to read this", "Model output on the selected dataset"):
                st.write(f"This prediction is **{result['predicted_score'] - train.final_exam_score.mean():+.1f} points** from the training cohort's average final score of **{train.final_exam_score.mean():.1f}**.")
                result_metrics = mock_metadata["test_metrics"] if include_mock_marks else metadata["test_score_metrics"]
                st.write(f"On the separate test set, the model's mean absolute error was **{result_metrics['mae']:.2f} points**. That is an average evaluation result, not a guaranteed range for one student.")
                if include_mock_marks:
                    st.caption("This model assumes the four subject marks were known before the final exam. Their timing has not been independently verified from the dataset source.")
                st.caption("The source and real-world representativeness of this dataset have not been verified.")

        st.markdown('<div class="divider-label">Study plan</div>', unsafe_allow_html=True)
        st.caption("These are rule-based areas to review, not model explanations or guaranteed score gains. Discuss the plan with a teacher.")
        if include_mock_marks:
            st.caption("The earlier mock marks were used for this prediction and can guide subject-focus advice below.")
        else:
            with st.expander("Optional: add earlier mock marks for a subject focus"):
                st.caption("These four marks are used only for subject advice. They never change the predicted score or level.")
                with st.form("mock_marks_form"):
                    mark_columns = st.columns(4, gap="small")
                    study_marks = {}
                    for column, (name, label) in zip(mark_columns, [("math", "Maths"), ("reading", "Reading"), ("writing", "Writing"), ("science", "Science")]):
                        with column:
                            study_marks[name] = st.number_input(f"{label} mock mark", min_value=0.0, max_value=100.0, value=None, step=1.0)
                    save_marks = st.form_submit_button("Add marks to study plan")
                if save_marks:
                    if any(mark is None for mark in study_marks.values()):
                        st.warning("Enter all four mock marks to compare subjects.")
                    else:
                        st.session_state["mock_subject_marks"] = study_marks
                if st.session_state.get("mock_subject_marks"):
                    st.success("Mock marks added to the study plan; the prediction is unchanged.")
                    if st.button("Remove mock marks"):
                        st.session_state.pop("mock_subject_marks", None)

        suggestions = generate_recommendations(
            st.session_state["last_inputs"],
            predicted_level=result["predicted_level"],
            subject_marks=st.session_state.get("mock_subject_marks"),
        )
        for start in range(0, len(suggestions), 2):
            for column, suggestion in zip(st.columns(2, gap="medium"), suggestions[start:start + 2]):
                with column:
                    with st.container(border=True):
                        st.markdown(f"**{suggestion['title']}**")
                        st.write(suggestion["action"])
                        st.caption(suggestion["reason"])
    else:
        st.info("Fill in the fields and select Generate prediction to see an estimated score and performance level.")


def model_insights(cleaned: pd.DataFrame, predictions: pd.DataFrame, metadata: dict) -> None:
    header("Model Insights", "See how the prediction system performed on unseen test records.", cleaned)
    metrics = metadata["test_score_metrics"]
    levels = metadata["test_score_derived_level_metrics"]
    mock_metadata = json.loads((MODULE3 / "artifacts/mock_marks_model_metadata.json").read_text())
    cols = st.columns(4)
    for col, item in zip(cols, [
        ("Test students", f"{len(predictions):,}", "Held out during training", "♧"),
        ("Score MAE", f"{metrics['mae']:.2f}", "Average absolute error · points", "↗"),
        ("Score R²", f"{metrics['r2']:.3f}", "Held-out test fit", "◫"),
        ("Level macro F1", f"{levels['macro_f1']:.3f}", "Balances all three levels", "◉"),
    ]):
        with col:
            card(*item)
    st.markdown('<div class="divider-label">Prediction quality</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        with panel("Actual vs predicted", "Each dot is one held-out test student"):
            fig = go.Figure(go.Scattergl(x=predictions.actual_score, y=predictions.predicted_score,
                                        mode="markers", marker=dict(size=5, color=TEAL, opacity=.35),
                                        customdata=predictions.student_id,
                                        hovertemplate="%{customdata}<br>Actual %{x:.1f}<br>Predicted %{y:.1f}<extra></extra>"))
            fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color=BLUE, dash="dash", width=2))
            fig.update_layout(**chart_layout(360))
            fig.update_xaxes(range=[0, 100], title="Actual final score")
            fig.update_yaxes(range=[0, 100], title="Predicted final score")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        with panel("Performance level accuracy", "Rows are actual; columns are predicted"):
            order = ["Low", "Average", "High"]
            cross = pd.crosstab(predictions.actual_level, predictions.score_derived_predicted_level).reindex(index=order, columns=order, fill_value=0)
            fig = go.Figure(go.Heatmap(z=cross.values, x=order, y=order, colorscale=[[0, "#eff9f9"], [1, "#099b93"]],
                                       showscale=False, text=cross.values, texttemplate="%{text}",
                                       hovertemplate="Actual %{y}<br>Predicted %{x}<br>Students %{z}<extra></extra>"))
            fig.update_layout(**chart_layout(360))
            fig.update_xaxes(title="Predicted level")
            fig.update_yaxes(title="Actual level", autorange="reversed")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        with panel("Test prediction table", "Filter by actual performance level"):
            level_filter = st.selectbox("Actual level", ["All", "Low", "Average", "High"], key="test_level_filter")
            shown = predictions if level_filter == "All" else predictions.loc[predictions.actual_level.eq(level_filter)]
            shown = shown[["student_id", "actual_score", "predicted_score", "actual_level", "score_derived_predicted_level"]].copy()
            shown.columns = ["Student ID", "Actual score", "Predicted score", "Actual level", "Predicted level"]
            shown[["Actual score", "Predicted score"]] = shown[["Actual score", "Predicted score"]].round(1)
            st.dataframe(shown, hide_index=True, width="stretch", height=260)
            st.download_button("Download shown predictions", shown.to_csv(index=False).encode("utf-8"), "module4_test_predictions.csv", "text/csv")
    with right:
        with panel("What the model can and cannot show", "Evaluation context"):
            st.write(f"**Selected score model:** {metadata['selected_score_model']}, chosen by five-fold cross-validation on training records.")
            st.write(f"**Per-level recall:** Low {levels['per_class']['Low']['recall']:.1%} · Average {levels['per_class']['Average']['recall']:.1%} · High {levels['per_class']['High']['recall']:.1%}.")
            st.write("**Inputs:** Study behavior, attendance, previous GPA, and learning environment. Subject marks and source outcome labels are excluded.")
            st.caption("This default model excludes subject marks. The optional second model treats them as earlier mock marks based on the user's confirmation; the original data source and timing remain independently unverified.")

    with panel("Optional earlier-marks model", "Used only when a user chooses to include four earlier mock marks"):
        with_marks = mock_metadata["test_metrics"]
        st.write(
            f"The separate Polynomial Ridge model reached **{with_marks['level_accuracy']:.2%}** "
            f"Low/Average/High accuracy, **{with_marks['level_macro_f1']:.3f}** macro F1, and "
            f"**{with_marks['mae']:.2f} points** MAE on the same 2,000-student test set."
        )
        st.caption("These results assume the subject columns are earlier mock marks, as confirmed by the user. Their timing and the dataset source have not been independently verified; the test set is not fresh external validation.")


def main() -> None:
    apply_style()
    try:
        cleaned, train, predictions, metadata = load_data()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.error(f"Required Module 1 or Module 3 file is unavailable: {exc}")
        st.stop()

    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-icon">S</span><span>SkillOrbit</span></div>', unsafe_allow_html=True)
        page = st.radio("Navigate", ["Overview", "Student Explorer", "Predict Performance", "Model Insights"], label_visibility="collapsed")
        st.markdown('<div class="divider-label">Explore cohort</div>', unsafe_allow_html=True)
        level = st.selectbox("Performance level", ["All", "Low", "Average", "High"])
        attendance_range = st.slider("Attendance rate (%)", min_value=0, max_value=100, value=(0, 100))
        st.markdown('<div class="divider-label">Dataset</div>', unsafe_allow_html=True)
        st.markdown('<span class="pill">10,000 records</span>', unsafe_allow_html=True)
        st.caption("Analytics: 8,000 training records\n\nEvaluation: 2,000 test records")
        st.markdown('<p class="small-note">Capstone project · Modules 4–5<br>Dashboard and study plan</p>', unsafe_allow_html=True)

    filtered = train.copy()
    if level != "All":
        filtered = filtered.loc[filtered.performance_level.eq(level)]
    filtered = filtered.loc[filtered.attendance_rate.between(*attendance_range)]

    if page == "Overview":
        overview(cleaned, predictions, metadata, filtered)
    elif page == "Student Explorer":
        student_explorer(cleaned)
    elif page == "Predict Performance":
        prediction_studio(cleaned, train, metadata)
    else:
        model_insights(cleaned, predictions, metadata)


if __name__ == "__main__":
    main()
