"""Metrics computation and plot generation."""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SKILLS = ["regex", "json_proc", "file_ops", "math_compute", "text_transform", "data_struct"]
CONDITIONS = ["random", "fixed", "metacognitive"]
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "plots")


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def learning_gain(initial_acc: float, final_acc: float) -> float:
    return round(final_acc - initial_acc, 4)


def skill_accuracy(eval_logs: list[dict]) -> dict[str, float]:
    """Per-skill accuracy from a list of eval log entries."""
    counts: dict[str, list[int]] = defaultdict(list)
    for entry in eval_logs:
        counts[entry["skill"]].append(int(entry["passed"]))
    return {
        skill: round(sum(v) / len(v), 4) if v else 0.0
        for skill, v in counts.items()
    }


def calibration_error(practice_logs: list[dict]) -> float:
    """Mean absolute error between estimated competence and actual success rate.

    Only applicable to metacognitive runs that carry estimated_current_competence.
    """
    skill_estimates: dict[str, list[float]] = defaultdict(list)
    skill_actual: dict[str, list[int]] = defaultdict(list)

    for entry in practice_logs:
        if entry.get("estimated_current_competence") is not None:
            skill = entry["skill"]
            skill_estimates[skill].append(float(entry["estimated_current_competence"]))
            skill_actual[skill].append(int(entry["passed"]))

    if not skill_estimates:
        return float("nan")

    errors = []
    for skill in skill_estimates:
        est = np.mean(skill_estimates[skill])
        act = np.mean(skill_actual[skill])
        errors.append(abs(est - act))

    return round(float(np.mean(errors)), 4)


def task_selection_quality(practice_logs: list[dict]) -> float:
    """Fraction of metacognitive selections that targeted a weak skill.

    A skill is "weak" if its current success rate is below the agent's average.
    Evaluated at the moment of each selection using the history so far.
    """
    cumulative: dict[str, list[int]] = defaultdict(list)
    weak_selections = 0
    total = 0

    for entry in practice_logs:
        if entry.get("selected_by") != "llm_metacognition":
            continue

        # Compute current rates before this task
        if cumulative:
            rates = {
                s: sum(v) / len(v) for s, v in cumulative.items() if v
            }
            avg = np.mean(list(rates.values())) if rates else 0.5
            skill_rate = rates.get(entry["skill"], 0.0)
            if skill_rate < avg:
                weak_selections += 1
        else:
            # No history yet — first selection always counts as targeting weak
            weak_selections += 1

        total += 1
        cumulative[entry["skill"]].append(int(entry["passed"]))

    return round(weak_selections / total, 4) if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "condition": r["condition"],
                "seed": r["seed"],
                "initial_accuracy": r["initial_accuracy"],
                "final_accuracy": r["final_accuracy"],
                "learning_gain": r["learning_gain"],
                "practice_accuracy": r.get("practice_accuracy", float("nan")),
                "calibration_error": r.get("calibration_error", float("nan")),
                "task_selection_quality": r.get("task_selection_quality", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("condition")
        .agg(
            initial_accuracy=("initial_accuracy", "mean"),
            final_accuracy=("final_accuracy", "mean"),
            learning_gain=("learning_gain", "mean"),
            learning_gain_std=("learning_gain", "std"),
            practice_accuracy=("practice_accuracy", "mean"),
            calibration_error=("calibration_error", "mean"),
            task_selection_quality=("task_selection_quality", "mean"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_accuracy_by_condition(df: pd.DataFrame):
    """Bar chart: initial vs final accuracy per condition."""
    _ensure_plots_dir()
    agg = aggregate_summary(df)
    agg = agg.set_index("condition").reindex(CONDITIONS).dropna(how="all")

    x = np.arange(len(agg))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, agg["initial_accuracy"], width, label="Initial", color="#6baed6")
    bars2 = ax.bar(x + width / 2, agg["final_accuracy"], width, label="Final", color="#2171b5")

    ax.set_xlabel("Condition")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Before and After Practice")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Annotate learning gain
    for xi, (init, final) in enumerate(zip(agg["initial_accuracy"], agg["final_accuracy"])):
        gain = final - init
        ax.text(xi, final + 0.03, f"+{gain:.2f}", ha="center", fontsize=9, color="#08519c")

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "accuracy_by_condition.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_learning_gain_by_condition(df: pd.DataFrame):
    """Bar chart with error bars: mean learning gain per condition."""
    _ensure_plots_dir()
    agg = aggregate_summary(df)
    agg = agg.set_index("condition").reindex(CONDITIONS).dropna(how="all")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"random": "#fc8d59", "fixed": "#ffffbf", "metacognitive": "#91bfdb"}
    x = np.arange(len(agg))
    bars = ax.bar(
        x,
        agg["learning_gain"],
        yerr=agg["learning_gain_std"].fillna(0),
        color=[colors.get(c, "gray") for c in agg.index],
        capsize=5,
        edgecolor="black",
    )
    ax.set_xlabel("Condition")
    ax.set_ylabel("Learning Gain (Final − Initial Accuracy)")
    ax.set_title("Learning Gain by Curriculum Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for bar, val in zip(bars, agg["learning_gain"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=10,
        )

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "learning_gain_by_condition.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_skill_gain(skill_results: dict):
    """Grouped bar chart: per-skill learning gain across conditions.

    skill_results: {condition: {skill: gain}}
    """
    _ensure_plots_dir()

    skills = SKILLS
    x = np.arange(len(skills))
    n_cond = len(CONDITIONS)
    width = 0.25
    colors = {"random": "#fc8d59", "fixed": "#ffffbf", "metacognitive": "#91bfdb"}

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, cond in enumerate(CONDITIONS):
        gains = [skill_results.get(cond, {}).get(s, 0.0) for s in skills]
        offset = (i - n_cond / 2 + 0.5) * width
        ax.bar(x + offset, gains, width, label=cond, color=colors.get(cond, "gray"), edgecolor="black")

    ax.set_xlabel("Skill")
    ax.set_ylabel("Learning Gain")
    ax.set_title("Skill-wise Learning Gain by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(skills, rotation=15)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "skill_gain.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_calibration(logs: list[dict]):
    """Scatter plot: estimated competence vs actual success rate (metacognitive only)."""
    _ensure_plots_dir()

    estimates, actuals, skills_list = [], [], []
    skill_actual_map: dict[str, list[int]] = defaultdict(list)
    skill_estimate_map: dict[str, list[float]] = defaultdict(list)

    for entry in logs:
        if (
            entry.get("condition") == "metacognitive"
            and entry.get("estimated_current_competence") is not None
            and entry.get("phase") == "practice"
        ):
            skill = entry["skill"]
            skill_estimate_map[skill].append(float(entry["estimated_current_competence"]))
            skill_actual_map[skill].append(int(entry["passed"]))

    for skill in skill_estimate_map:
        estimates.append(np.mean(skill_estimate_map[skill]))
        actuals.append(np.mean(skill_actual_map[skill]))
        skills_list.append(skill)

    if not estimates:
        print("No calibration data to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(estimates, actuals, s=100, zorder=5)
    for x, y, label in zip(estimates, actuals, skills_list):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Estimated Competence")
    ax.set_ylabel("Actual Success Rate")
    ax.set_title("Calibration: Estimated vs Actual Competence (Metacognitive)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "calibration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Load logs from JSONL
# ---------------------------------------------------------------------------

def load_logs(logs_path: str) -> list[dict]:
    if not os.path.exists(logs_path):
        return []
    entries = []
    with open(logs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
