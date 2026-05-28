#!/usr/bin/env python3
"""Main experiment runner.

Usage:
  # Run all conditions with 3 seeds:
  python src/run_experiment.py

  # Run a single condition:
  python src/run_experiment.py --conditions metacognitive --seeds 1

  # Only generate analysis plots from existing logs:
  python src/run_experiment.py --analyze
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

# Make sure sibling modules are importable when called from project root
sys.path.insert(0, os.path.dirname(__file__))

import curriculum
import evaluator
import llm_client
import metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
PROMPTS_DIR = os.path.join(ROOT, "prompts")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
LOGS_PATH = os.path.join(OUTPUTS_DIR, "logs.jsonl")
RESULTS_PATH = os.path.join(OUTPUTS_DIR, "results.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> list | dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt(name: str) -> str:
    with open(os.path.join(PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def append_log(entry: dict):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(LOGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def print_progress(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Evaluation pass (no feedback given)
# ---------------------------------------------------------------------------

def evaluate_tasks(
    eval_tasks: list[dict],
    memory: list[str],
    solve_template: str,
    condition: str,
    seed: int,
    run_id: str,
    phase: str,  # "initial" or "final"
) -> list[dict]:
    results = []
    for task in eval_tasks:
        try:
            code, raw_response = llm_client.solve_task(task, memory, solve_template)
            test_result = evaluator.run_tests(code, task)
        except Exception as exc:
            code = ""
            test_result = {
                "passed": False,
                "passed_tests": 0,
                "num_tests": len(task.get("tests", [])) or 1,
                "error_type": "llm_timeout",
                "feedback": f"LLM call failed: {exc}",
            }

        entry = {
            "run_id": run_id,
            "condition": condition,
            "seed": seed,
            "phase": "eval",
            "phase_detail": phase,
            "task_id": task["id"],
            "skill": task["skill"],
            "selected_by": "evaluation",
            "solution_code": code,
            "passed": test_result["passed"],
            "passed_tests": test_result["passed_tests"],
            "num_tests": test_result["num_tests"],
            "error_type": test_result.get("error_type"),
            "memory_snapshot": memory[:],
        }
        append_log(entry)
        results.append(entry)

        status = "PASS" if test_result["passed"] else f"FAIL({test_result['error_type']})"
        print_progress(
            f"  eval [{phase}] {task['id']} → {status} "
            f"({test_result['passed_tests']}/{test_result['num_tests']})"
        )
        time.sleep(0.2)

    return results


# ---------------------------------------------------------------------------
# One condition × seed run
# ---------------------------------------------------------------------------

def run_one(
    condition: str,
    seed: int,
    practice_tasks: list[dict],
    eval_tasks: list[dict],
    solve_template: str,
    select_template: str,
    reflection_template: str,
    num_practice: int = 12,
) -> dict:
    run_id = f"{condition}_seed_{seed}"
    print_progress(f"\n{'='*60}")
    print_progress(f"Starting run: {run_id}")
    print_progress(f"{'='*60}")

    rng = random.Random(seed)
    memory: list[str] = []
    skill_stats: dict[str, dict] = defaultdict(lambda: {"attempted": 0, "solved": 0, "errors": []})
    performance_history: list[dict] = []

    # --- Initial evaluation ---
    print_progress("Phase: initial evaluation")
    initial_results = evaluate_tasks(
        eval_tasks, memory, solve_template, condition, seed, run_id, "initial"
    )
    initial_acc = sum(r["passed"] for r in initial_results) / len(initial_results)
    print_progress(f"  Initial accuracy: {initial_acc:.3f}")

    # --- Practice loop ---
    remaining = practice_tasks[:num_practice]
    rng.shuffle(remaining)  # shuffle so fixed/metacognitive don't always start from index 0

    for iteration in range(min(num_practice, len(remaining))):
        if not remaining:
            break

        # --- Select task ---
        meta_info: dict | None = None
        if condition == "random":
            task, remaining = curriculum.choose_random(remaining, rng)
            selected_by = "random"
        elif condition == "fixed":
            task, remaining = curriculum.choose_fixed(remaining)
            selected_by = "fixed"
        else:  # metacognitive
            task, remaining, meta_info = curriculum.choose_metacognitive(
                remaining,
                performance_history,
                memory,
                skill_stats,
                select_template,
                rng,
            )
            selected_by = "llm_metacognition" if not (meta_info or {}).get("_fallback") else "random_fallback"

        # --- Solve ---
        try:
            code, _ = llm_client.solve_task(task, memory, solve_template)
            test_result = evaluator.run_tests(code, task)
        except Exception as exc:
            code = ""
            test_result = {
                "passed": False,
                "passed_tests": 0,
                "num_tests": len(task.get("tests", [])) or 1,
                "error_type": "llm_timeout",
                "feedback": f"LLM call failed: {exc}",
            }

        # --- Update stats ---
        skill_stats[task["skill"]]["attempted"] += 1
        if test_result["passed"]:
            skill_stats[task["skill"]]["solved"] += 1
        elif test_result.get("error_type"):
            skill_stats[task["skill"]]["errors"].append(test_result["error_type"])

        performance_history.append(
            {
                "task_id": task["id"],
                "skill": task["skill"],
                "passed": test_result["passed"],
                "iteration": iteration,
            }
        )

        # --- Reflect on failure ---
        reflection: dict | None = None
        if not test_result["passed"]:
            try:
                reflection, _ = llm_client.reflect(
                    task, code, test_result["feedback"], reflection_template
                )
            except Exception as exc:
                print_progress(f"  reflection skipped (LLM error: {exc})")
                reflection = None
            if reflection and reflection.get("general_rule_for_future"):
                rule = f"[{task['skill']}] {reflection['general_rule_for_future']}"
                memory.append(rule)
                if len(memory) > 12:
                    memory = memory[-12:]

        # --- Log ---
        log_entry: dict = {
            "run_id": run_id,
            "condition": condition,
            "seed": seed,
            "phase": "practice",
            "iteration": iteration,
            "task_id": task["id"],
            "skill": task["skill"],
            "selected_by": selected_by,
            "solution_code": code,
            "passed": test_result["passed"],
            "passed_tests": test_result["passed_tests"],
            "num_tests": test_result["num_tests"],
            "error_type": test_result.get("error_type"),
            "reflection": reflection,
            "memory_after": memory[:],
        }
        if meta_info:
            log_entry["chosen_task_id"] = meta_info.get("chosen_task_id")
            log_entry["estimated_current_competence"] = meta_info.get("estimated_current_competence")
            log_entry["expected_learning_gain"] = meta_info.get("expected_learning_gain")
            log_entry["selection_reason"] = meta_info.get("reason")

        append_log(log_entry)

        status = "PASS" if test_result["passed"] else f"FAIL({test_result['error_type']})"
        print_progress(
            f"  [{iteration+1:2d}/{num_practice}] {condition} | {task['id']} "
            f"({task['skill']}) [{selected_by}] → {status}"
        )
        time.sleep(0.2)

    # --- Final evaluation ---
    print_progress("Phase: final evaluation")
    final_results = evaluate_tasks(
        eval_tasks, memory, solve_template, condition, seed, run_id, "final"
    )
    final_acc = sum(r["passed"] for r in final_results) / len(final_results)
    print_progress(f"  Final accuracy: {final_acc:.3f}")
    print_progress(f"  Learning gain: {final_acc - initial_acc:+.3f}")

    # --- Skill-wise accuracies ---
    initial_skill = metrics.skill_accuracy(initial_results)
    final_skill = metrics.skill_accuracy(final_results)
    skill_gain = {
        s: round(final_skill.get(s, 0.0) - initial_skill.get(s, 0.0), 4)
        for s in set(initial_skill) | set(final_skill)
    }

    # --- Extra metrics for metacognitive ---
    practice_logs = [e for e in metrics.load_logs(LOGS_PATH)
                     if e.get("run_id") == run_id and e.get("phase") == "practice"]
    cal_error = metrics.calibration_error(practice_logs) if condition == "metacognitive" else float("nan")
    tsq = metrics.task_selection_quality(practice_logs) if condition == "metacognitive" else float("nan")

    practice_acc = (
        sum(p["passed"] for p in performance_history) / len(performance_history)
        if performance_history else 0.0
    )

    return {
        "run_id": run_id,
        "condition": condition,
        "seed": seed,
        "initial_accuracy": round(initial_acc, 4),
        "final_accuracy": round(final_acc, 4),
        "learning_gain": round(final_acc - initial_acc, 4),
        "practice_accuracy": round(practice_acc, 4),
        "calibration_error": cal_error,
        "task_selection_quality": tsq,
        "initial_skill_accuracy": initial_skill,
        "final_skill_accuracy": final_skill,
        "skill_gain": skill_gain,
        "memory_final": memory,
        "num_practice_tasks": len(performance_history),
    }


# ---------------------------------------------------------------------------
# Analysis / plotting from saved logs
# ---------------------------------------------------------------------------

def run_analysis():
    import pandas as pd

    logs = metrics.load_logs(LOGS_PATH)
    if not logs:
        print("No logs found. Run the experiment first.")
        return

    results_csv = os.path.join(OUTPUTS_DIR, "results.csv")
    if not os.path.exists(results_csv):
        print("No results.csv found. Run the experiment first.")
        return

    df = pd.read_csv(results_csv)
    print("\nAggregate results:")
    print(metrics.aggregate_summary(df).to_string(index=False))

    metrics.plot_accuracy_by_condition(df)
    metrics.plot_learning_gain_by_condition(df)

    # Skill gain: build per-condition skill gain dicts
    skill_results: dict[str, dict[str, float]] = defaultdict(dict)
    for _, row in df.iterrows():
        for col in df.columns:
            if col.startswith("skill_gain_"):
                skill = col.replace("skill_gain_", "")
                skill_results[row["condition"]][skill] = row[col]

    metrics.plot_skill_gain(skill_results)
    metrics.plot_calibration(logs)
    print("\nAnalysis complete. Plots saved to outputs/plots/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run metacognitive LLM experiment")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["random", "fixed", "metacognitive", "all"],
        default=["all"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--num-practice",
        type=int,
        default=7,
        help="Number of practice tasks per run (max 36)",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=7,
        help="Number of eval tasks per run; selected round-robin by skill for balance",
    )
    parser.add_argument("--analyze", action="store_true", help="Only run analysis on existing logs")
    args = parser.parse_args()

    if args.analyze:
        run_analysis()
        return

    conditions = (
        ["random", "fixed", "metacognitive"]
        if "all" in args.conditions
        else args.conditions
    )

    practice_tasks: list[dict] = load_json(os.path.join(DATA_DIR, "tasks_practice_bigcodebench.json"))
    all_eval_tasks: list[dict] = load_json(os.path.join(DATA_DIR, "tasks_eval_bigcodebench.json"))

    # Round-robin by skill so every skill is represented before any repeats
    _by_skill: dict[str, list] = defaultdict(list)
    for t in all_eval_tasks:
        _by_skill[t["skill"]].append(t)
    eval_tasks: list[dict] = []
    _skills_sorted = sorted(_by_skill)
    _i = 0
    while len(eval_tasks) < args.num_eval:
        added = False
        for skill in _skills_sorted:
            if _i < len(_by_skill[skill]) and len(eval_tasks) < args.num_eval:
                eval_tasks.append(_by_skill[skill][_i])
                added = True
        _i += 1
        if not added:
            break

    solve_template = load_prompt("solve_prompt.txt")
    select_template = load_prompt("select_task_prompt.txt")
    reflection_template = load_prompt("reflection_prompt.txt")

    print(f"Conditions: {conditions}")
    print(f"Seeds: {args.seeds}")
    print(f"Practice tasks per run: {args.num_practice}")
    print(f"Eval tasks: {len(eval_tasks)} (of {len(all_eval_tasks)} available)")

    import pandas as pd

    all_results = []

    for condition in conditions:
        for seed in args.seeds:
            result = run_one(
                condition=condition,
                seed=seed,
                practice_tasks=practice_tasks,
                eval_tasks=eval_tasks,
                solve_template=solve_template,
                select_template=select_template,
                reflection_template=reflection_template,
                num_practice=args.num_practice,
            )
            all_results.append(result)

            # Save after every completed run, merging with any existing results
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            _rows = []
            for r in all_results:
                _row = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                for skill, gain in r.get("skill_gain", {}).items():
                    _row[f"skill_gain_{skill}"] = gain
                _rows.append(_row)
            new_df = pd.DataFrame(_rows)
            if os.path.exists(RESULTS_PATH):
                existing = pd.read_csv(RESULTS_PATH)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["run_id"], keep="last")
            else:
                combined = new_df
            combined.to_csv(RESULTS_PATH, index=False)

    print(f"\nResults saved to {RESULTS_PATH}")

    # Load full results (includes prior runs from other conditions/seeds)
    df = pd.read_csv(RESULTS_PATH)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    agg = metrics.aggregate_summary(df)
    print(agg.to_string(index=False))

    # Generate plots
    metrics.plot_accuracy_by_condition(df)
    metrics.plot_learning_gain_by_condition(df)

    skill_results: dict[str, dict[str, float]] = defaultdict(dict)
    for _, row in df.iterrows():
        for col in df.columns:
            if col.startswith("skill_gain_"):
                skill = col.replace("skill_gain_", "")
                skill_results[row["condition"]][skill] = row[col]

    metrics.plot_skill_gain(skill_results)

    logs = metrics.load_logs(LOGS_PATH)
    metrics.plot_calibration(logs)

    print("\nDone.")


if __name__ == "__main__":
    main()
