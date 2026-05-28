#!/usr/bin/env python3
"""Fetch BigCodeBench and produce tasks_practice_bigcodebench.json + tasks_eval_bigcodebench.json.

Run once from the project root after installing the datasets library:
    pip install datasets
    python data/fetch_bigcodebench.py
"""

import ast
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# All standard-library module names that are safe to run in subprocess
STDLIB_ALLOWED = {
    "re", "json", "csv", "datetime", "calendar", "time", "math", "statistics",
    "decimal", "fractions", "os", "pathlib", "shutil", "io", "glob", "string",
    "textwrap", "difflib", "unicodedata", "collections", "itertools", "functools",
    "operator", "copy", "random", "hashlib", "base64", "struct", "heapq",
    "bisect", "enum", "dataclasses", "typing", "abc", "contextlib", "unittest",
    "tempfile", "fnmatch", "zipfile", "tarfile", "pprint", "types",
    "ast", "inspect", "sys", "builtins",
}

# Map skill name → set of "primary" libraries that define that skill
SKILL_LIBS = {
    "regex":          {"re"},
    "json_proc":      {"json", "csv"},
    "file_ops":       {"os", "pathlib", "shutil", "tempfile", "glob", "zipfile", "tarfile"},
    "math_compute":   {"math", "statistics", "decimal", "fractions"},
    "text_transform": {"string", "textwrap", "difflib", "unicodedata"},
    "data_struct":    {"collections", "heapq", "bisect", "itertools", "functools"},
}

PRACTICE_PER_SKILL = 6
EVAL_PER_SKILL = 2


def get_libs(row) -> set:
    try:
        return set(ast.literal_eval(row["libs"]))
    except Exception:
        return set()


def classify_skill(libs: set) -> str | None:
    if not libs.issubset(STDLIB_ALLOWED):
        return None
    for skill, primary in SKILL_LIBS.items():
        if libs & primary:
            return skill
    return None


def test_uses_only_stdlib(test_code: str) -> bool:
    """Return False if test_code imports any non-stdlib package."""
    for m in re.finditer(r"^(?:import|from)\s+(\w+)", test_code, re.MULTILINE):
        mod = m.group(1)
        if mod not in STDLIB_ALLOWED and mod != "unittest":
            return False
    return True


def extract_imports(code_prompt: str) -> str:
    lines = code_prompt.strip().split("\n")
    def_idx = max((i for i, l in enumerate(lines) if l.startswith("def ")), default=-1)
    if def_idx <= 0:
        return ""
    return "\n".join(lines[:def_idx]).strip()


def extract_signature(code_prompt: str) -> str:
    lines = code_prompt.strip().split("\n")
    def_idx = max((i for i, l in enumerate(lines) if l.startswith("def ")), default=-1)
    return lines[def_idx].strip() if def_idx >= 0 else "def task_func():"


def count_test_methods(test_code: str) -> int:
    return len(re.findall(r"^\s+def test_", test_code, re.MULTILINE))


def assign_difficulty(n_methods: int) -> str:
    if n_methods <= 5:
        return "easy"
    elif n_methods <= 7:
        return "medium"
    return "hard"


def convert_row(row, skill: str, task_id: str) -> dict:
    code_context = extract_imports(row["code_prompt"])
    signature = extract_signature(row["code_prompt"])
    n_methods = count_test_methods(row["test"])
    return {
        "id": task_id,
        "skill": skill,
        "difficulty": assign_difficulty(n_methods),
        "prompt": row["instruct_prompt"].strip(),
        "entry_point": row["entry_point"],
        "signature": signature,
        "code_context": code_context,
        "test_code": row["test"],
        "tests": [],
    }


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print("Loading BigCodeBench v0.1.2 ...")
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2", trust_remote_code=False)
    print(f"Total tasks: {len(ds)}")

    buckets = defaultdict(list)
    for row in ds:
        libs = get_libs(row)
        skill = classify_skill(libs)
        if skill and test_uses_only_stdlib(row["test"]):
            buckets[skill].append(row)

    print("Stdlib-only tasks per skill:")
    for skill in sorted(buckets):
        print(f"  {skill}: {len(buckets[skill])}")

    rng = random.Random(42)
    practice_tasks = []
    eval_tasks = []

    for skill in sorted(SKILL_LIBS.keys()):
        rows = buckets[skill][:]
        rng.shuffle(rows)
        need = PRACTICE_PER_SKILL + EVAL_PER_SKILL
        if len(rows) < need:
            print(f"Warning: {skill} only has {len(rows)} tasks (need {need})")
        for j, row in enumerate(rows[:PRACTICE_PER_SKILL]):
            practice_tasks.append(convert_row(row, skill, f"bcb_{skill}_{j + 1:02d}"))
        for j, row in enumerate(rows[PRACTICE_PER_SKILL:PRACTICE_PER_SKILL + EVAL_PER_SKILL]):
            eval_tasks.append(convert_row(row, skill, f"bcb_{skill}_eval_{j + 1:02d}"))

    out = Path(__file__).parent
    p_file = out / "tasks_practice_bigcodebench.json"
    e_file = out / "tasks_eval_bigcodebench.json"

    p_file.write_text(json.dumps(practice_tasks, indent=2, ensure_ascii=False))
    e_file.write_text(json.dumps(eval_tasks, indent=2, ensure_ascii=False))

    from collections import Counter
    diff_counts = Counter(t["difficulty"] for t in practice_tasks)
    print(f"\nWrote {len(practice_tasks)} practice tasks → {p_file}")
    print(f"Wrote {len(eval_tasks)} eval tasks → {e_file}")
    print(f"Difficulty distribution (practice): {dict(diff_counts)}")


if __name__ == "__main__":
    main()
