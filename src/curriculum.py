"""Task selection strategies: random, fixed, and metacognitive."""
import random as _random

# Human-designed progression order for the fixed curriculum (simpler → compositional)
FIXED_SKILL_ORDER = [
    "regex",
    "text_transform",
    "json_proc",
    "math_compute",
    "data_struct",
    "file_ops",
]


def choose_random(remaining: list[dict], rng: _random.Random) -> tuple[dict, list[dict]]:
    """Select a random task from the remaining pool."""
    idx = rng.randrange(len(remaining))
    task = remaining[idx]
    new_remaining = remaining[:idx] + remaining[idx + 1 :]
    return task, new_remaining


def choose_fixed(remaining: list[dict]) -> tuple[dict, list[dict]]:
    """Select the next task following the predefined skill order.

    Within each skill group, tasks are taken in their natural ID order.
    """
    for skill in FIXED_SKILL_ORDER:
        for i, task in enumerate(remaining):
            if task["skill"] == skill:
                new_remaining = remaining[:i] + remaining[i + 1 :]
                return task, new_remaining
    # Fallback: take the first remaining task
    return remaining[0], remaining[1:]


def choose_metacognitive(
    remaining: list[dict],
    performance_history: list[dict],
    memory: list[str],
    skill_stats: dict,
    select_template: str,
    rng: _random.Random,
) -> tuple[dict, list[dict], dict | None]:
    """Ask the LLM to select the next task based on metacognitive self-assessment.

    Returns (task, new_remaining, meta_info).
    Falls back to choose_random if LLM response is invalid.
    """
    import llm_client  # local import to avoid circular dependency

    try:
        meta_json, _ = llm_client.select_task(
            remaining,
            performance_history,
            memory,
            skill_stats,
            select_template,
        )
    except Exception:
        meta_json = None

    chosen_id = meta_json.get("chosen_task_id") if meta_json else None

    if chosen_id:
        for i, task in enumerate(remaining):
            if task["id"] == chosen_id:
                new_remaining = remaining[:i] + remaining[i + 1 :]
                return task, new_remaining, meta_json

    # LLM returned an invalid or unrecognised task id — fall back to random
    task, new_remaining = choose_random(remaining, rng)
    if meta_json:
        meta_json["_fallback"] = True
    return task, new_remaining, meta_json
