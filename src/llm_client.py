"""LLM API wrapper for the metacognitive experiment.

Backend: Yandex AI Studio via OpenAI-compatible HTTP API.
Model:   gpt://b1gabcde1234/gpt-oss-20b  (override with EXPERIMENT_MODEL)

Environment variables:
  YANDEX_API_KEY    required (static API key)
  EXPERIMENT_MODEL  optional model URI override
"""
import concurrent.futures
import json
import os
import re
import time

import openai

MODEL = os.environ.get("EXPERIMENT_MODEL", "gpt://b1gabcde1234/gpt-oss-20b")

MAX_RETRIES = 3
RETRY_WAIT_BASE = 2
LLM_TIMEOUT_SECONDS = 120

_client = None  # lazily initialised SDK client

_BASE_URL = "https://ai.api.cloud.yandex.net/v1"


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def _get_sdk_client():
    global _client
    if _client is not None:
        return _client

    from openai import OpenAI
    api_key = os.environ.get("YANDEX_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "YANDEX_API_KEY is not set. "
            "Export it before running: export YANDEX_API_KEY='your-key'"
        )
    _client = OpenAI(api_key=api_key, base_url=_BASE_URL, timeout=LLM_TIMEOUT_SECONDS)
    return _client


# ---------------------------------------------------------------------------
# Backend-specific call implementations
# ---------------------------------------------------------------------------

def _extract_from_reasoning(reasoning: str) -> str | None:
    """Try to pull a Python code block out of a reasoning_content string.

    Reasoning models (gpt-oss-20b, o1-style) sometimes write the final
    solution inside the chain-of-thought when finish_reason='stop'.
    Returns the last ```python ... ``` block, or the last def-statement
    block found, or None if nothing usable is present.
    """
    if not reasoning:
        return None
    # Prefer the last explicit ```python block
    blocks = re.findall(r"```python\s*(.*?)```", reasoning, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    # Fall back to the last bare def block (everything from last 'def ' onward)
    idx = reasoning.rfind("\ndef ")
    if idx == -1:
        idx = reasoning.rfind("def ")
    if idx != -1:
        return reasoning[idx:].strip()
    return None


def _call_openai_compat(client, system: str, user: str, max_tokens: int,
                        _reasoning_retry: bool = False) -> str:
    """Shared path for gpt-oss-20b and Yandex (both OpenAI-compatible).

    Reasoning models (gpt-oss-20b, o1-style) write chain-of-thought into
    reasoning_content before writing the actual answer into content.
    If the token budget is exhausted during reasoning (finish_reason='length'),
    we automatically retry once with a 4× budget so the model can finish
    writing its answer after thinking.  Non-reasoning models are unaffected.
    """
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )

    choice  = response.choices[0]
    message = choice.message
    content = message.content
    finish  = getattr(choice, "finish_reason", None)

    # Happy path
    if content:
        return str(content).strip()

    reasoning = getattr(message, "reasoning_content", None)

    # Reasoning model hit the token limit mid-think — retry once with scaled budget
    if finish == "length" and reasoning and not _reasoning_retry:
        scaled = max(max_tokens * 4, 2048)
        return _call_openai_compat(client, system, user, scaled, _reasoning_retry=True)

    # content is None but reasoning finished — extract answer from chain-of-thought
    if reasoning and finish == "stop":
        extracted = _extract_from_reasoning(reasoning)
        if extracted:
            return extracted

    # Produce a clear error with diagnostics
    try:
        dumped = response.model_dump_json(indent=2)
    except Exception:
        dumped = str(response)

    hint = "Check the full response below for clues."
    if finish == "length":
        hint = f"Token budget ({max_tokens}) exhausted even after scaling. Check model output."

    raise RuntimeError(
        f"Model returned empty content (finish_reason={finish}). {hint}\n"
        f"Full response:\n{dumped}"
    )


# ---------------------------------------------------------------------------
# Public call entry point
# ---------------------------------------------------------------------------

def _call(system: str, user: str, max_tokens: int = 1024) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            client = _get_sdk_client()

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = pool.submit(_call_openai_compat, client, system, user, max_tokens)
            try:
                result = future.result(timeout=LLM_TIMEOUT_SECONDS)
                pool.shutdown(wait=False)
                return result
            except concurrent.futures.TimeoutError:
                pool.shutdown(wait=False)  # don't block waiting for the hung thread
                raise RuntimeError(
                    f"LLM call timed out after {LLM_TIMEOUT_SECONDS}s — marking as failed"
                )

        except openai.APITimeoutError as exc:
            raise RuntimeError(
                f"LLM call timed out after {LLM_TIMEOUT_SECONDS}s — marking as failed"
            ) from exc

        except RuntimeError:
            raise  # timeout already formatted, propagate immediately

        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT_BASE ** attempt)
            else:
                raise RuntimeError(
                    f"LLM call failed after {MAX_RETRIES} attempts: {exc}"
                ) from exc

    raise RuntimeError("LLM call failed after all retries")


def extract_code(text: str | None) -> str:
    """Extract Python function code from LLM response."""
    if text is None:
        raise ValueError("extract_code received None — model returned no content.")
    text = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no code block, return as-is (might already be clean code)
    return text


def extract_json(text: str) -> dict | None:
    """Extract a JSON object from text, handling markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try stripping markdown fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Find first complete { ... } block
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def solve_task(
    task: dict, memory: list[str], solve_template: str
) -> tuple[str, str]:
    """Ask the model to solve a task. Returns (code, raw_response)."""
    memory_text = "\n".join(memory) if memory else "(none)"
    user_msg = solve_template.format(
        memory=memory_text,
        task_prompt=task["prompt"],
        signature=task["signature"],
    )
    code_context = task.get("code_context", "").strip()
    if code_context:
        user_msg += f"\n\nAvailable imports (already in scope — do not re-import at the top level):\n{code_context}"
    system_msg = (
        "You are an expert Python programmer. "
        "Return ONLY the Python function code, nothing else."
    )
    # Use a generous token budget: reasoning models (e.g. gpt-oss-20b) spend
    # ~1000-2500 tokens on internal chain-of-thought before writing the code.
    # 8192 ensures the model always has room to finish after reasoning.
    raw = _call(system_msg, user_msg, max_tokens=8192)
    code = extract_code(raw)
    return code, raw


def select_task(
    remaining_tasks: list[dict],
    performance_history: list[dict],
    memory: list[str],
    skill_stats: dict,
    select_template: str,
) -> tuple[dict | None, str]:
    """Ask the model to select the next practice task (metacognitive mode).

    Returns (parsed_json_or_None, raw_response).
    """
    perf_lines = []
    for skill, stats in sorted(skill_stats.items()):
        attempted = stats["attempted"]
        solved = stats["solved"]
        rate = f"{solved / attempted:.2f}" if attempted > 0 else "N/A"
        errors = stats.get("errors", [])
        recent = ", ".join(errors[-3:]) if errors else "none"
        perf_lines.append(
            f"- {skill}: {solved}/{attempted} solved ({rate}), recent errors: {recent}"
        )
    perf_text = "\n".join(perf_lines) if perf_lines else "No attempts yet."

    memory_text = "\n".join(memory) if memory else "(none)"

    task_lines = []
    for t in remaining_tasks:
        preview = t["prompt"][:80].replace("\n", " ")
        task_lines.append(f"- {t['id']} [{t['skill']}, {t['difficulty']}]: {preview}...")
    task_list_text = "\n".join(task_lines)

    user_msg = select_template.format(
        performance_history=perf_text,
        memory=memory_text,
        task_list=task_list_text,
    )
    system_msg = (
        "You are a metacognitive learning agent. "
        "Analyse the skill statistics and select the most beneficial next task. "
        "Return ONLY valid JSON."
    )
    raw = _call(system_msg, user_msg, max_tokens=4096)
    return extract_json(raw), raw


def reflect(
    task: dict,
    solution_code: str,
    test_feedback: str,
    reflection_template: str,
) -> tuple[dict | None, str]:
    """Ask the model to reflect on a failed attempt. Returns (parsed_json_or_None, raw_response)."""
    user_msg = reflection_template.format(
        task_prompt=task["prompt"],
        solution_code=solution_code,
        test_feedback=test_feedback,
    )
    system_msg = (
        "You are reflecting on a programming mistake. "
        "Be concise and extract a general, reusable rule. "
        "Return ONLY valid JSON."
    )
    raw = _call(system_msg, user_msg, max_tokens=4096)
    return extract_json(raw), raw
