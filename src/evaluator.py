"""Safe code execution and test evaluation via subprocess."""
import json
import os
import subprocess
import tempfile

TIMEOUT_SECONDS = 10

# ---------------------------------------------------------------------------
# BigCodeBench harness (unittest.TestCase style, task_func in global scope)
# ---------------------------------------------------------------------------

_BCB_HARNESS_TEMPLATE = """\
import json as _json
import io as _io
import unittest

# Imports required by the solution
CODE_CONTEXT_PLACEHOLDER

# Model solution
SOLUTION_PLACEHOLDER

# BigCodeBench test class (references task_func directly from module scope)
TEST_CODE_PLACEHOLDER

_suite = unittest.TestLoader().loadTestsFromTestCase(TestCases)
_stream = _io.StringIO()
_runner = unittest.TextTestRunner(verbosity=0, stream=_stream)
_result = _runner.run(_suite)

_n = _suite.countTestCases()
_failed_items = _result.failures + _result.errors
_err_msg = None
for _t, _trace in _failed_items[:1]:
    _err_msg = _trace[-400:]

_etype = None
if _result.failures:
    _etype = "wrong_answer"
elif _result.errors:
    _etype = "runtime_error"

print(_json.dumps({
    "passed": bool(_result.wasSuccessful()),
    "passed_tests": _n - len(_failed_items),
    "num_tests": _n,
    "error_type": _etype,
    "error": _err_msg,
}))
"""

# ---------------------------------------------------------------------------
# Original structured harness (input/expected dict tests)
# ---------------------------------------------------------------------------

_HARNESS_TEMPLATE = """\
import json
import copy

SOLUTION_PLACEHOLDER

def _compare(got, expected):
    if isinstance(expected, bool):
        return type(got) == type(expected) and got == expected
    try:
        return json.dumps(got, sort_keys=True) == json.dumps(expected, sort_keys=True)
    except (TypeError, ValueError):
        return got == expected

def main():
    tests = TEST_CASES_PLACEHOLDER
    results = []
    for i, test in enumerate(tests):
        try:
            inp = copy.deepcopy(test['input'])
            expected = test['expected']
            got = solve(**inp)
            passed = _compare(got, expected)
            results.append({
                'index': i,
                'passed': bool(passed),
                'got': repr(got),
                'expected': repr(expected),
            })
        except Exception as e:
            results.append({
                'index': i,
                'passed': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'expected': repr(test.get('expected')),
            })
    print(json.dumps(results))

main()
"""


def _build_harness(solution_code: str, tests: list) -> str:
    harness = _HARNESS_TEMPLATE.replace("SOLUTION_PLACEHOLDER", solution_code)
    harness = harness.replace("TEST_CASES_PLACEHOLDER", repr(tests))
    return harness


def _build_bcb_harness(solution_code: str, task: dict) -> str:
    harness = _BCB_HARNESS_TEMPLATE.replace("CODE_CONTEXT_PLACEHOLDER", task.get("code_context", ""))
    harness = harness.replace("SOLUTION_PLACEHOLDER", solution_code)
    harness = harness.replace("TEST_CODE_PLACEHOLDER", task["test_code"])
    return harness


def _run_subprocess(harness: str, num_tests: int) -> dict:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(harness)
            tmp_path = f.name

        proc = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )

        if proc.returncode != 0 and not proc.stdout.strip():
            error_msg = (proc.stderr or "unknown error").strip()[:400]
            error_type = "syntax_error" if "SyntaxError" in error_msg else "runtime_error"
            return _error_result(num_tests, error_type, f"Error running code:\n{error_msg}")

        return json.loads(proc.stdout.strip())

    except subprocess.TimeoutExpired:
        return _error_result(num_tests, "timeout", "Solution exceeded time limit.")

    except json.JSONDecodeError as exc:
        return _error_result(num_tests, "parse_error", f"Could not parse test output: {exc}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _bcb_result_to_standard(raw: dict) -> dict:
    num_tests = raw.get("num_tests", 1)
    passed_count = raw.get("passed_tests", 0)
    all_passed = raw.get("passed", False)
    error_type = raw.get("error_type")
    err_msg = raw.get("error", "")

    feedback_parts = []
    if not all_passed:
        feedback_parts.append(f"Passed {passed_count}/{num_tests} tests.")
        if err_msg:
            feedback_parts.append(err_msg[-300:])

    return {
        "passed": all_passed,
        "passed_tests": passed_count,
        "num_tests": num_tests,
        "error_type": error_type,
        "failed_test": None,
        "feedback": "\n".join(feedback_parts) if feedback_parts else "All tests passed.",
    }


def run_tests(solution_code: str, task: dict) -> dict:
    """Execute solution against task test cases in a subprocess.

    Returns:
        passed: bool — all tests passed
        passed_tests: int
        num_tests: int
        error_type: str | None
        failed_test: dict | None
        feedback: str — human-readable summary for the reflection prompt
    """
    if task.get("test_code"):
        return _run_bcb_tests(solution_code, task)
    return _run_structured_tests(solution_code, task)


def _run_bcb_tests(solution_code: str, task: dict) -> dict:
    """Run BigCodeBench-style unittest.TestCase tests."""
    if not solution_code or not solution_code.strip():
        return _error_result(1, "empty_solution", "No code was generated.")

    harness = _build_bcb_harness(solution_code, task)
    raw = _run_subprocess(harness, num_tests=1)
    # _run_subprocess returns an _error_result dict (has "feedback") on subprocess failure
    if "feedback" in raw:
        return raw
    return _bcb_result_to_standard(raw)


def _run_structured_tests(solution_code: str, task: dict) -> dict:
    """Run original structured {input, expected} tests."""
    tests = task["tests"]
    num_tests = len(tests)

    if not solution_code or not solution_code.strip():
        return _error_result(num_tests, "empty_solution", "No code was generated.")

    harness = _build_harness(solution_code, tests)
    raw = _run_subprocess(harness, num_tests)

    if not isinstance(raw, list):
        return raw  # error dict already populated by _run_subprocess

    test_results = raw
    passed_count = sum(1 for r in test_results if r.get("passed"))
    all_passed = passed_count == num_tests

    first_failure = next((r for r in test_results if not r.get("passed")), None)
    error_type = None
    feedback_parts = []

    if not all_passed and first_failure:
        if "error_type" in first_failure:
            error_type = first_failure["error_type"].lower()
            feedback_parts.append(
                f"Test {first_failure['index'] + 1} raised {first_failure['error_type']}: "
                f"{first_failure.get('error', '')}"
            )
        else:
            error_type = "wrong_answer"
            feedback_parts.append(
                f"Test {first_failure['index'] + 1} failed: "
                f"expected {first_failure['expected']}, got {first_failure['got']}"
            )
        feedback_parts.append(f"Passed {passed_count}/{num_tests} tests.")

    return {
        "passed": all_passed,
        "passed_tests": passed_count,
        "num_tests": num_tests,
        "error_type": error_type,
        "failed_test": first_failure,
        "feedback": "\n".join(feedback_parts) if feedback_parts else "All tests passed.",
    }


def _error_result(num_tests: int, error_type: str, feedback: str) -> dict:
    return {
        "passed": False,
        "passed_tests": 0,
        "num_tests": num_tests,
        "error_type": error_type,
        "failed_test": None,
        "feedback": feedback,
    }
