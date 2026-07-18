"""Stage 02 summarization — pure logic.

The Spark/LLM path needs a cluster. These pin the parts that are pure: the ai_query
SQL builder (incl. quote escaping), the prompt fingerprint, and the pipeline's metric
helpers.
"""

import importlib.util
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE02 = os.path.join(ROOT, "02_llm_summarization")
sys.path.insert(0, STAGE02)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"ph02_{name}", os.path.join(STAGE02, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ph02_{name}"] = mod
    sys.modules[name] = mod            # pipeline imports `summarize` / `evaluate` bare
    spec.loader.exec_module(mod)
    return mod


sm = _load("summarize")
_load("evaluate")
pl = _load("pipeline")


# --- prompt fingerprint -------------------------------------------------------

def test_fingerprint_is_stable_and_short():
    a = sm.prompt_fingerprint(sm.PROBLEM_PROMPT, "m1")
    assert a == sm.prompt_fingerprint(sm.PROBLEM_PROMPT, "m1")   # deterministic
    assert len(a) == 12


def test_fingerprint_changes_with_prompt_and_with_model():
    base = sm.prompt_fingerprint(sm.PROBLEM_PROMPT, "m1")
    assert sm.prompt_fingerprint(sm.INCIDENT_PROMPT, "m1") != base   # prompt edit
    assert sm.prompt_fingerprint(sm.PROBLEM_PROMPT, "m2") != base    # model swap


# --- ai_query SQL builder -----------------------------------------------------

def test_ai_query_uses_failonerror_false_by_default():
    expr = sm._ai_query_result_expr("mdl", "Prefix: ", "input_text", fail_on_error=False)
    assert "failOnError => false" in expr and expr.endswith(".result")


def test_ai_query_raises_mode_has_no_result_unwrap():
    expr = sm._ai_query_result_expr("mdl", "Prefix: ", "input_text", fail_on_error=True)
    assert "failOnError" not in expr and not expr.endswith(".result")


def test_ai_query_escapes_single_quotes_in_prompt():
    """An apostrophe in a prompt would otherwise terminate the SQL string literal."""
    expr = sm._ai_query_result_expr("mdl", "don't stop", "input_text", fail_on_error=False)
    assert "don''t stop" in expr
    assert "'don't" not in expr


def test_ai_query_coalesces_null_text():
    expr = sm._ai_query_result_expr("mdl", "P: ", "input_text", fail_on_error=False)
    assert "COALESCE(input_text, '')" in expr


# --- pipeline metric helpers --------------------------------------------------

@pytest.mark.parametrize("part, whole, expected", [
    (1, 4, 25.0),
    (0, 10, 0.0),
    (10, 10, 100.0),
    (1, 3, 33.33),        # rounded to 2dp
    (5, 0, 0.0),          # zero denominator must not raise
])
def test_pct(part, whole, expected):
    assert pl._pct(part, whole) == expected


def test_avg_len_returns_value():
    class _Spark:
        def sql(self, q):
            return types.SimpleNamespace(collect=lambda: [[42.5]])
    assert pl._avg_len(_Spark(), "t", "c") == 42.5


def test_avg_len_returns_zero_when_table_empty():
    class _Spark:
        def sql(self, q):
            return types.SimpleNamespace(collect=lambda: [[None]])   # AVG of no rows
    assert pl._avg_len(_Spark(), "t", "c") == 0.0


def test_avg_len_never_raises(capsys):
    class _Spark:
        def sql(self, q):
            raise RuntimeError("table missing")
    assert pl._avg_len(_Spark(), "t", "c") is None
    assert "skipped" in capsys.readouterr().out
