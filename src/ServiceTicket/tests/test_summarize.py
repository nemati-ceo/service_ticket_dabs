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


# --- run.limit ----------------------------------------------------------------

def test_with_limit_is_a_no_op_when_unset():
    sql = "SELECT a FROM t GROUP BY a"
    assert pl._with_limit(sql, None) == sql
    assert pl._with_limit(sql, 0) == sql


def test_with_limit_wraps_so_union_and_group_by_survive():
    """A bare `... LIMIT n` would bind to the last UNION arm only."""
    out = pl._with_limit("SELECT a FROM t UNION ALL SELECT b FROM u", 100)
    assert out == "SELECT * FROM (SELECT a FROM t UNION ALL SELECT b FROM u) LIMIT 100"


def test_with_limit_coerces_to_int():
    """Guards against a string limit reaching the SQL string."""
    assert pl._with_limit("SELECT 1", "50").endswith("LIMIT 50")


# --- token estimate -----------------------------------------------------------

class _FakeSpark:
    """Answers the handful of queries summarize_entity runs, with canned numbers.

    Records every statement so a test can assert on the SQL that decides a number
    (the fallback exclusion) rather than only on the arithmetic.
    """

    def __init__(self, *, total, changed, in_chars, fallbacks, out_chars):
        self.stmts = []
        self._counts = {"src": total, "changed": changed}
        self._changed_row = {"n": changed, "chars": in_chars}
        self._staging_row = {"fallbacks": fallbacks, "out_chars": out_chars}

    def sql(self, q):
        self.stmts.append(q)
        if "FROM {}_changed".format("problem") in q or "_changed\n" in q:
            row = self._changed_row
        elif "used_fallback" in q and q.strip().upper().startswith("SELECT"):
            row = self._staging_row
        else:
            row = None
        return types.SimpleNamespace(collect=lambda: [row] if row else [[0]])

    def table(self, name):
        key = "src" if name.endswith("_src") else "changed"
        return types.SimpleNamespace(count=lambda: self._counts[key])


def _run(**kw):
    spark = _FakeSpark(**kw)
    out = sm.summarize_entity(
        spark, entity="problem", model="m", source_sql="SELECT 1",
        key_col="problem_id", text_col="txt", summary_col="problem_summary",
        prompt_prefix="P" * 40, out_table="t", drop_deleted=False)
    return spark, out


def test_est_tokens_rounds_on_chars_per_token():
    assert sm._est_tokens(0) == 0
    assert sm._est_tokens(400) == 100
    assert sm._est_tokens(None) == 0


def test_tokens_count_the_prompt_once_per_row_sent():
    """The prefix is prepended to every request, so it is billed `changed` times."""
    _, (changed, total, fallbacks, tok) = _run(
        total=10, changed=2, in_chars=800, fallbacks=0, out_chars=400)
    assert (changed, total, fallbacks) == (2, 10, 0)
    assert tok["input"] == (40 * 2 + 800) / 4      # prompt x2 + source text
    assert tok["output"] == 100
    assert tok["total"] == tok["input"] + tok["output"]


def test_fallback_rows_are_excluded_from_output_tokens():
    """Fallback text was copied from the input, not generated — counting it would
    inflate output tokens on exactly the runs where the LLM produced the least."""
    spark, _ = _run(total=5, changed=5, in_chars=400, fallbacks=5, out_chars=0)
    agg = [q for q in spark.stmts if "out_chars" in q][0]
    assert "used_fallback = 0" in agg


def test_fully_cached_run_reports_zero_spend():
    """Nothing sent to the LLM -> no tokens, even though the corpus is large."""
    _, (changed, total, fallbacks, tok) = _run(
        total=5000, changed=0, in_chars=0, fallbacks=0, out_chars=0)
    assert (changed, total) == (0, 5000)
    assert tok == {"input": 0, "output": 0, "total": 0}
