"""Stage 01b pure-logic — table-spec parsing + redact_text guards.

The presidio/spaCy engine path is covered (and skipped when absent) by
test_pii_redaction.py. Here we pin the orchestration bits that need no engine:
_tables config parsing and redact_text's None/blank short-circuits.
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE01B = os.path.join(ROOT, "01b_pii_redaction")
sys.path.insert(0, STAGE01B)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"ph01b_{name}", os.path.join(STAGE01B, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ph01b_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


rd = _load("redact")
# pipeline imports `redact` by bare name — alias it so the import resolves.
sys.modules["redact"] = rd
pl = _load("pipeline")


# --- _tables ------------------------------------------------------------------

def test_tables_list_form():
    pc = {"tables": [{"input_table": "a", "output_table": "b", "text_columns": ["x"]}]}
    assert pl._tables(pc) == pc["tables"]


def test_tables_single_triple_form():
    pc = {"input_table": "a", "output_table": "b", "text_columns": ["x", "y"]}
    assert pl._tables(pc) == [{"input_table": "a", "output_table": "b", "text_columns": ["x", "y"]}]


def test_tables_raises_when_incomplete():
    with pytest.raises(ValueError, match="pii_redaction needs"):
        pl._tables({"input_table": "a"})            # no output_table


# --- redact_text guards (no engine needed for None/blank) ---------------------

def test_redact_text_passes_none_through():
    assert rd.redact_text(None, None, None, entities=["PERSON"]) is None


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_redact_text_passes_blank_through(blank):
    # blank text short-circuits before the analyzer is touched
    assert rd.redact_text(None, None, blank, entities=["PERSON"]) == blank


def test_redact_text_default_threshold_is_safe():
    """Default must stay < 0.4 or Presidio silently drops phone numbers (score 0.4)."""
    import inspect
    default = inspect.signature(rd.redact_text).parameters["score_threshold"].default
    assert default < 0.4
