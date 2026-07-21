"""Stage 01b PII redaction tests.

The engine tests need presidio + a spaCy model and are skipped when either is absent
(CI/local without the redzone Volume). The config tests always run — they guard the
score-threshold trap, which is silent and leaks PII when wrong.
"""

import os
import re
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "01b_pii_redaction"))

CONFIG = yaml.safe_load(open(os.path.join(ROOT, "config.yml")))
PII = CONFIG["pii_redaction"]
REDACTED_TABLES = [t["output_table"] for t in PII["tables"]]
RAW_TABLES = [t["input_table"] for t in PII["tables"]]


# --- config guards (no presidio needed) ---------------------------------------

def test_threshold_below_phone_score():
    """Presidio scores PHONE_NUMBER at exactly 0.4.

    A threshold >= 0.4 drops every phone number while still redacting names and
    emails — the redaction looks like it works and silently leaks phone numbers.
    """
    assert PII["score_threshold"] < 0.4


def test_summarization_reads_redacted_table():
    """Stage 02 is the only stage that sends text off-cluster. It must never read raw."""
    assert CONFIG["summarization"]["input_table"] in REDACTED_TABLES


def test_clustering_gap_fill_reads_a_redacted_table():
    """Stage 05's gap-fill also calls ai_query() — it is a SECOND egress point.

    It summarizes the UNLINKED incidents (the `cluster` table). If it reads them raw,
    names/phones/close_notes go straight to the LLM even though stage 02 is clean.
    """
    sql = CONFIG["clustering"]["summarize_source_sql"]
    assert any(t in sql for t in REDACTED_TABLES), \
        "clustering gap-fill does not read a redacted table — raw PII would reach ai_query()"


def test_no_downstream_stage_reads_a_raw_table():
    """Only stage 01b may read the raw tables — they are the thing being redacted.

    Any other stage reading one republishes unredacted text: linking.py, for one, copies
    every column of its incidents frame straight into the production linking table.
    """
    downstream = ("summarization", "reranking", "gbm_inference", "clustering")
    for stage in downstream:
        blob = yaml.safe_dump(CONFIG[stage])
        for raw in RAW_TABLES:
            assert raw not in blob, f"{stage} reads the RAW table {raw} — PII would leak"


def test_every_llm_reachable_table_is_redacted():
    """The cluster table feeds ai_query via stage 05. It must be in the redaction list."""
    synced = [t["target"] for t in CONFIG["input_sync"]["tables"]]
    cluster_mirror = [t for t in synced if "cluster" in t]
    assert cluster_mirror, "the cluster table is not synced"
    assert any(c in RAW_TABLES for c in cluster_mirror), \
        "cluster_synced is synced but NOT redacted — stage 05 sends it to the LLM"


def test_every_custom_entity_has_a_recognizer():
    """An entity listed but not built-in and not in custom_recognizers is never detected."""
    builtin = {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION"}
    custom = set(PII.get("custom_recognizers") or {})
    for entity in PII["entities"]:
        assert entity in builtin or entity in custom, f"{entity} has no recognizer"


@pytest.mark.parametrize("text,expected", [
    ("abc1234", True),
    ("ABC9876", True),
    ("ab123", False),
    ("abcd1234", False),
    ("1234567", False),
])
def test_user_id_regex(text, expected):
    assert bool(re.search(PII["custom_recognizers"]["USER_ID"], text)) is expected


@pytest.mark.parametrize("text,expected", [
    ("720 E Wisconsin Ave", True),
    ("1 Main Street", True),
    ("450 N Sunnyslope Rd Suite 300", True),
    ("no address here", False),
])
def test_street_address_regex(text, expected):
    assert bool(re.search(PII["custom_recognizers"]["STREET_ADDRESS"], text)) is expected


# --- engine tests (need presidio + a spaCy model) ------------------------------
#
# Skipped per-test, NOT via a module-level importorskip: that would also skip the
# config guards above, and those are the ones that must never stop running.

import importlib.util

HAS_PRESIDIO = all(importlib.util.find_spec(m) is not None
                   for m in ("presidio_analyzer", "presidio_anonymizer", "spacy"))

needs_presidio = pytest.mark.skipif(
    not HAS_PRESIDIO, reason="presidio/spacy not installed")


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    """Stage a small spaCy model to disk and load it BY PATH — same flow as the Volume."""
    import spacy

    import redact as rd
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    dest = tmp_path_factory.mktemp("volume") / "en_core_web_sm"
    os.makedirs(dest, exist_ok=True)
    nlp.to_disk(dest)
    return rd.build_engines(str(dest), "en_core_web_sm", PII["custom_recognizers"])


def _redact(engines, text):
    import redact as rd
    analyzer, anonymizer = engines
    return rd.redact_text(analyzer, anonymizer, text, PII["entities"],
                          score_threshold=PII["score_threshold"])


@needs_presidio
@pytest.mark.parametrize("pii", [
    "nancy@example.com",
    "312-555-1234",
    "abc1234",
])
def test_pii_never_survives(engines, pii):
    """The literal PII string must not appear in the output."""
    assert pii not in _redact(engines, f"Ticket from {pii} about a login failure.")


@needs_presidio
def test_clean_text_untouched(engines):
    text = "Payment gateway timeout after deploy. Restarted the pod."
    assert _redact(engines, text) == text


@needs_presidio
def test_null_and_empty_survive(engines):
    assert _redact(engines, None) is None
    assert _redact(engines, "") == ""


def test_missing_model_raises_loudly():
    """A missing Volume model must fail fast, not fall back to a firewalled download.

    Deliberately NOT gated on presidio: the path check runs before the presidio import, so
    a config error stays a config error even where that import chain is broken. CI hit
    exactly that — presidio's import died on `nltk.__spec__ is None` and buried the
    FileNotFoundError this test asserts.
    """
    import redact as rd
    with pytest.raises(FileNotFoundError):
        rd.build_engines("/no/such/volume/path", "en_core_web_lg",
                         PII["custom_recognizers"])
