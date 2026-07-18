"""Stage 01 preprocessing — characterization tests.

preprocessing.py is transcribed cleaning logic. Before/after a refactor its OUTPUT must
not move, so these tests pin the exact output of every live function on representative
inputs. nltk is stubbed (deterministic tokenizer + stopwords) so the module imports and
clean_text is reproducible off-cluster.
"""

import importlib.util
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE01 = os.path.join(ROOT, "01_problem_health")
sys.path.insert(0, STAGE01)

# Deterministic nltk stub: word_tokenize = whitespace split, a fixed stopword set.
_nltk = types.ModuleType("nltk")
_nltk.data = types.SimpleNamespace(find=lambda p: None)
_nltk.download = lambda *a, **k: True
sys.modules["nltk"] = _nltk
sys.modules["nltk.tokenize"] = types.SimpleNamespace(word_tokenize=lambda t: t.split())
sys.modules["nltk.corpus"] = types.SimpleNamespace(
    stopwords=types.SimpleNamespace(words=lambda lang: ["the", "a", "is", "to"]))

spec = importlib.util.spec_from_file_location("pp", os.path.join(STAGE01, "preprocessing.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)


# --- golden outputs captured from the original transcribed code ---------------

@pytest.mark.parametrize("text, expected", [
    ("Service Request Description: user cannot login please help mailto:x@nm.com ABCD1234 xyz45 abc1234",
     " : user cannot login   :   "),
    ("Client ID \n secret line\nGroup ID: 555\nhttps://nm-help.zendesk.com/foo?a=1 Ticket steps",
     "\n\n  steps"),
    ("What do you need help with? Impacted User Name John Smith Escalating T2 the issue",
     ""),
])
def test_clean_description_text(text, expected):
    assert pp.clean_description_text(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Phone Call Escalation Urgent CLR HUB the issue GApple abc12/Foo Termination",
     "         "),
    ("Bomgar Chat Candidate Portal problem incident SONB",
     "    "),
])
def test_clean_shortDescription_text(text, expected):
    assert pp.clean_shortDescription_text(text) == expected


def test_remove_general_problem_text():
    src = "Who is impacted What is the impact Workaround https://nm-help.zendesk.com/agent/users/99 Business"
    assert pp.removeGeneralProblemText(src) == "    "


def test_remove_url():
    src = ("see https://nm-help.zendesk.com/attachments/token/abc123 and "
           "https://nm-help.zendesk.com/agent/users/5 end")
    assert pp.removeURL(src) == "see  and  end"


@pytest.mark.parametrize("text, expected", [
    ("The User Is Unable To Login!!!", "user unable login"),
    ("Reset, the password. A quick FIX to it", "reset password quick fix it"),
])
def test_clean_text(text, expected):
    assert pp.clean_text(text) == expected


# --- composed trio (shared by pandas + spark cleaners) ------------------------

def test_composed_trio_exists_and_coerces_to_str():
    for fn in (pp.clean_inc_short, pp.clean_inc_desc, pp.clean_prob):
        assert callable(fn)
        assert fn(None) == fn("None")          # None -> str("None"), no crash
