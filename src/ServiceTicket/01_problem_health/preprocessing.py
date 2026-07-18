"""preprocessing.py — text cleaning (transcribed from source, refactored).

The literal remove-lists and regexes are hoisted to module constants; the repeated
loop bodies are three small helpers (_replace_all / _sub_prefixed / _strip_leading).
nltk data is loaded LAZILY (first clean_text call), so importing this module has no
network side effect — matters in the redzone where downloads can hang.
"""
import re
import string

# --- literal remove-lists (declared once) ------------------------------------

DESC_URL_BASES = ["https://nm-help.zendesk.com", "https://help.northwesternmutual.com"]

DESC_NEXTLINE_TEMPLATE = ["Client ID", "Group ID", "Scenario ID", "Composite ID"]

DESC_REMOVE_TEMPLATE = ["What do you need help with?", "What is this request related to?:",
                        "Impacted User Name", "Impacted User Preferred Mode of Contact",
                        "Service Request #", "Ticket created from Slack by", "Caller/Chatter",
                        "WinSignID", "Client/Prospect Name", "Group ID", "Scenario ID",
                        "Composite ID", "Rep's Name", "Working as", "Client ID"]

DESC_REMOVE_TEXT = ["The caller has specified a unique Preferred Phone number for follow-ups; please access this phone number in Zendesk and utilize for any callbacks",
                    "What type of issue", "Comment", "What do you need help with",
                    "Service Request", "Description", "issue", "problem", "incident", "please", "help", "the",
                    "mailto", "Issue/CallNotes:", "Resolution/Follow-Up:", "Escalating", "T1", "T2", "T3",
                    "Issue/Question (Please include screenshots):", "Ticket", "Steps you have taken to Troubleshoot already"]

SHORT_REMOVE_TEXT = ["Escalation", "EventHUBAPI", "AMP", "DYNATRACE", "issue", "problem", "incident", "please", "help", "the",
                     "Urgent", "CLR HUB", "CLR Hub", "Hub", "CLR", "Termination", "Terminations",
                     "Candidate Portal", "Term", "Contract Change", "SONB"]

SHORT_PREFIX_REMOVE = ["Phone Call", "Bomgar Chat", "Phone", "Bomgar"]

GENERAL_PROBLEM_REMOVE = ["Who is impacted", "What is the impact", "What Is The Impact", "How many are impacted",
                          "Expected behavior", "Expected Behavior", "Workaround", "Subject Matter Expert",
                          "Customer Success", "ESD", "Product", "Business"]

URL_GLOBS = ["https://nm-help.zendesk.com/attachments/token/", "https://nm-help.zendesk.com/agent/users/"]

# --- compiled regexes --------------------------------------------------------

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b')
FIELDID_RE = re.compile(r'\b[A-Za-z]{4}\d{4}\b')
HOID_RE = re.compile(r'\b[A-Za-z]{3}\d{2}\b')
TECHID_RE = re.compile(r'\b[A-Za-z]{3}\d{4}\b')
GA_RE = re.compile(r'\bGA\w*\b')
SHORT_ALNUM_RE = re.compile(r'\b[A-Za-z]{3}\d{2}[ /][A-Za-z]+\b')
PUNCT_RE = re.compile(f'[{re.escape(string.punctuation)}]')


# --- shared loop bodies ------------------------------------------------------

def _replace_all(text, phrases):
    """Delete each literal phrase (str.replace)."""
    for p in phrases:
        text = text.replace(p, "")
    return text


def _sub_prefixed(text, phrases, suffix, flags=0):
    """re.sub(escape(phrase) + suffix, '') for each phrase — used for URL globs and
    template lines whose tail (`/\\S*`, `.*$`, `\\s+\\n.*`) is stripped along with the phrase."""
    for p in phrases:
        text = re.sub(re.escape(p) + suffix, "", text, flags=flags)
    return text


def _strip_leading(text, prefixes):
    """Strip each prefix from the start, in order (sequential, matches the original)."""
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):]
    return text


# --- lazy nltk ---------------------------------------------------------------

_STOPWORDS = None


def _ensure_nltk():
    """Download nltk data on first use (not at import). Caches the stopword set."""
    global _STOPWORDS
    if _STOPWORDS is not None:
        return
    import nltk
    for _pkg, _path in (("stopwords", "corpora/stopwords"),
                        ("punkt", "tokenizers/punkt"),
                        ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(_path)
        except LookupError:
            nltk.download(_pkg, quiet=True)
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words('english'))


# --- public cleaners ---------------------------------------------------------

def clean_description_text(text):
    text = _sub_prefixed(text, DESC_URL_BASES, r"/\S*")
    text = _sub_prefixed(text, DESC_NEXTLINE_TEMPLATE, r"\s+\n.*")
    text = _sub_prefixed(text, DESC_REMOVE_TEMPLATE, r".*$", flags=re.MULTILINE)
    text = _replace_all(text, DESC_REMOVE_TEXT)
    text = EMAIL_RE.sub("", text)
    text = FIELDID_RE.sub("", text)
    text = HOID_RE.sub("", text)
    text = TECHID_RE.sub("", text)
    return text


def clean_shortDescription_text(text):
    text = _replace_all(text, SHORT_REMOVE_TEXT)
    text = _strip_leading(text, SHORT_PREFIX_REMOVE)
    text = GA_RE.sub("", text)
    text = SHORT_ALNUM_RE.sub(" ", text)
    return text


def clean_text(text):
    _ensure_nltk()
    from nltk.tokenize import word_tokenize
    text = str(text).lower()
    text = PUNCT_RE.sub("", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in _STOPWORDS]
    return ' '.join(tokens)


def removeURL(input_string):
    return _sub_prefixed(input_string, URL_GLOBS, r"\S*")


def removeGeneralProblemText(input_string):
    return removeURL(_replace_all(input_string, GENERAL_PROBLEM_REMOVE))


# --- composed cleaners shared by cleaning.py (pandas) and cleaning_spark.py (Spark udf).
# Defined ONCE here so the two engines cannot drift out of sync.

def clean_inc_short(s):
    return clean_text(clean_shortDescription_text(str(s)))


def clean_inc_desc(s):
    return clean_text(clean_description_text(str(s)))


def clean_prob(s):
    return removeGeneralProblemText(str(s))
