"""
preprocessing.py — Nancy's text cleaning (optimized: set + precompiled regex).
REPLACE the function bodies / lists with Nancy's exact preprocessing.py content.
Only clean_text is shown optimized; the others keep your original logic.
"""
import re, string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

_STOP = set(stopwords.words('english'))
_PUNCT_RE = re.compile(f'[{re.escape(string.punctuation)}]')

def clean_text(text):
    text = str(text).lower()
    text = _PUNCT_RE.sub('', text)
    tokens = word_tokenize(text)
    return ' '.join(w for w in tokens if w not in _STOP)

def clean_description_text(text):
    # <<< PASTE Nancy's exact body
    return str(text)

def clean_shortDescription_text(text):
    # <<< PASTE Nancy's exact body
    return str(text)

def clean_close_notes(text):
    # <<< PASTE Nancy's exact body
    return str(text)

def removeURL(input_string):
    # <<< PASTE Nancy's exact body
    return input_string

def removeGeneralProblemText(input_string):
    # <<< PASTE Nancy's exact body
    return removeURL(str(input_string))
