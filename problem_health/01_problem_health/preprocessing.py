"""preprocessing.py — text cleaning (transcribed from source)."""
import re
import string
import nltk

for _pkg, _path in [("stopwords", "corpora/stopwords"),
                    ("punkt", "tokenizers/punkt"),
                    ("punkt_tab", "tokenizers/punkt_tab")]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

_STOPWORDS = set(stopwords.words('english'))


def clean_description_text(text):
    removeText = ["The caller has specified a unique Preferred Phone number for follow-ups; please access this phone number in Zendesk and utilize for any callbacks",
                  "What type of issue", "Comment", "What do you need help with",
                  "Service Request", "Description", "issue", "problem", "incident", "please", "help", "the",
                  "mailto", "Issue/CallNotes:", "Resolution/Follow-Up:", "Escalating", "T1", "T2", "T3",
                  "Issue/Question (Please include screenshots):", "Ticket", "Steps you have taken to Troubleshoot already"
                  ]


    removeTemplateText = ["What do you need help with?", "What is this request related to?:", "Impacted User Name",
                          "Impacted User Preferred Mode of Contact", "Service Request #", "Ticket created from Slack by",
                          "Caller/Chatter", "WinSignID", "Client/Prospect Name",
                          "Group ID", "Scenario ID", "Composite ID", "Rep's Name", "Working as",
                          "Client ID"]

    nextLineTemplate = ["Client ID", "Group ID", "Scenario ID", "Composite ID"]


    removeURLs = ["https://nm-help.zendesk.com", "https://help.northwesternmutual.com"]

    for url in removeURLs:
        escaped_base_url = re.escape(url)
        regex_pattern = rf"{escaped_base_url}/\S*"
        text = re.sub(regex_pattern, "", text)

    for nextLineText in nextLineTemplate:
        regex_pattern = re.escape(nextLineText) + r"\s+\n.*"
        text = re.sub(regex_pattern, "", text)

    for templateText in removeTemplateText:
        regex_pattern = re.escape(templateText) + r".*$"
        text = re.sub(regex_pattern, "", text, flags=re.MULTILINE)

    for remove_text in removeText:
        text = text.replace(remove_text, '')

    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
    text = email_pattern.sub('', text)

    fieldId_pattern = re.compile(r'\b[A-Za-z]{4}\d{4}\b')
    text = fieldId_pattern.sub('', text)

    HOId_pattern = re.compile(r'\b[A-Za-z]{3}\d{2}\b')
    text = HOId_pattern.sub('', text)

    techId_pattern = re.compile(r'\b[A-Za-z]{3}\d{4}\b')
    text = techId_pattern.sub('', text)

    return text


def clean_shortDescription_text(text):
    removeText = ["Escalation", "EventHUBAPI", "AMP", "DYNATRACE", "issue", "problem", "incident", "please", "help", "the",
                  "Urgent", "CLR HUB", "CLR Hub", "Hub", "CLR", "Termination", "Terminations",
                  "Candidate Portal", "Term", "Contract Change", "SONB"]
    prefixToRemove = ["Phone Call", "Bomgar Chat", "Phone", "Bomgar"]

    for textToRemove in removeText:
        text = text.replace(textToRemove, '')

    for startText in prefixToRemove:
        if text.startswith(startText):
            text = text[len(startText):]

    pattern = r'\bGA\w*\b'
    text = re.sub(pattern, '', text)

    pattern = re.compile(r'\b[A-Za-z]{3}\d{2}[ /][A-Za-z]+\b')
    text = pattern.sub(' ', text)

    return text


def clean_close_notes(text):
    text = str(text)
    removeTemplateText = ["Greetings"]

    removeText = ["Thank you for contacting the Planning Application Support Team!",
                  "Best Regards,", "Planning Application Technical Support Team",
                  "If you have any additional questions or still require assistance, please feel free to reach out via:",
                  "Live Chat with the Planning Application Support Team", "Click here to submit a ticket", "close", "closing",
                  "Close", "Closing"]

    for templateText in removeTemplateText:
        regex_pattern = re.escape(templateText) + r".*$"
        text = re.sub(regex_pattern, "", text, flags=re.MULTILINE)

    for remove_text in removeText:
        text = text.replace(remove_text, '')

    return text


def clean_text(text):
    text = str(text)
    text = text.lower()

    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in _STOPWORDS]
    return ' '.join(tokens)


def removeURL(input_string):
    urlToRemove = ["https://nm-help.zendesk.com/attachments/token/", "https://nm-help.zendesk.com/agent/users/"]
    cleanedText = input_string
    for url in urlToRemove:
        pattern = rf'{re.escape(url)}\S*'
        cleanedText = re.sub(pattern, '', cleanedText)

    return cleanedText


def removeGeneralProblemText(input_string):
    textToRemove = ["Who is impacted", "What is the impact", "What Is The Impact", "How many are impacted", "Expected behavior",
                    "Expected Behavior", "Workaround", "Subject Matter Expert", "Customer Success", "ESD", "Product", "Business"]
    cleanedText = input_string
    for removeText in textToRemove:
        cleanedText = cleanedText.replace(removeText, "")

    cleanedText = removeURL(cleanedText)
    return cleanedText
