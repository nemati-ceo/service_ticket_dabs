"""redact.py — Presidio + spaCy PII redaction. Engine construction and text scrubbing.

The spaCy model is loaded from a Volume path, never by name: `spacy.load("en_core_web_lg")`
resolves against spacy.io, which the redzone firewalls — the call hangs rather than
failing fast. Stage the model once (`nlp.to_disk(path)`) and point model_path at it.
"""

import os


def _pattern_recognizer(entity, regex, score=0.9):
    """Register a regex-backed entity Presidio has no built-in for (USER_ID, STREET_ADDRESS)."""
    from presidio_analyzer import Pattern, PatternRecognizer
    pattern = Pattern(name=entity.lower(), regex=regex, score=score)
    return PatternRecognizer(supported_entity=entity, patterns=[pattern])


def build_engines(model_path, spacy_model, custom_recognizers, language="en"):
    """Return (analyzer, anonymizer) with the spaCy model loaded from the Volume.

    custom_recognizers: {ENTITY_NAME: regex} for entities Presidio does not ship.
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"spaCy model not found at {model_path}. Stage it into the Volume first "
            f"(spacy.load('{spacy_model}') then nlp.to_disk(...)); the redzone blocks "
            f"downloading it at runtime.")

    # model_name is the Volume PATH, not the package name — this is what keeps the
    # load offline.
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": language, "model_name": model_path}],
    })
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=[language])
    for entity, regex in (custom_recognizers or {}).items():
        registry.add_recognizer(_pattern_recognizer(entity, regex))

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry,
                              supported_languages=[language])
    return analyzer, AnonymizerEngine()


def redact_text(analyzer, anonymizer, text, entities, language="en", score_threshold=0.5):
    """Replace every detected PII span with an <ENTITY> placeholder. Irreversible."""
    if text is None:
        return None
    text = str(text)
    if not text.strip():
        return text

    results = analyzer.analyze(text=text, entities=entities, language=language,
                               score_threshold=score_threshold)
    if not results:
        return text
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def redact_series(texts, *, model_path, spacy_model, custom_recognizers, entities,
                  language="en", score_threshold=0.35):
    """Redact an iterable of strings. Engines are built ONCE per call (per Spark task)."""
    analyzer, anonymizer = build_engines(model_path, spacy_model, custom_recognizers, language)
    return [redact_text(analyzer, anonymizer, t, entities, language, score_threshold)
            for t in texts]
