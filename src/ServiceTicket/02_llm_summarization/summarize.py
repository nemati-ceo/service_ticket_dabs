"""summarize.py — LLM text normalization via Databricks ai_query, done Spark-native"""

import hashlib


def prompt_fingerprint(prompt_prefix, model):
    """Short stable hash of prompt+model — identifies which prompt version made a summary."""
    return hashlib.md5(f"{prompt_prefix}||{model}".encode()).hexdigest()[:12]


# ~4 characters per token for English technical text. ai_query() returns no usage struct
# and Spark SQL has no tokenizer, so this is the only count available inside the run.
# Treat it as a COST SIGNAL — "is this run 10x yesterday?" — not a billing figure. The
# authoritative per-request numbers live in `system.serving.endpoint_usage`.
_CHARS_PER_TOKEN = 4


def _est_tokens(chars):
    return int(round((chars or 0) / _CHARS_PER_TOKEN))

PROBLEM_PROMPT = (
    "You are a ServiceNow text normalizer for Northwestern Mutual Technology Customer Success team. "
    "Rewrite the following problem record into a clean two to three sentence technical description "
    "optimized for semantic matching against incident tickets. "
    "* Remove all names, dates, ticket numbers, URLs, and email addresses. "
    "* Focus on: what system or service is affected, and what the root cause or pattern is. "
    "* Use consistent technical language. "
    "* Do not start with This problem - state the issue directly. "
    "* Do not mention PII. "
    "* Do not add any header. "
    "* If text is empty or unintelligible, respond with exactly: NO_CONTENT. "
    "Problem: "
)

INCIDENT_PROMPT = (
    "You are a ServiceNow text normalizer for Northwestern Mutual Technology Customer Success team. "
    "Rewrite the following incident ticket into a clean two to three sentence technical description "
    "optimized for semantic matching against problem records. "
    "* Remove all names, dates, ticket numbers, URLs, and email addresses. "
    "* Focus on: what system or service is affected, what the symptom is, and what the root cause is if stated. "
    "* Use consistent technical language. "
    "* Do not start with This incident - state the issue directly. "
    "* Do not mention PII. "
    "* Do not add any header. "
    "* If text is empty or unintelligible, respond with exactly: NO_CONTENT. "
    "Incident: "
)


def _ai_query_result_expr(model, prompt_prefix, text_col, fail_on_error):
    """SQL expression returning the LLM text (unwrapping the failOnError struct)."""
    prefix = prompt_prefix.replace("'", "''")
    prompt = f"CONCAT('{prefix}', COALESCE({text_col}, ''))"
    if fail_on_error:
        return f"ai_query('{model}', {prompt})"
    return f"ai_query('{model}', {prompt}, failOnError => false).result"


def summarize_entity(spark, *, entity, model, source_sql, key_col, text_col,
                     summary_col, prompt_prefix, out_table,
                     fail_on_error=False, drop_deleted=True):
    """Summarize one entity. Returns (changed, total, fallbacks, tokens).

    Rows whose text is already summarized under the SAME prompt+model are skipped —
    no LLM call, no re-billing. `fallbacks` counts rows the LLM returned NO_CONTENT
    or null for, which keep their original text.

    `tokens` is {"input", "output", "total"}, ESTIMATED (see _est_tokens). Only rows
    actually sent to the LLM this run are counted, so a fully cached run reports zero —
    which is the point: the metric tracks spend, not corpus size.
    """
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
            {key_col} STRING,
            {summary_col} STRING,
            summary_input_hash STRING,
            model_name STRING,
            summarized_at TIMESTAMP
        ) USING DELTA
    """)

    # Cache key = text + prompt + model, so a prompt/model edit re-summarizes the rows it affects.
    fingerprint = (prompt_prefix + "||" + model).replace("'", "''")

    # ONE ROW PER KEY (_rn = 1): the source key is (number, problem_id), so an incident on
    # N problems arrives N times — duplicates would bill the LLM N times and break the MERGE.
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW {entity}_src AS
        SELECT {key_col}, input_text, summary_input_hash
        FROM (
            SELECT CAST({key_col} AS STRING)        AS {key_col},
                   {text_col}                       AS input_text,
                   md5(CONCAT(COALESCE({text_col}, ''), '||', '{fingerprint}'))
                                                    AS summary_input_hash,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST({key_col} AS STRING)
                       ORDER BY {text_col} DESC NULLS LAST) AS _rn
            FROM ( {source_sql} )
            WHERE {key_col} IS NOT NULL
        )
        WHERE _rn = 1
    """)
    total = spark.table(f"{entity}_src").count()

    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW {entity}_changed AS
        SELECT s.* FROM {entity}_src s
        LEFT ANTI JOIN {out_table} o
          ON s.{key_col} = o.{key_col} AND s.summary_input_hash = o.summary_input_hash
    """)
    # Count and measure in ONE pass — the anti-join must be evaluated before the MERGE
    # below makes it empty, and scanning it twice doubles the work for no reason.
    _row = spark.sql(f"""
        SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(input_text)), 0) AS chars
        FROM {entity}_changed
    """).collect()[0]
    changed, in_chars = int(_row["n"]), int(_row["chars"])
    tokens = {"input": 0, "output": 0, "total": 0}
    print(f"[ph02:{entity}] {changed}/{total} new or changed -> LLM "
          f"({total - changed} reused, no LLM call)")

    if changed == 0:
        if drop_deleted:
            _drop_deleted(spark, entity, out_table, key_col)
        return changed, total, 0, tokens

    result_expr = _ai_query_result_expr(model, prompt_prefix, "input_text", fail_on_error)

    # MATERIALIZE to a staging table — a lazy view over ai_query() re-executes on every
    # action, so counting fallbacks then MERGEing would call (and bill) the LLM twice.
    staging = f"{out_table}_staging"
    spark.sql(f"DROP TABLE IF EXISTS {staging}")
    spark.sql(f"""
        CREATE TABLE {staging} USING DELTA AS
        SELECT {key_col},
               summary_input_hash,
               CASE WHEN raw_result IS NULL OR raw_result = 'NO_CONTENT'
                    THEN input_text ELSE raw_result END AS {summary_col},
               CASE WHEN raw_result IS NULL OR raw_result = 'NO_CONTENT'
                    THEN 1 ELSE 0 END              AS used_fallback,
               '{model}'           AS model_name,
               current_timestamp() AS summarized_at
        FROM (
            SELECT {key_col}, summary_input_hash, input_text,
                   {result_expr} AS raw_result
            FROM {entity}_changed
        )
    """)

    # Rows the LLM refused/failed on, which kept their ORIGINAL text. A spike here means
    # summaries are silently degrading to raw ticket text.
    # Output chars EXCLUDE fallback rows: their text was copied from the input, not
    # generated, and counting it would inflate the output-token estimate on exactly the
    # runs where the LLM produced the least.
    _agg = spark.sql(f"""
        SELECT COALESCE(SUM(used_fallback), 0) AS fallbacks,
               COALESCE(SUM(CASE WHEN used_fallback = 0
                                 THEN LENGTH({summary_col}) ELSE 0 END), 0) AS out_chars
        FROM {staging}
    """).collect()[0]
    fallbacks, out_chars = int(_agg["fallbacks"]), int(_agg["out_chars"])
    if fallbacks:
        print(f"[ph02:{entity}] {fallbacks}/{changed} returned NO_CONTENT/null "
              f"-> fell back to original text")

    # The prompt prefix is prepended to every row, so it is billed `changed` times.
    tokens["input"] = _est_tokens(len(prompt_prefix) * changed + in_chars)
    tokens["output"] = _est_tokens(out_chars)
    tokens["total"] = tokens["input"] + tokens["output"]
    print(f"[ph02:{entity}] est. tokens: in={tokens['input']:,} out={tokens['output']:,} "
          f"total={tokens['total']:,}  (~{_CHARS_PER_TOKEN} chars/token, estimate)")

    spark.sql(f"""
        MERGE INTO {out_table} t
        USING (SELECT {key_col}, {summary_col}, summary_input_hash, model_name, summarized_at
               FROM {staging}) s
        ON t.{key_col} = s.{key_col}
        WHEN MATCHED THEN UPDATE SET
            t.{summary_col}       = s.{summary_col},
            t.summary_input_hash  = s.summary_input_hash,
            t.model_name          = s.model_name,
            t.summarized_at       = s.summarized_at
        WHEN NOT MATCHED THEN INSERT
            ({key_col}, {summary_col}, summary_input_hash, model_name, summarized_at)
            VALUES (s.{key_col}, s.{summary_col}, s.summary_input_hash, s.model_name, s.summarized_at)
    """)
    print(f"[ph02:{entity}] upserted {changed} summaries -> {out_table}")
    spark.sql(f"DROP TABLE IF EXISTS {staging}")

    if drop_deleted:
        _drop_deleted(spark, entity, out_table, key_col)

    return changed, total, fallbacks, tokens


def _drop_deleted(spark, entity, out_table, key_col):
    """Remove summaries whose key no longer exists in the current source.
    Uses MERGE ... NOT MATCHED BY SOURCE: Delta rejects subqueries in a DELETE condition."""
    spark.sql(f"""
        MERGE INTO {out_table} t
        USING (SELECT DISTINCT {key_col} FROM {entity}_src) s
        ON t.{key_col} = s.{key_col}
        WHEN NOT MATCHED BY SOURCE THEN DELETE
    """)
