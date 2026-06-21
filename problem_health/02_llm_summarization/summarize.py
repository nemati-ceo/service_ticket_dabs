"""
summarize.py — LLM text normalization via Databricks ai_query, done Spark-native
and incrementally (NO duplicate summarization).

Per entity (problem / incident):
  1. ensure the output Delta table exists
  2. build a source view with summary_input_hash = md5(text)
  3. LEFT ANTI JOIN against the output on (key, hash) -> only NEW/CHANGED rows
  4. run ai_query() ONLY on those rows (unchanged rows are never re-billed)
  5. NO_CONTENT / error -> fall back to the original text
  6. MERGE upsert into the output table (keyed by id -> never duplicates)
  7. optionally DELETE summaries whose key no longer exists in the source
"""

# Prompts copied verbatim from the original PH02 script.
PROBLEM_PROMPT = (
    "You are a ServiceNow text normalizer for Northwestern Mutual Technology Customer Success team. "
    "Rewrite the following problem record into a clean two to three sentence technical description "
    "optimized for semantic matching against incident tickets. "
    "* Remove all names, dates, ticket numbers, URLs, and email addresses. "
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
    "* If text is empty or unintelligible, respond with exactly: NO_CONTENT. "
    "Incident: "
)


def _ai_query_result_expr(model, prompt_prefix, text_col, fail_on_error):
    """SQL expression returning the LLM text (unwrapping the failOnError struct)."""
    prefix = prompt_prefix.replace("'", "''")          # escape quotes for SQL literal
    prompt = f"CONCAT('{prefix}', COALESCE({text_col}, ''))"
    if fail_on_error:
        return f"ai_query('{model}', {prompt})"                       # returns STRING
    # failOnError => false returns STRUCT<result, errorMessage>; take .result
    return f"ai_query('{model}', {prompt}, failOnError => false).result"


def summarize_entity(spark, *, entity, model, source_sql, key_col, text_col,
                     summary_col, prompt_prefix, out_table,
                     fail_on_error=False, drop_deleted=True):
    """Run incremental LLM summarization for one entity. Returns (changed, total)."""
    # 1. output table (first run -> empty, so the anti-join treats all rows as new)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
            {key_col} STRING,
            {summary_col} STRING,
            summary_input_hash STRING,
            model_name STRING,
            summarized_at TIMESTAMP
        ) USING DELTA
    """)

    # 2. source + content hash
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW {entity}_src AS
        SELECT CAST({key_col} AS STRING)        AS {key_col},
               {text_col}                       AS input_text,
               md5(COALESCE({text_col}, ''))    AS summary_input_hash
        FROM ( {source_sql} )
    """)
    total = spark.table(f"{entity}_src").count()

    # 3. only NEW or CHANGED rows (key+hash not already summarized)
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW {entity}_changed AS
        SELECT s.* FROM {entity}_src s
        LEFT ANTI JOIN {out_table} o
          ON s.{key_col} = o.{key_col} AND s.summary_input_hash = o.summary_input_hash
    """)
    changed = spark.table(f"{entity}_changed").count()
    print(f"[ph02:{entity}] {changed}/{total} new or changed -> LLM "
          f"({total - changed} reused, no LLM call)")

    if changed == 0:
        if drop_deleted:
            _drop_deleted(spark, entity, out_table, key_col)
        return changed, total

    # 4-5. ai_query ONLY on changed rows; NO_CONTENT/null -> original text
    result_expr = _ai_query_result_expr(model, prompt_prefix, "input_text", fail_on_error)
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW {entity}_new AS
        SELECT {key_col},
               summary_input_hash,
               CASE WHEN raw_result IS NULL OR raw_result = 'NO_CONTENT'
                    THEN input_text ELSE raw_result END AS {summary_col},
               '{model}'           AS model_name,
               current_timestamp() AS summarized_at
        FROM (
            SELECT {key_col}, summary_input_hash, input_text,
                   {result_expr} AS raw_result
            FROM {entity}_changed
        )
    """)

    # 6. upsert (keyed by id -> no duplicate rows)
    spark.sql(f"""
        MERGE INTO {out_table} t
        USING {entity}_new s
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

    # 7. deletes
    if drop_deleted:
        _drop_deleted(spark, entity, out_table, key_col)

    return changed, total


def _drop_deleted(spark, entity, out_table, key_col):
    """Remove summaries whose key no longer exists in the current source."""
    spark.sql(f"""
        DELETE FROM {out_table}
        WHERE {key_col} NOT IN (SELECT {key_col} FROM {entity}_src)
    """)
