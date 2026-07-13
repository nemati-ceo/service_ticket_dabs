"""summarize.py — LLM text normalization via Databricks ai_query, done Spark-native"""

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
    """Summarize one entity. Returns (changed, total, fallbacks).

    Rows whose text is already summarized under the SAME prompt+model are skipped —
    no LLM call, no re-billing. `fallbacks` counts rows the LLM returned NO_CONTENT
    or null for, which keep their original text.
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

    # The cache key covers the TEXT, the PROMPT and the MODEL — not just the text.
    # Hashing text alone means editing a prompt (or switching model) leaves every
    # existing row matching its old hash, so the anti-join below skips it and the
    # stale summary lives forever. The fix silently does nothing. Including the
    # prompt+model makes a prompt edit re-summarize exactly the rows it affects.
    fingerprint = (prompt_prefix + "||" + model).replace("'", "''")

    # ONE ROW PER KEY, enforced here rather than trusted from the caller. The source's
    # natural key is (number, problem_id), so an incident on N problems arrives N times.
    # Duplicates would bill the LLM N times, insert N summary rows, and then break the
    # MERGE below with DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW on the next run.
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
    changed = spark.table(f"{entity}_changed").count()
    print(f"[ph02:{entity}] {changed}/{total} new or changed -> LLM "
          f"({total - changed} reused, no LLM call)")

    if changed == 0:
        if drop_deleted:
            _drop_deleted(spark, entity, out_table, key_col)
        return changed, total, 0

    result_expr = _ai_query_result_expr(model, prompt_prefix, "input_text", fail_on_error)

    # MATERIALIZE the LLM output to a staging table — do NOT leave it as a view.
    # A view over ai_query() is lazy and re-executes on every action, so counting the
    # fallbacks and then MERGEing would call the LLM TWICE and bill twice. Writing it
    # to a table once means exactly one ai_query pass per run.
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

    # How many rows the LLM refused or failed on, and so kept their ORIGINAL text.
    # A spike here means summaries are silently degrading to raw ticket text.
    fallbacks = int(spark.sql(
        f"SELECT COALESCE(SUM(used_fallback), 0) FROM {staging}").collect()[0][0])
    if fallbacks:
        print(f"[ph02:{entity}] {fallbacks}/{changed} returned NO_CONTENT/null "
              f"-> fell back to original text")

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

    return changed, total, fallbacks


def _drop_deleted(spark, entity, out_table, key_col):
    """Remove summaries whose key no longer exists in the current source.

    Uses MERGE ... WHEN NOT MATCHED BY SOURCE rather than
    `DELETE ... WHERE key NOT IN (SELECT ...)`: Delta does not support subqueries in a
    DELETE condition (DELTA_UNSUPPORTED_SUBQUERY). Databricks' runtime tolerates it,
    open-source Delta does not — so the DELETE form cannot run or be tested anywhere
    but a Databricks cluster.
    """
    spark.sql(f"""
        MERGE INTO {out_table} t
        USING (SELECT DISTINCT {key_col} FROM {entity}_src) s
        ON t.{key_col} = s.{key_col}
        WHEN NOT MATCHED BY SOURCE THEN DELETE
    """)
