"""
incremental.py — detect which incident records are NEW or CHANGED,
so cleaning/scoring only runs on those (never re-process unchanged old data).

Strategy (best practice, hybrid):
  1. KEY (number)        -> detect brand-new records
  2. TIMESTAMP           -> detect changed records when update col is reliable
  3. CONTENT HASH        -> fallback / safety net when timestamps are missing
                            or unreliable (Nancy: update_date/create_date later)

Deletes are handled separately in the pipeline (rows in target but not in source).
"""

import hashlib
import pandas as pd


# ---------------------------------------------------------------------------
# Content hash — fingerprint of the fields that matter for scoring.
# If any of these change, the record must be re-scored even with no timestamp.
# ---------------------------------------------------------------------------
def compute_row_hash(row, hash_cols):
    """Stable MD5 of selected columns for one row."""
    parts = [str(row.get(c, "")) for c in hash_cols]
    joined = "||".join(parts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def add_content_hash(df, hash_cols, hash_col_name="content_hash"):
    """Add a content_hash column computed from hash_cols. Vectorized."""
    concat = df[hash_cols].fillna("").astype(str).agg("||".join, axis=1)
    df[hash_col_name] = concat.map(
        lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()
    )
    return df


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------
def identify_changes(
    df_new,
    df_existing,
    key_col,
    hash_cols,
    update_col=None,
    hash_col_name="content_hash",
):
    """
    Returns (df_to_process, df_unchanged).

    df_to_process : new + changed rows (must be cleaned/scored)
    df_unchanged  : old rows whose scores can be reused as-is

    Detection priority per row:
      - key not in existing            -> NEW
      - content hash differs           -> CHANGED  (catches edits even w/o timestamp)
      - update_col newer than existing  -> CHANGED  (when update_col provided & reliable)
    """
    # always add hash to incoming data
    df_new = add_content_hash(df_new.copy(), hash_cols, hash_col_name)

    # first run / nothing scored yet -> everything is new
    if df_existing is None or len(df_existing) == 0:
        print(f"[incremental] no existing data -> processing all {len(df_new)} rows")
        return df_new, pd.DataFrame()

    # make sure existing has a hash to compare against
    if hash_col_name not in df_existing.columns:
        df_existing = add_content_hash(df_existing.copy(), hash_cols, hash_col_name)

    new_keys = df_new[key_col].astype(str)
    existing_keys = set(df_existing[key_col].astype(str))

    # map existing key -> existing hash (for change comparison)
    existing_hash = (
        df_existing.assign(_k=df_existing[key_col].astype(str))
        .set_index("_k")[hash_col_name]
        .to_dict()
    )

    is_new = ~new_keys.isin(existing_keys)

    incoming_hash = df_new[hash_col_name]
    prior_hash = new_keys.map(existing_hash)          # NaN where new
    is_changed_hash = (~is_new) & (incoming_hash != prior_hash)

    # optional timestamp signal (only if column exists on both sides)
    is_changed_ts = pd.Series(False, index=df_new.index)
    if update_col and update_col in df_new.columns and update_col in df_existing.columns:
        existing_ts = (
            df_existing.assign(_k=df_existing[key_col].astype(str))
            .set_index("_k")[update_col]
            .to_dict()
        )
        prior_ts = pd.to_datetime(new_keys.map(existing_ts), errors="coerce")
        new_ts = pd.to_datetime(df_new[update_col], errors="coerce")
        is_changed_ts = (~is_new) & prior_ts.notna() & (new_ts > prior_ts)

    to_process_mask = is_new | is_changed_hash | is_changed_ts
    df_to_process = df_new[to_process_mask].copy()

    # unchanged incoming rows -> reuse their existing scores
    unchanged_keys = set(df_new[~to_process_mask][key_col].astype(str))
    df_unchanged = df_existing[
        df_existing[key_col].astype(str).isin(unchanged_keys)
    ].copy()

    print(
        f"[incremental] new={int(is_new.sum())} "
        f"changed_hash={int(is_changed_hash.sum())} "
        f"changed_ts={int(is_changed_ts.sum())} "
        f"-> process={len(df_to_process)} reuse={len(df_unchanged)}"
    )
    return df_to_process, df_unchanged


# ---------------------------------------------------------------------------
# Deletes — rows scored before but no longer in source
# ---------------------------------------------------------------------------
def find_deleted_keys(df_new, df_existing, key_col):
    if df_existing is None or len(df_existing) == 0:
        return set()
    current = set(df_new[key_col].astype(str))
    previous = set(df_existing[key_col].astype(str))
    deleted = previous - current
    if deleted:
        print(f"[incremental] {len(deleted)} deleted keys to drop from scores")
    return deleted
