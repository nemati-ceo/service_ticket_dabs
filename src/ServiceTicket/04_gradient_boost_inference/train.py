"""train.py — stage 04 TRAIN branch: fit the GradientBoostingClassifier and save it.

Runs when config `mode: train`. Reuses the SAME feature matrix builder as inference
(features.build_feature_matrix), so the columns the model is fitted on cannot drift
from the columns it is later scored on.

The label is already in the feature matrix: a candidate row is positive when the
candidate problem is the incident's gold problem_id.

The holdout split is BY INCIDENT (GroupShuffleSplit on `number`), never by row. Each
incident contributes ~top_k candidate rows; a row-wise split would place candidates of
the same incident on both sides and leak, which inflates top-k accuracy.
"""

import os
import time
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import evaluate as ev
from features import FEATURE_COLS


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def split_by_incident(feature_df, group_col, test_size, random_state):
    """Group-wise train/test split. No incident appears on both sides."""
    groups = feature_df[group_col].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_state)
    train_idx, test_idx = next(splitter.split(feature_df, groups=groups))
    train = feature_df.iloc[train_idx].copy()
    test = feature_df.iloc[test_idx].copy()
    print(f"[ph04:train] split by {group_col}: "
          f"train {len(train)} rows / {train[group_col].nunique()} incidents | "
          f"test {len(test)} rows / {test[group_col].nunique()} incidents")

    overlap = set(train[group_col].astype(str)) & set(test[group_col].astype(str))
    if overlap:
        raise RuntimeError(
            f"group split leaked: {len(overlap)} incident(s) in both train and test")
    return train, test


def fit(train_df, params):
    """Fit the GBM on FEATURE_COLS -> label."""
    X, y = train_df[FEATURE_COLS], train_df["label"].astype(int)
    pos = int(y.sum())
    print(f"[ph04:train] fitting on {len(X)} rows | positives={pos} "
          f"({pos / len(X) * 100:.2f}%) | features={FEATURE_COLS}")
    if pos == 0:
        raise ValueError(
            "no positive labels in the training set — every candidate_problem_id "
            "differs from the gold problem_id. Check the stage-03 candidates.")

    model = GradientBoostingClassifier(**params)
    model.fit(X, y)
    return model


def _score_split(model, df, number_col, k_values, prefix):
    """AUCs + incident-level top-k for one split."""
    X, y = df[FEATURE_COLS], df["label"].astype(int)
    proba = model.predict_proba(X)[:, 1]

    metrics = {}
    if y.nunique() > 1:
        metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(y, proba))
        metrics[f"{prefix}_pr_auc"] = float(average_precision_score(y, proba))
    else:
        print(f"[ph04:train] {prefix} split has a single class — skipping AUC")

    scored = df.copy()
    scored["gbm_propensity"] = proba
    ranked = ev.rank_candidates(scored, number_col=number_col, problem_id_col="problem_id")
    try:
        topk = ev.topk_accuracy(ranked, k_values, number_col=number_col)
        for k, v in topk.items():
            metrics[f"{prefix}_top{k}_accuracy"] = float(v)
    except Exception as e:
        print(f"[ph04:train] {prefix} top-k eval skipped ({e})")
        topk = None
    return metrics, topk


def evaluate(model, train_df, test_df, number_col, k_values):
    """Score BOTH splits and print them side by side.

    Test numbers alone cannot show overfitting. 200 trees on 3 features with a ~2%
    positive rate can memorize the training incidents; a large train-vs-test gap is the
    only signal that has happened, so both are reported.
    """
    train_metrics, train_topk = _score_split(model, train_df, number_col, k_values, "train")
    test_metrics, test_topk = _score_split(model, test_df, number_col, k_values, "test")

    header = "  ".join(f"Top-{k}".rjust(8) for k in k_values)
    print(f"\n[ph04:train] {'Set'.ljust(7)}{header}")
    for name, topk in (("Train", train_topk), ("Test", test_topk)):
        if topk:
            row = "  ".join(f"{topk[k]:8.4f}" for k in k_values)
            print(f"[ph04:train] {name.ljust(7)}{row}")

    if train_topk and test_topk:
        k0 = k_values[0]
        gap = train_topk[k0] - test_topk[k0]
        print(f"[ph04:train] train-test gap @Top-{k0}: {gap:+.4f}"
              + ("   <-- large gap: the model is memorizing incidents" if gap > 0.15 else ""))

    metrics = {**train_metrics, **test_metrics}
    for key in ("train_roc_auc", "test_roc_auc", "train_pr_auc", "test_pr_auc"):
        if key in metrics:
            print(f"[ph04:train]   {key} = {metrics[key]:.4f}")
    return metrics, test_topk


def save_model(model, model_dir, model_name):
    """Write the .pkl to the Volume. Production reads exactly this path."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, model_name)
    joblib.dump(model, path)
    print(f"[ph04:train] model saved -> {path}")
    return path


def run_gbm_train(spark, cfg, feature_df):
    """Fit, evaluate, persist. Writes NO production linking table.

    feature_df comes from the shared loader in pipeline.py, so train and production
    see byte-identical features.
    """
    tc = cfg["gbm_train"]
    gc = cfg["gbm_inference"]
    number_col = gc.get("number_col", "number")
    group_col = tc.get("group_col", number_col)

    t0 = time.perf_counter()
    print(f"[ph04:train] started {_ts()} | TRAIN MODE — production table untouched")

    labeled = feature_df[feature_df["problem_id"].notna()].copy()
    dropped = len(feature_df) - len(labeled)
    if dropped:
        print(f"[ph04:train] dropped {dropped} row(s) with no gold problem_id")
    if labeled.empty:
        raise ValueError("no labeled rows — cannot train")

    train_df, test_df = split_by_incident(
        labeled, group_col, tc.get("test_size", 0.2), tc.get("random_state", 42))

    params = dict(tc.get("params") or {})
    params.setdefault("random_state", tc.get("random_state", 42))
    model = fit(train_df, params)

    metrics, topk = evaluate(model, train_df, test_df, number_col,
                             (tc.get("eval") or {}).get("k_values", [1, 5, 7, 10]))

    path = save_model(model, tc["model_dir"], tc.get("model_name",
                                                     "PH04_gradient_boosting_model.pkl"))
    total = time.perf_counter() - t0

    mu = _mlflow_utils()
    with mu.stage_run(cfg, "ph04_gbm_train") as ml:
        ml.log_params({**params, "group_col": group_col,
                       "test_size": tc.get("test_size", 0.2),
                       "features": ",".join(FEATURE_COLS)})
        ml.set_tags({"mode": "train", "model_path": path})
        ml.log_metrics({**metrics,
                        "train_rows": len(train_df),
                        "test_rows": len(test_df),
                        "train_incidents": int(train_df[group_col].nunique()),
                        "test_incidents": int(test_df[group_col].nunique()),
                        "train_positives": int(train_df["label"].sum()),
                        "wall_clock_s": total})
        if topk:
            ml.log_dict({str(k): v for k, v in topk.items()}, "holdout_topk_accuracy.json")
        try:
            import mlflow.sklearn
            mlflow.sklearn.log_model(model, "gbm")
        except Exception as e:
            print(f"[ph04:train] mlflow log_model skipped ({e})")

    feature_importance = dict(zip(FEATURE_COLS,
                                  np.round(model.feature_importances_, 4).tolist()))
    print("=" * 60)
    print("Stage 04 TRAIN complete!")
    print(f"  Model:              {path}")
    print(f"  Feature importance: {feature_importance}")
    print(f"  Total wall-clock:   {total:.2f}s  (finished {_ts()})")
    print("=" * 60)
    return path, metrics


def _mlflow_utils():
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "mlflow_utils", os.path.join(root, "mlflow_utils.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
