"""inference.py — load the trained GradientBoostingClassifier and score feature rows."""

import os

import joblib
import numpy as np

from features import FEATURE_COLS


def load_model(model_path):
    """Load the pre-trained GBM (.pkl) from a Volume."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"GBM model not found at {model_path} — run stage 04 in mode: train first, "
            f"or point gbm_inference.model_path at an existing .pkl")
    return joblib.load(model_path)


def score(model, feature_df, batch_size=500_000):
    """Add `gbm_propensity` = P(label=1) per row, scored in bounded batches."""
    missing = [c for c in FEATURE_COLS if c not in feature_df.columns]
    if missing:
        raise ValueError(f"feature matrix is missing {missing}; has {list(feature_df.columns)}")
    # A model fitted on a different column set scores garbage silently — sklearn only
    # checks the count, not the names or the order.
    expected = getattr(model, "n_features_in_", len(FEATURE_COLS))
    if expected != len(FEATURE_COLS):
        raise ValueError(
            f"model expects {expected} features but FEATURE_COLS has {len(FEATURE_COLS)} "
            f"({FEATURE_COLS}) — the model and the feature builder are out of sync")
    X = feature_df[FEATURE_COLS]
    preds = [model.predict_proba(X.iloc[s:s + batch_size])[:, 1]
             for s in range(0, len(X), batch_size)]
    feature_df["gbm_propensity"] = np.concatenate(preds) if preds else np.array([])
    return feature_df
