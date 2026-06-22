"""inference.py — load the trained GradientBoostingClassifier and score feature rows."""

import joblib
import numpy as np

from features import FEATURE_COLS


def load_model(model_path):
    """Load the pre-trained GBM (.pkl) from a Volume."""
    return joblib.load(model_path)


def score(model, feature_df, batch_size=500_000):
    """Add `gbm_propensity` = P(label=1) per row, scored in bounded batches."""
    X = feature_df[FEATURE_COLS]
    preds = [model.predict_proba(X.iloc[s:s + batch_size])[:, 1]
             for s in range(0, len(X), batch_size)]
    feature_df["gbm_propensity"] = np.concatenate(preds) if preds else np.array([])
    return feature_df
