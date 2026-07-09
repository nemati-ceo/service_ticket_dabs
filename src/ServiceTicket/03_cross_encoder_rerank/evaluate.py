"""evaluate.py — Top-K retrieval accuracy for the reranked candidates, vs baselines."""

import numpy as np


def topk_accuracy(true_problem_ids, candidate_problem_ids, reranked_scores, k_values):
    """true_problem_ids      : (n_incidents,) gold problem_id per incident"""
    true_ids = np.asarray(true_problem_ids).astype(str)
    cand_ids = np.asarray(candidate_problem_ids).astype(str)
    order = np.argsort(-reranked_scores, axis=1)
    n = len(true_ids)
    rows = np.arange(n)[:, None]

    results = {}
    print("=== Reranked Top-K Accuracy (Rerank) ===")
    for k in k_values:
        kk = min(k, cand_ids.shape[1])
        topk_pids = cand_ids[rows, order[:, :kk]]
        hits = int((topk_pids == true_ids[:, None]).any(axis=1).sum())
        acc = hits / n if n else 0.0
        results[k] = acc
        print(f"Top-{k}: {hits}/{n} = {acc:.4f}")
    return results


def print_baselines(baselines):
    """Echo the prior-stage baselines for side-by-side comparison."""
    if not baselines:
        return
    print("\n=== Baselines ===")
    for name, scores in baselines.items():
        parts = ", ".join(f"Top-{k}={v}" for k, v in scores.items())
        print(f"{name}: {parts}")
