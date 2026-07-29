"""model_cache.py — load a sentence-transformers model from a Volume, never re-download.

Root-level and shared: stage 03 (cross-encoder + bi-encoder) and stage 05 (clustering
embedder) all pull the same few hundred MB, and a per-run download is pure waste.
"""

import os


def _is_cached(volume_path):
    """True when volume_path is a directory with something in it."""
    return bool(volume_path) and os.path.isdir(volume_path) and bool(os.listdir(volume_path))


def load_cached(model_name, volume_path, cls, **kwargs):
    """Load from Volume if cached there; else download from HF and cache to Volume.

    A cached copy that no longer LOADS counts as a cache miss, not a fatal error: the
    folder encodes the sentence-transformers version that wrote it, so an upgrade can
    orphan it. Stage 01's loader already re-downloads on a bad Volume copy.
    """
    if _is_cached(volume_path):
        print(f"  loading {cls.__name__} from Volume: {volume_path}")
        try:
            return cls(volume_path, **kwargs)
        except Exception as e:
            print(f"  WARNING: the Volume copy at {volume_path} did not load ({e}); "
                  f"re-downloading {model_name} and replacing the cache.")
    model = cls(model_name, **kwargs)
    if volume_path:
        try:
            os.makedirs(volume_path, exist_ok=True)
            model.save(volume_path)
            print(f"  {cls.__name__} downloaded and cached to Volume: {volume_path}")
        except Exception as e:
            print(f"  WARNING: could not cache model to Volume ({e})")
    else:
        print(f"  WARNING: no volume_path for {model_name} — it re-downloads every run")
    return model
