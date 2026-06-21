"""
embeddings.py — load the sentence-transformer model and encode text.

- backend="onnx"  -> ~2x faster on CPU, ~99% identical embeddings (best for prod CPU)
- backend="torch" -> original behavior

Load order (first hit wins):
  1. VOLUME path  -> load local files, no network, no registry  (fastest, prod default)
  2. UC registry  -> models:/<name>/<latest>
  3. HuggingFace  -> download, then register + save to Volume for next time
"""

import os

from sentence_transformers import SentenceTransformer


def load_or_save_model(model_name, registry_name, backend="onnx", volume_path=None):
    """Load MiniLM from Volume -> registry -> HF download. Returns model."""
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_registry_uri("databricks-uc")

    # 0. VOLUME first — if the model files are already on the Volume, load them
    #    directly. No HF download, no registry round-trip.
    if volume_path and os.path.isdir(volume_path) and os.listdir(volume_path):
        try:
            print(f"  Loading model from Volume: {volume_path}")
            model = SentenceTransformer(volume_path, backend=backend)
            print("  Model loaded successfully from Volume.")
            return model
        except Exception as e:
            print(f"  Volume load failed ({e}); trying registry...")
    elif volume_path:
        print(f"  No model files at Volume path {volume_path}; trying registry...")

    # try registry — always resolve the LATEST version (never hardcode /1)
    try:
        client = MlflowClient(registry_uri="databricks-uc")
        versions = client.search_model_versions(f"name='{registry_name}'")
        if not versions:
            raise RuntimeError(f"no versions registered for {registry_name}")
        latest = max(versions, key=lambda v: int(v.version)).version
        model_uri = f"models:/{registry_name}/{latest}"
        print(f"  Loading model from UC registry: {model_uri}")
        model = mlflow.sentence_transformers.load_model(model_uri)
        print(f"  Model loaded successfully from registry (version {latest}).")
        return model
    except Exception as e:
        print(f"  Model not found in registry ({e}). Downloading...")

    # download (with backend) + register
    print(f"  Downloading '{model_name}'...")
    try:
        model = SentenceTransformer(model_name, backend=backend)
    except Exception as e:
        print(f"  ONNX backend failed ({e}); falling back to torch.")
        model = SentenceTransformer(model_name)

    # save to Volume so future runs load locally (no download next time)
    if volume_path:
        try:
            os.makedirs(volume_path, exist_ok=True)
            model.save(volume_path)
            print(f"  Model saved to Volume: {volume_path}")
        except Exception as e:
            print(f"  Warning: could not save model to Volume ({e}).")

    print(f"  Registering model to UC: {registry_name}...")
    try:
        with mlflow.start_run():
            mlflow.sentence_transformers.log_model(
                model, artifact_path="model",
                registered_model_name=registry_name,
            )
        print(f"  Model registered: {registry_name}")
    except Exception as e:
        print(f"  Warning: could not register model ({e}). Using in-memory model.")
    return model


def encode_texts(model, texts, batch_size=256):
    """Encode a list of texts -> numpy array (normalized)."""
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
