"""embeddings.py — load the sentence-transformer model and encode text."""

import os

from sentence_transformers import SentenceTransformer


def load_or_save_model(model_name, registry_name, backend="onnx", volume_path=None):
    """Load MiniLM as ONNX (Volume -> download+save -> registry fallback)."""
    import mlflow
    mlflow.set_registry_uri("databricks-uc")

    if volume_path and os.path.isdir(volume_path) and os.listdir(volume_path):
        try:
            print(f"  Loading model from Volume (backend={backend}): {volume_path}")
            model = SentenceTransformer(volume_path, backend=backend)
            print("  Model loaded successfully from Volume.")
            return model
        except Exception as e:
            print(f"  Volume load failed ({e}); downloading instead...")
    elif volume_path:
        print(f"  No model at Volume {volume_path}; downloading (first run only)...")

    print(f"  Downloading '{model_name}' (backend={backend})...")
    try:
        model = SentenceTransformer(model_name, backend=backend)
    except Exception as e:
        print(f"  {backend} backend download failed ({e}); trying registry fallback...")
        model = _load_from_registry(mlflow, registry_name)
        if model is not None:
            return model
        print("  Registry unavailable; falling back to torch download.")
        model = SentenceTransformer(model_name)

    _save_to_volume(model, volume_path)
    _register(mlflow, model, registry_name)
    return model


def _load_from_registry(mlflow, registry_name):
    """Last-resort load of the registered (torch) model. Returns None on failure."""
    from mlflow import MlflowClient
    try:
        client = MlflowClient(registry_uri="databricks-uc")
        versions = client.search_model_versions(f"name='{registry_name}'")
        if not versions:
            return None
        latest = max(versions, key=lambda v: int(v.version)).version
        model_uri = f"models:/{registry_name}/{latest}"
        print(f"  Loading model from UC registry: {model_uri}")
        model = mlflow.sentence_transformers.load_model(model_uri)
        print(f"  Model loaded from registry (version {latest}) — NOTE: torch backend, slower on CPU.")
        return model
    except Exception as e:
        print(f"  Registry load failed ({e}).")
        return None


def _save_to_volume(model, volume_path):
    """Save the model (incl. ONNX export) to the Volume for local reuse."""
    if not volume_path:
        return
    try:
        os.makedirs(volume_path, exist_ok=True)
        model.save(volume_path)
        print(f"  Model saved to Volume: {volume_path}")
    except Exception as e:
        print(f"  Warning: could not save model to Volume ({e}).")


def _register(mlflow, model, registry_name):
    """Best-effort UC registration (governance); never fails the run."""
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


def encode_texts(model, texts, batch_size=256):
    """Encode a list of texts -> numpy array (normalized)."""
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
