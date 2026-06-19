"""
embeddings.py — load the sentence-transformer model and encode text.

- backend="onnx"  -> ~2x faster on CPU, ~99% identical embeddings (best for prod CPU)
- backend="torch" -> original behavior
Tries UC Model Registry first, falls back to download + register.
"""

from sentence_transformers import SentenceTransformer


def load_or_save_model(model_name, registry_name, backend="onnx"):
    """Load MiniLM from UC registry; register on first run. Returns model."""
    import mlflow
    mlflow.set_registry_uri("databricks-uc")

    # try registry
    try:
        model_uri = f"models:/{registry_name}/1"
        print(f"  Loading model from UC registry: {model_uri}")
        model = mlflow.sentence_transformers.load_model(model_uri)
        print("  Model loaded successfully from registry.")
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
