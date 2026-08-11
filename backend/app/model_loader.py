import os
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf

logger = logging.getLogger("har_app")


def _resolve_model_path() -> Path:
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # Standard path relative to this source file: backend/models/har_final_model.keras
    candidate_1 = Path(__file__).resolve().parent.parent / "models" / "har_final_model.keras"
    if candidate_1.exists():
        return candidate_1

    # Fallback to current working directory relative paths
    candidate_2 = Path.cwd() / "backend" / "models" / "har_final_model.keras"
    if candidate_2.exists():
        return candidate_2

    candidate_3 = Path.cwd() / "models" / "har_final_model.keras"
    if candidate_3.exists():
        return candidate_3

    return candidate_1


MODEL_PATH = _resolve_model_path()


def _remove_quantization_config(config):
    if isinstance(config, dict):
        config.pop("quantization_config", None)
        for value in config.values():
            _remove_quantization_config(value)
    elif isinstance(config, list):
        for item in config:
            _remove_quantization_config(item)


def _patch_dense_from_config():
    """
    Keras 3 / TensorFlow 2.x compatibility patch:
    Strips unexpected 'quantization_config: None' from serialized Dense layer configs
    when deserializing .keras models saved across different Keras versions.
    """
    layers_to_patch = []
    if hasattr(tf, "keras") and hasattr(tf.keras, "layers") and hasattr(tf.keras.layers, "Dense"):
        layers_to_patch.append(tf.keras.layers.Dense)
    try:
        import keras
        if hasattr(keras, "layers") and hasattr(keras.layers, "Dense"):
            if keras.layers.Dense not in layers_to_patch:
                layers_to_patch.append(keras.layers.Dense)
    except ImportError:
        pass

    for Dense in layers_to_patch:
        original_from_config = Dense.from_config

        @classmethod
        def patched_from_config(cls, config):
            if isinstance(config, dict):
                config.pop("quantization_config", None)
                _remove_quantization_config(config)
            return original_from_config(config)

        Dense.from_config = patched_from_config


# Confirmed from model's config.json: Dense(6, activation="softmax")
# Order matches the UCI HAR dataset's standard label order.
CLASS_LABELS = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

# Confirmed from model's InputLayer: batch_shape = [null, 128, 9]
SEQ_LEN = 128
N_FEATURES = 9

_model = None


def get_model():
    """
    Loads and caches the .keras model once at startup or on first call.
    Uses compile=False for optimal inference speed and zero optimizer overhead.
    """
    global _model
    if _model is None:
        model_path = _resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        _patch_dense_from_config()
        logger.info(f"Loading .keras model from: {model_path}")

        try:
            _model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as tf_err:
            logger.warning(f"tf.keras.models.load_model failed ({tf_err}), attempting keras.models.load_model fallback...")
            try:
                import keras
                _model = keras.models.load_model(model_path, compile=False)
            except Exception:
                raise tf_err

        logger.info(f"Model loaded successfully: {_model.name}")
    return _model


def predict_sequence(data: np.ndarray):
    """
    data: numpy array of shape (128, 9) — one window of sensor readings.
    Returns: {"activity": str, "confidence": float, "probabilities": {label: float}}
    """
    if data.shape != (SEQ_LEN, N_FEATURES):
        raise ValueError(
            f"Expected input shape ({SEQ_LEN}, {N_FEATURES}), got {data.shape}"
        )

    model = get_model()
    batch = np.expand_dims(data, axis=0)  # (128, 9) -> (1, 128, 9)
    probs = model.predict(batch, verbose=0)[0]  # (6,)

    prob_dict = {CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}
    top_idx = int(np.argmax(probs))

    return {
        "activity": CLASS_LABELS[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": prob_dict,
    }
