import numpy as np
import tensorflow as tf
from pathlib import Path

# Your model file — already included in this project.
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "har_final_model.keras"


def _remove_quantization_config(config):
    if isinstance(config, dict):
        config.pop("quantization_config", None)
        for value in config.values():
            _remove_quantization_config(value)
    elif isinstance(config, list):
        for item in config:
            _remove_quantization_config(item)


def _patch_dense_from_config():
    Dense = tf.keras.layers.Dense
    original_from_config = Dense.from_config

    @classmethod
    def patched_from_config(cls, config):
        if isinstance(config, dict):
            config.pop("quantization_config", None)
            _remove_quantization_config(config)
        return original_from_config(config)

    Dense.from_config = patched_from_config

# Confirmed from your model's config.json: Dense(6, activation="softmax")
# Order matches the UCI HAR dataset's standard label order.
CLASS_LABELS = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

# Confirmed from your model's InputLayer: batch_shape = [null, 128, 9]
SEQ_LEN = 128
N_FEATURES = 9

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _patch_dense_from_config()
        _model = tf.keras.models.load_model(MODEL_PATH)
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
