import asyncio
import io
import json
import logging
import os
import re
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("har_app")

groq_key = os.getenv("GROQ_API_KEY")
logger.info(f"[Startup] GROQ_API_KEY configured: {bool(groq_key)} (Prefix: {groq_key[:8] if groq_key else 'None'})")

from .model_loader import predict_sequence, CLASS_LABELS, SEQ_LEN, N_FEATURES
from .llm import explain_prediction as llm_explain_prediction

EXPECTED_CHANNEL_LABELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

app = FastAPI(title="Human Activity Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExplainRequest(BaseModel):
    activity: str
    confidence: float
    probabilities: dict[str, float]


def _is_numeric(value):
    try:
        float(value)
        return True
    except Exception:
        return False


def _normalize_label(label):
    if label is None:
        return ""
    return str(label).strip().lower().replace(" ", "_")


def _canonicalize_channel_label(label):
    normalized = _normalize_label(label)
    if not normalized:
        return None

    mappings = [
        (r'^(?:body[_\s-]*)?(?:acc|accelerometer)[_\s-]*([xyz])$', 'body_acc_{}'),
        (r'^(?:body[_\s-]*)?(?:gyro|gyroscope)[_\s-]*([xyz])$', 'body_gyro_{}'),
        (r'^(?:total[_\s-]*)?(?:acc|accelerometer)[_\s-]*([xyz])$', 'total_acc_{}'),
        (r'^t[_\s-]*acc[_\s-]*([xyz])$', 'total_acc_{}'),
        (r'^acc[_\s-]*([xyz])$', 'body_acc_{}'),
    ]

    for pattern, template in mappings:
        match = re.match(pattern, normalized)
        if match:
            return template.format(match.group(1))

    return None


def _parse_channel_labels(raw):
    if not raw:
        return None
    try:
        labels = json.loads(raw)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="channel_labels must be valid JSON: an array of 9 sensor channel names.",
        )
    if not isinstance(labels, list) or len(labels) != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"channel_labels must be a JSON array with exactly {N_FEATURES} items.",
        )
    if not all(isinstance(label, str) and label.strip() for label in labels):
        raise HTTPException(
            status_code=400,
            detail="Each channel label must be a non-empty string.",
        )
    return [str(label).strip() for label in labels]


def _load_sensor_file(contents, filename):
    if filename.endswith(".csv"):
        text = contents.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text), header=None)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(contents), header=None, engine="openpyxl")
    else:
        raise HTTPException(status_code=400, detail="Only .csv or .xlsx files are supported")

    if df.shape[1] != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {N_FEATURES} columns but found {df.shape[1]}. "
                f"Please upload a file with exactly {N_FEATURES} numeric sensor columns."
            ),
        )

    first_row = df.iloc[0]
    header_is_text = first_row.apply(lambda v: isinstance(v, str) and not _is_numeric(v.strip())).any()
    if header_is_text:
        header = [str(x).strip() for x in first_row.tolist()]
        data_df = df.iloc[1:].reset_index(drop=True)
    else:
        header = None
        data_df = df

    return header, data_df


def _derive_total_acc(data_df, mapped_positions):
    return pd.DataFrame(
        {
            "total_acc_x": data_df.iloc[:, mapped_positions["body_acc_x"]].astype(np.float32),
            "total_acc_y": data_df.iloc[:, mapped_positions["body_acc_y"]].astype(np.float32),
            "total_acc_z": data_df.iloc[:, mapped_positions["body_acc_z"]].astype(np.float32),
        }
    )


def _order_and_map_dataframe(header, data_df, explicit_labels=None):
    info = {
        "mapping": [],
        "warning": None,
        "used_default_order": False,
        "approx_total_acc": False,
        "header_labels": None,
        "resolved_channel_labels": EXPECTED_CHANNEL_LABELS,
    }

    if explicit_labels is not None:
        header_labels = explicit_labels
        info["header_labels"] = header_labels
    elif header is not None:
        header_labels = [str(x).strip() for x in header]
        info["header_labels"] = header_labels
    else:
        header_labels = None

    if header_labels is None:
        info["warning"] = (
            "No header row detected — using default channel order "
            "(body_acc x/y/z, body_gyro x/y/z, total_acc x/y/z). Verify this is correct."
        )
        info["used_default_order"] = True
        return data_df.iloc[:, list(range(N_FEATURES))], info

    normalized_headers = [_normalize_label(label) for label in header_labels]
    mapped_columns = [_canonicalize_channel_label(label) for label in normalized_headers]

    info["mapping"] = [
        {
            "source": header_labels[idx] or f"column_{idx + 1}",
            "resolved": mapped_columns[idx] or "unrecognized",
        }
        for idx in range(len(header_labels))
    ]

    recognized = [label for label in mapped_columns if label is not None]
    recognized_set = set(recognized)

    if set(mapped_columns) == set(EXPECTED_CHANNEL_LABELS):
        column_order = [mapped_columns.index(name) for name in EXPECTED_CHANNEL_LABELS]
        info["resolved_channel_labels"] = EXPECTED_CHANNEL_LABELS
        return data_df.iloc[:, column_order], info

    body_acc_gyro = {
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
    }

    if body_acc_gyro.issubset(recognized_set) and not any(label.startswith("total_acc") for label in recognized_set):
        mapped_positions = {label: mapped_columns.index(label) for label in body_acc_gyro}
        channel_order = [
            data_df.iloc[:, mapped_positions[label]].astype(np.float32)
            for label in [
                "body_acc_x",
                "body_acc_y",
                "body_acc_z",
                "body_gyro_x",
                "body_gyro_y",
                "body_gyro_z",
            ]
        ]
        total_acc_df = _derive_total_acc(data_df, mapped_positions)
        combined = pd.concat(channel_order + [total_acc_df[col] for col in ["total_acc_x", "total_acc_y", "total_acc_z"]], axis=1)
        combined.columns = EXPECTED_CHANNEL_LABELS
        info["warning"] = (
            "Missing total_acc labels were approximated from body_acc values. "
            "Gravity is not available, so total_acc is approximated as body_acc."
        )
        info["approx_total_acc"] = True
        info["resolved_channel_labels"] = EXPECTED_CHANNEL_LABELS
        return combined, info

    if len(recognized) == 0:
        info["warning"] = (
            "Header row contains unrecognized labels. Using default channel order "
            "(body_acc x/y/z, body_gyro x/y/z, total_acc x/y/z). Verify this is correct."
        )
        info["used_default_order"] = True
        return data_df.iloc[:, list(range(N_FEATURES))], info

    info["warning"] = (
        "Header labels were partially recognized. Using default channel order "
        "(body_acc x/y/z, body_gyro x/y/z, total_acc x/y/z). Verify this is correct."
    )
    info["used_default_order"] = True
    return data_df.iloc[:, list(range(N_FEATURES))], info


def _validate_dataframe(data_df, labels):
    if data_df.shape != (SEQ_LEN, N_FEATURES):
        raise HTTPException(
            status_code=400,
            detail=(
                f"This model expects exactly {SEQ_LEN} rows x {N_FEATURES} columns, "
                f"but your upload contains {data_df.shape[0]} rows x {data_df.shape[1]} columns."
            ),
        )

    if data_df.isnull().values.any():
        raise HTTPException(status_code=400, detail="Uploaded sensor data contains missing values.")

    try:
        arr = data_df.astype(np.float32).to_numpy()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded sensor data must contain only numeric values.")

    if np.isnan(arr).any():
        raise HTTPException(status_code=400, detail="Uploaded sensor data contains NaN values.")

    return arr


def _process_sensor_file(contents, filename, explicit_labels=None):
    header, data_df = _load_sensor_file(contents, filename)
    ordered_df, info = _order_and_map_dataframe(header, data_df, explicit_labels=explicit_labels)
    arr = _validate_dataframe(ordered_df, explicit_labels or EXPECTED_CHANNEL_LABELS)
    return arr, ordered_df, info


@app.get("/status")
def root():
    return {
        "status": "ok",
        "classes": CLASS_LABELS,
        "expected_shape": f"{SEQ_LEN} rows x {N_FEATURES} columns",
        "expected_channel_labels": EXPECTED_CHANNEL_LABELS,
    }


@app.get("/model-info")
def get_model_info():
    try:
        from .model_loader import get_model
        model = get_model()
        layers_info = []
        for layer in model.layers:
            input_shape = str(layer.input_shape) if hasattr(layer, "input_shape") else "N/A"
            output_shape = str(layer.output_shape) if hasattr(layer, "output_shape") else "N/A"
            layers_info.append({
                "name": layer.name,
                "type": layer.__class__.__name__,
                "input_shape": input_shape,
                "output_shape": output_shape,
            })
        total_params = int(model.count_params()) if hasattr(model, "count_params") else None
    except Exception as e:
        layers_info = []
        total_params = None

    return {
        "model_name": "HAR Recurrent Neural Network (GRU)",
        "input_shape": f"(None, {SEQ_LEN}, {N_FEATURES})",
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "n_classes": len(CLASS_LABELS),
        "class_labels": CLASS_LABELS,
        "expected_channels": EXPECTED_CHANNEL_LABELS,
        "layers": layers_info,
        "total_params": total_params,
        "dataset": "UCI HAR Dataset (Inertial Signals)",
        "sampling_rate": "50 Hz (2.56-second time windows)",
    }


@app.post("/preview")
async def preview(file: UploadFile = File(...), channel_labels: str = Form(None)):
    filename = (file.filename or "").lower()
    contents = await file.read()

    try:
        labels = _parse_channel_labels(channel_labels) if channel_labels else None
    except HTTPException:
        raise

    try:
        _, ordered_df, info = _process_sensor_file(contents, filename, explicit_labels=labels)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    return {
        "source_labels": info["header_labels"],
        "mapping": info["mapping"],
        "warning": info["warning"],
        "used_default_order": info["used_default_order"],
        "approx_total_acc": info["approx_total_acc"],
        "resolved_channel_labels": info["resolved_channel_labels"],
        "preview_series": ordered_df.astype(float).round(6).values.tolist(),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    subject_id: str = Form(None),
    sampling_rate: str = Form("50"),
    device_type: str = Form(None),
    recording_timestamp: str = Form(None),
    channel_labels: str = Form(None),
):
    filename = (file.filename or "").lower()
    contents = await file.read()

    try:
        labels = _parse_channel_labels(channel_labels) if channel_labels else None
    except HTTPException:
        raise

    try:
        arr, _, info = _process_sensor_file(contents, filename, explicit_labels=labels)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    try:
        result = predict_sequence(arr)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    metadata = {
        "subject_id": subject_id,
        "sampling_rate": sampling_rate,
        "device_type": device_type,
        "recording_timestamp": recording_timestamp or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return {
        **result,
        "metadata": metadata,
        "used_channel_order": labels,
        "header_mapping": info["mapping"],
        "mapping_warning": info["warning"],
        "approx_total_acc": info["approx_total_acc"],
    }


@app.post("/explain")
async def explain(req: ExplainRequest):
    payload = req.dict()
    result = llm_explain_prediction(payload)
    if not result.ok:
        status_code = 429 if result.error_code == "QUOTA_EXCEEDED" else 502
        raise HTTPException(status_code=status_code, detail=result.error_message)
    return {"explanation": result.text}


# Mount frontend static files AFTER the API routes so the `/predict` POST
# endpoint keeps precedence (static files only handle GET/HEAD). This serves
# the UI from the backend on the same origin, avoiding cross-origin POSTs.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/{file_path:path}")
    def serve_frontend_files(file_path: str):
        target = FRONTEND_DIR / file_path
        if target.exists() and target.is_file():
            return FileResponse(target)
        raise HTTPException(status_code=404, detail="Not found")
