# Activity Signal — Human Activity Recognition (HAR) Platform

## 1. Project Overview

**Activity Signal** is a full-stack Human Activity Recognition (HAR) platform that combines deep temporal sequence modeling with real-time sensor inspection, dynamic probability analytics, and LLM-powered biomechanical motion insights.

The system processes fixed windows of **128 consecutive inertial sensor timesteps × 9 channels** (sampled at 50 Hz, representing ~2.56 seconds of movement) and classifies the motion into one of **6 physical activities**:
- 🏃 **Walking**
- 🧗 **Walking Upstairs**
- 🚶 **Walking Downstairs**
- 🪑 **Sitting**
- 🧍 **Standing**
- 🛌 **Laying**

### Key Platform Highlights
- **GRU Deep Sequence Classification:** Powered by a Gated Recurrent Unit neural network trained on the benchmark UCI HAR inertial dataset.
- **9-Channel Time-Series Inspection:** Live interactive visualization of tri-axial body acceleration, angular velocity, and total acceleration.
- **Softmax Probability Distribution:** Granular class probability scoring and confidence breakdown across all 6 activities.
- **AI Motion Insight:** Real-time natural language biomechanical summaries generated via Groq (`llama-3.3-70b-versatile`).
- **Dashboard-Style Sidebar Navigation:** Persistent, sticky left-hand navigation panel with real-time scroll-spy (`IntersectionObserver`), dynamic section unlocking, and collapsible rail toggle (`☰`).

---

## 2. System Architecture

The application is structured into decoupled backend, deep learning, LLM inference, and frontend presentation layers:

```
har-prediction-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py          # Python package initializer
│   │   ├── main.py              # FastAPI server, routing & static file delivery
│   │   ├── model_loader.py      # Keras GRU model loader and tensor inference pipeline
│   │   ├── llm.py               # Groq LLM client & grounded prompt engine
│   │   └── llm_service.py       # LLM interface definitions
│   ├── models/
│   │   └── har_final_model.keras # Trained deep GRU sequence classifier
│   ├── .env                     # Environment configuration (GROQ_API_KEY)
│   └── requirements.txt         # Python backend dependencies
├── frontend/
│   ├── index.html               # Semantic SPA layout (Home, Classify, About, Model Info)
│   ├── style.css                # Dark-theme design system and responsive grid
│   └── script.js                # Client router, Chart.js graphs & API integration
├── sample_128x9.csv             # Reference 128×9 inertial sensor test window
└── README.md                    # Platform documentation
```

### End-to-End Inference Flow
```
[User CSV / XLSX Upload]
         │
         ▼
[Sensor Ingestion & Normalization] ──► Extracts (128, 9) Matrix & Validates Shape
         │
         ▼
[GRU Sequence Encoder] ──────────────► Recurrent Gates compute temporal motion representations
         │
         ▼
[Dense Softmax Layer] ──────────────► Outputs 6-class probability distribution
         │
         ├───► [Chart.js Visualizer] ─► Renders 9-channel wave preview & probability bars
         │
         └───► [Groq LLM Engine] ────► Generates real-time "AI Motion Insight"
```

---

## 3. Setup Instructions

### Prerequisites
- **Python 3.9+** installed on your system.
- A free **Groq Cloud API Key** (obtainable from [console.groq.com](https://console.groq.com/keys)).

### Environment Configuration
1. Clone or extract the project directory.
2. Create or verify the `backend/.env` file with your Groq API key:
```env
GROQ_API_KEY="gsk_your_actual_groq_api_key_here"
```

### Virtual Environment Setup
In your terminal, navigate to the `backend/` folder and create an isolated Python environment:

```bash
cd backend

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

> **macOS (Apple Silicon M1/M2/M3/M4):** If building TensorFlow natively, ensure `tensorflow-macos` and `tensorflow-metal` packages are enabled in `requirements.txt`.

---

## 4. Steps to Run the Application

### 1. Launch the Backend Server
From the activated `backend/` directory, start the FastAPI server via Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Access the User Interface
Open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

*FastAPI directly serves the frontend single-page application and static assets on the same origin.*

### 3. Test with Sample Sensor Data
1. Navigate to the **Classify** tab.
2. Drag and drop the included `sample_128x9.csv` file into the upload drop zone.
3. Review the **Raw 9-Channel Signal Preview**.
4. Click **Get Prediction** to run inference, display the probability distribution bar chart, and receive the **AI Motion Insight**.

---

## 5. API Documentation and Usage

All endpoints are hosted locally under `http://127.0.0.1:8000`. Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### 1. System Health Check
- **Endpoint:** `GET /status`
- **Description:** Verifies server health, expected input tensor shape, and class labels.
- **Response:**
```json
{
  "status": "ok",
  "classes": [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying"
  ],
  "expected_shape": "128 rows x 9 columns",
  "expected_channel_labels": [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
  ]
}
```

---

### 2. Model Architecture & Topology
- **Endpoint:** `GET /model-info`
- **Description:** Inspects internal model architecture, parameters, input/output tensors, and training dataset specifications.
- **Response:**
```json
{
  "model_name": "HAR Recurrent Neural Network (GRU)",
  "input_shape": "(None, 128, 9)",
  "seq_len": 128,
  "n_features": 9,
  "n_classes": 6,
  "total_params": 157798,
  "dataset": "UCI HAR Dataset (Inertial Signals)",
  "sampling_rate": "50 Hz (2.56-second time windows)"
}
```

---

### 3. Raw Sensor Preview & Channel Mapping
- **Endpoint:** `POST /preview`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file` (File, required): `.csv` or `.xlsx` file containing a 128×9 matrix.
  - `channel_labels` (JSON String, optional): Custom array of 9 channel names.
- **Response:**
```json
{
  "source_labels": ["body_acc_x", "body_acc_y", "..."],
  "mapping": [],
  "warning": null,
  "used_default_order": true,
  "resolved_channel_labels": ["body_acc_x", "body_acc_y", "..."],
  "preview_series": [[0.018, 0.002, "..."], "..."]
}
```

---

### 4. Activity Classification
- **Endpoint:** `POST /predict`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file` (File, required): 128×9 sensor time-window file.
  - `sampling_rate` (String, optional): Sampling frequency (default: `50`).
- **Example Curl:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@sample_128x9.csv"
```
- **Response:**
```json
{
  "activity": "Laying",
  "confidence": 0.9999995,
  "probabilities": {
    "Walking": 1.22e-12,
    "Walking Upstairs": 1.31e-10,
    "Walking Downstairs": 1.25e-10,
    "Sitting": 5.30e-07,
    "Standing": 1.76e-09,
    "Laying": 0.9999995
  },
  "metadata": {
    "sampling_rate": "50",
    "recording_timestamp": "2026-08-10 23:15:00"
  }
}
```

---

### 5. AI Motion Insight
- **Endpoint:** `POST /explain`
- **Content-Type:** `application/json`
- **Payload:**
```json
{
  "activity": "Laying",
  "confidence": 0.9999995,
  "probabilities": {
    "Walking": 1.22e-12,
    "Walking Upstairs": 1.31e-10,
    "Walking Downstairs": 1.25e-10,
    "Sitting": 5.30e-07,
    "Standing": 1.76e-09,
    "Laying": 0.9999995
  }
}
```
{
  "explanation": "The model detected Laying with 99.9% confidence, indicating an unambiguous static posture. The tri-axial sensor signature demonstrates near-zero dynamic acceleration consistent with a resting position."
}
```

---

## 6. Hugging Face Spaces Deployment Guide

This repository includes a production-ready `Dockerfile` and `requirements.txt` configured specifically for deploying to **Hugging Face Spaces (Docker SDK)**.

### Step 1: Create a New Hugging Face Space
1. Log in to [huggingface.co](https://huggingface.co) and click **New Space**.
2. Set your Space Name (e.g. `har-activity-recognition`).
3. Select **Docker** as the Space SDK (Blank template).
4. Choose **Public** or **Private** visibility and click **Create Space**.

### Step 2: Configure Repository Secrets (Groq API Key)
1. In your newly created Space, navigate to **Settings** → **Variables and secrets**.
2. Under **Secrets**, click **New secret**.
3. Set Name: `GROQ_API_KEY`
4. Set Value: `gsk_your_actual_groq_api_key` (from [console.groq.com](https://console.groq.com/keys)).
5. Click **Save**.

### Step 3: Push the Codebase to Hugging Face Spaces
Clone your Hugging Face Space repository or add it as a git remote, then push the project:

```bash
# Add Hugging Face Space as a remote (replace with your Space URL)
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push to Hugging Face
git push space main
```

*(Note: The `.keras` model file is ~1.9 MB, well within standard git push limits.)*

### Step 4: Verify Deployment & Endpoints
Once Hugging Face finishes building the Docker container, your Space will be live:
- **Web Interface:** Direct access at `https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space`
- **Health Check:** `GET https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/status`
- **API Docs:** `GET https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/docs`
- **Inference:** `POST https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/predict` with `sample_128x9.csv`

