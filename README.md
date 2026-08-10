# Activity Signal — Human Activity Recognition (HAR) GRU Platform

**Activity Signal** is an end-to-end Human Activity Recognition platform combining deep temporal sequence modeling with real-time sensor inspection, dynamic probability analytics, and LLM-powered biomechanical motion insights.

The platform takes **128 consecutive inertial sensor timesteps × 9 channels** (captured at 50 Hz, ~2.56-second window duration) and predicts one of **6 physical activities**:
- 🏃 **Walking**
- 🧗 **Walking Upstairs**
- 🚶 **Walking Downstairs**
- 🪑 **Sitting**
- 🧍 **Standing**
- 🛌 **Laying**

---

## 📁 Repository Structure

```
har-prediction-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py          # Marks app as a Python package
│   │   ├── main.py              # FastAPI application, routing & static file server
│   │   ├── model_loader.py      # Loads .keras GRU model and executes sequence inference
│   │   ├── llm.py               # Groq LLM client & AI Motion Insight prompt engine
│   │   └── llm_service.py       # Re-exports LLM inference interfaces
│   ├── models/
│   │   └── har_final_model.keras # Trained deep GRU sequence classification model
│   ├── .env                     # Environment variables (GROQ_API_KEY)
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── index.html               # Responsive multi-page single-page application (SPA)
│   ├── style.css                # Custom dark-theme design system & dashboard layout
│   └── script.js                # Frontend router, Chart.js visuals, API integration
├── sample_128x9.csv             # Reference 128×9 inertial sensor test window
└── README.md                    # Documentation & setup guide
```

---

## ✨ Key Features & Architecture

1. **GRU Deep Sequence Classification (`POST /predict`):**
   - Recurrent Neural Network with Gated Recurrent Units trained on the benchmark UCI HAR dataset.
   - Evaluates temporal dynamics across 128 consecutive timesteps.

2. **9-Channel Sensor Signal Preview (`POST /preview`):**
   - Interactive multi-channel time-series graph plotting `body_acc_x/y/z`, `body_gyro_x/y/z`, and `total_acc_x/y/z`.
   - Intelligent header detection, channel normalization, and mapping.

3. **Softmax Probability Breakdown:**
   - Real-time Chart.js bar breakdown across all 6 target activity classes with confidence percentage ranking.

4. **AI Motion Insight (`POST /explain`):**
   - Real-time biomechanical analysis generated via Groq (`llama-3.3-70b-versatile`).
   - Contextually grounded explanations explaining confidence margins, landslide probabilities, and motion characteristics.

5. **Full-Height Dashboard Navigation Sidebar:**
   - Persistent, sticky left-hand navigation panel providing instant jump links:
     - **Preview** ➔ 9-Channel sensor signal graph
     - **Predicted Activity** ➔ Live activity badge & confidence score
     - **Full Probability Breakdown** ➔ 6-Class softmax distribution
     - **AI Motion Insight** ➔ LLM explanation card
   - **IntersectionObserver Scroll-Spy:** Automatically highlights the section currently in view.
   - **Dynamic Availability State:** Items unlock progressively as sensor data is previewed and predictions return.
   - **Collapsible Icon Rail (`☰`):** Toggles between expanded 240px dashboard panel and 68px compact icon rail.

6. **Interactive Multi-Page Views:**
   - **Home:** Interactive landing view with animated sensor pulse graphics.
   - **Classify:** Drag-and-drop file ingestion, signal inspection, and prediction workbench.
   - **About:** Comprehensive domain background, sensor specifications, and inference pipeline guide.
   - **Model Info:** Neural network architecture flow, tensor dimensions, and layer specifications.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.9+ installed on your system.
- A free [Groq Cloud API Key](https://console.groq.com/keys) for AI Motion Insights.

### 2. Environment Configuration
Create or update `backend/.env` with your Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 3. Setup & Start Backend Server

Open your terminal in the project directory:

```bash
cd backend

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server (serves both API & Frontend)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

> **Apple Silicon Mac Note (M1/M2/M3/M4):** If compiling TensorFlow natively on macOS, ensure `tensorflow-macos` and `tensorflow-metal` are installed.

### 4. Access the Application
Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📊 Sensor Input Specifications

The model strictly expects a single window of **128 rows × 9 numeric sensor columns**:

| Column Index | Sensor Channel | Units / Description |
|:---:|:---|:---|
| `1` | `body_acc_x` | Tri-axial body acceleration X-axis (g) |
| `2` | `body_acc_y` | Tri-axial body acceleration Y-axis (g) |
| `3` | `body_acc_z` | Tri-axial body acceleration Z-axis (g) |
| `4` | `body_gyro_x` | Tri-axial angular velocity X-axis (rad/s) |
| `5` | `body_gyro_y` | Tri-axial angular velocity Y-axis (rad/s) |
| `6` | `body_gyro_z` | Tri-axial angular velocity Z-axis (rad/s) |
| `7` | `total_acc_x` | Tri-axial total acceleration X-axis (g) |
| `8` | `total_acc_y` | Tri-axial total acceleration Y-axis (g) |
| `9` | `total_acc_z` | Tri-axial total acceleration Z-axis (g) |

*You can test the application immediately using the included `sample_128x9.csv` file.*

---

## 🛠️ API Reference

- `GET /status` — Health check endpoint returning model input shape and class definitions.
- `GET /model-info` — Inspects neural network architecture, parameter counts, and layer topologies.
- `POST /preview` — Ingests a `.csv` or `.xlsx` file, extracts 128×9 series, and returns normalized channel mapping.
- `POST /predict` — Runs GRU sequence classification and returns predicted activity with softmax distribution.
- `POST /explain` — Sends prediction context to Groq LLM and returns natural-language motion insight.

---

## 🔧 Troubleshooting

- **"Model not found at ..."**: Ensure `har_final_model.keras` is placed inside `backend/models/`.
- **"AI Motion Insight temporarily unavailable"**: Verify that `GROQ_API_KEY` is present in `backend/.env` with active quota.
- **"Expected 128 rows × 9 columns"**: Ensure uploaded CSVs contain exactly 128 rows and 9 numeric features without extraneous timestamp or identifier columns.
