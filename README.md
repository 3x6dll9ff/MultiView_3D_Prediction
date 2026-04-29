# TriView3D — Multi-View 3D Cell Reconstruction & Morphological Analysis

A comprehensive platform for predicting 3D cell morphology from three 2D micrographs (top, bottom, and side projections). The system combines a two-stage coarse-to-fine reconstruction pipeline, a generative Conditional VAE, an MLP anomaly classifier, and an AI-powered morphological analysis agent built on Google Gemini.

---

## 🚀 Key Features

- **Coarse-to-Fine 3D Reconstruction**: Two-stage pipeline — Base Autoencoder captures global volume, Detail Refiner adds surface nuances.
- **Generative Synthesis (CVAE)**: Conditional Variational Autoencoder produces biologically plausible cell shapes with smooth surfaces and supports latent space interpolation.
- **Anomaly Classification**: MLP classifier operates on the latent space to detect morphological anomalies (Normal vs Anomaly) with high confidence.
- **AI Morphological Analysis (Agent + RAG)**: A two-agent Gemini-based pipeline (Writer + Verifier) generates structured scientific reports grounded in a local knowledge base and real-time literature search.
- **Interactive 3D Dashboard**: React + Three.js frontend with real-time mesh visualization (Marching Cubes), metric comparison, and pipeline progress tracking.
- **Cloud-Ready Training**: Google Colab notebooks with AMP support for T4/A100 GPUs.

---

## 🏗️ Architecture

### Pipeline 1: Reconstruction + Classification

```
Input (3 × 64×64 projections)
  │
  ├─► Base Autoencoder (CNN) ──► Coarse 3D Volume
  │     └─► Detail Refiner ──────► Refined 3D Volume
  │
  ├─► Conditional VAE ──────────► Generated 3D Volume
  │
  └─► Latent Vector ──► MLP Classifier ──► Normal / Anomaly
```

**Stage 1 — Base Autoencoder**: Three 2D encoders process top, bottom, and side projections into a shared 256-dim latent vector, which is decoded into a 64³ voxel grid.

**Stage 2 — Detail Refiner**: Takes the coarse output + lifted projections and refines surface details via a residual 3D U-Net.

**Conditional VAE**: An alternative probabilistic branch that reconstructs volumes through a regularized latent space, producing smoother surfaces suited for shape analysis.

**MLP Classifier**: A lightweight 4-layer classifier that reads the autoencoder's latent vector to distinguish between normal and anomalous cell morphologies.

### Pipeline 2: Agent + RAG (Morphological Analysis)

```
Morphometric Extraction
  │
  ├─► RAG Retrieve (local JSONL knowledge base)
  ├─► Literature Search (Europe PMC API)
  │
  ├─► Writer Agent (Gemini 2.5 Flash) ──► Structured JSON Report
  └─► Verifier Agent (Gemini 2.5 Flash) ──► Validated Report + Corrections
```

The analysis pipeline extracts morphometric features (volume, sphericity, convexity, eccentricity, surface roughness) from the reconstructed volume and passes them, along with RAG-retrieved scientific context, to two LLM agents:

1. **Writer Agent** — Generates a structured report with classification interpretation, key metric deviations, evidence citations, limitations, and recommendations.
2. **Verifier Agent** — Validates the draft against the knowledge base, corrects numeric inaccuracies, softens overclaiming, and logs all corrections.

Both agents enforce guardrails: no diagnostic claims, no mutation attributions, morphology-first language only.

---

## 📊 Performance Metrics

### Reconstruction Quality

| Metric | Base Model | + Refiner | CVAE |
|---|---|---|---|
| **Dice Score** | 0.91 | **0.94** | 0.92 |
| **IoU** | 0.83 | **0.88** | 0.85 |
| **Surface HD95** | 2.5 vox | **1.8 vox** | 2.1 vox |

### Classification

| Metric | Value |
|---|---|
| **Accuracy** | 97.5% |
| **Precision** | 98.1% |
| **Recall** | 96.8% |

---

## 📂 Dataset: SHAPR

The models are trained on the **SHAPR Dataset** (Red Blood Cells):

- **Source**: [Zenodo — SHAPR](https://zenodo.org/records/7031924)
- **Content**: 602 RBC instances — discocytes, stomatocytes, echinocytes, spherocytes.
- **Projections**: 64×64 images generated via sum pooling across three orthogonal axes.
- **Ground Truth**: 64³ binary voxel grids.

---

## 🛠️ Quick Start

### Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/3x6dll9ff/TriView3D.git
cd TriView3D

# Create .env with your Gemini API key (for AI analysis)
echo "GEMINI_API_KEY=your_key_here" > .env

# Launch the full stack
docker compose up --build
```

| Service | URL |
|---|---|
| **Web UI** | http://localhost:5173 |
| **API** | http://localhost:8000 |
| **Swagger Docs** | http://localhost:8000/docs |

### Manual Installation

```bash
# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download and prepare the dataset
python3 src/download_data.py
python3 src/prepare_dataset.py

# Start the API server
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## 🎓 Training (Google Colab)

Pre-configured notebooks are available in the `notebooks/` directory:

| Notebook | Purpose |
|---|---|
| `train_colab.ipynb` | Base Autoencoder (Stage 1) |
| `train_refiner_colab.ipynb` | Detail Refiner (Stage 2) |
| `train_vae_colab.ipynb` | Conditional VAE |
| `train_classifier_colab.ipynb` | MLP Anomaly Classifier |

After training, place the weight files in the `results/` directory:
- `best_autoencoder.pt`
- `best_refiner.pt`
- `best_vae.pt`
- `best_classifier.pt`

---

## 📡 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/cells` | GET | List available cell samples |
| `/api/preview/{filename}` | GET | Get 2D projection previews |
| `/api/predict/{filename}` | POST | CNN + Refiner reconstruction with classification |
| `/api/predict-vae/{filename}` | POST | CVAE reconstruction |
| `/api/metrics` | GET | CNN model evaluation metrics |
| `/api/metrics-vae` | GET | CVAE model evaluation metrics |
| `/api/status` | GET | Available models and system status |
| `/health` | GET | Health check |

### Agent Pipeline Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/agent/retrieve` | POST | RAG retrieval from local knowledge base |
| `/api/agent/search` | POST | Literature search via Europe PMC |
| `/api/agent/generate` | POST | Writer Agent — LLM-generated structured report |
| `/api/agent/verify` | POST | Verifier Agent — report validation and correction |
| `/api/agent/answer` | POST | Legacy fallback answer endpoint |

---

## 💻 Tech Stack

| Layer | Technologies |
|---|---|
| **ML Core** | PyTorch (AMP), NumPy, SciPy, scikit-image, scikit-learn |
| **LLM** | Google Gemini 2.5 Flash via `google-genai` SDK |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | React 19, TypeScript, Vite 8, React Three Fiber, Three.js |
| **Infrastructure** | Docker Compose, Google Colab |

---

## 📁 Project Structure

```
TriView3D/
├── src/                        # Python backend
│   ├── api.py                  # FastAPI application (all endpoints)
│   ├── autoencoder.py          # Base CNN Autoencoder (Stage 1)
│   ├── refiner.py              # Detail Refiner (Stage 2)
│   ├── vae.py                  # Conditional VAE
│   ├── classifier.py           # MLP anomaly classifier
│   ├── llm.py                  # Gemini LLM integration (Writer + Verifier)
│   ├── morphometrics.py        # Morphometric feature extraction
│   ├── reconstruction_utils.py # Volume projection and lifting utils
│   ├── evaluate.py             # Model evaluation pipeline
│   ├── train_*.py              # Training scripts
│   ├── dataset.py              # Dataset loading and preprocessing
│   └── visualize.py            # Visualization utilities
│
├── frontend/                   # React + TypeScript frontend
│   └── src/
│       ├── App.tsx             # Main application (dashboard, 3D viewer)
│       ├── index.css           # Design system and component styles
│       └── components/         # PipelineTracker, MetricStrip, Sidebar
│
├── data/
│   ├── processed/              # Pre-processed projection/volume data
│   └── rag/                    # RAG knowledge base (JSONL)
│
├── notebooks/                  # Google Colab training notebooks
├── results/                    # Trained model weights
├── docs/                       # Technical documentation
├── docker-compose.yml          # Full-stack orchestration
├── Dockerfile.backend          # Backend container
└── requirements-api.txt        # Backend dependencies (Docker)
```

---

## 📄 License

This project is developed as part of a university thesis. See individual files for licensing details.
