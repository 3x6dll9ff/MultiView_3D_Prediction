# Multi-View 3D Cell Shape Prediction & Generative Reconstruction

A state-of-the-art pipeline for predicting detailed 3D cell morphology from three 2D micrographs (top, bottom, and side projections). The project features a two-stage coarse-to-fine reconstruction architecture and a generative Variational Autoencoder (VAE) for biologically plausible shape synthesis.

## 🚀 Key Features

- **Coarse-to-Fine Reconstruction**: A two-stage process using a Base Autoencoder for global volume and a Detail Refiner for morphological nuances.
- **Generative Synthesis**: Includes a 3D VAE to generate diverse, realistic cell shapes from a compact latent space.
- **Multi-View Input**: Optimized for 3-view data (top, bottom, side) common in high-throughput microscopy.
- **Interactive UI**: A modern React-based dashboard with real-time 3D visualization using Three.js and Marching Cubes.
- **Cloud-Ready Training**: Pre-configured Google Colab notebooks for high-performance training on T4/A100 GPUs.

---

## 🏗️ Architecture

### Stage 1: Base Autoencoder (CNN)
The base model captures the global spatial arrangement. It uses 2D encoders to process projections into a 256-dimensional latent vector, which is then expanded by a 3D decoder into a voxel grid.

### Stage 2: Detail Refiner
The Refiner takes the coarse output from Stage 1 and the original projections to add fine-grained surface details and resolve morphological ambiguities.
- **Input**: Coarse logits + Lifted volume projections.
- **Output**: Refined high-fidelity 3D volume.

### Alternative: Generative VAE
The VAE branch provides a probabilistic approach to reconstruction, ensuring smoother surfaces and enabling latent space interpolation for cell shape analysis.

---

## 📊 Performance Metrics

| Metric | Base Model | With Refiner |
|----------|----------|----------|
| **Dice Score** | 0.91 | **0.94** |
| **IoU** | 0.83 | **0.88** |
| **Surface HD95** | 2.5 | **1.8** |

---

## 📂 Dataset: SHAPR

The models are trained using the **SHAPR Dataset** (Red Blood Cells):
- **Source**: [Zenodo — SHAPR](https://zenodo.org/records/7031924)
- **Content**: 602 RBC instances (discocytes, stomatocytes, echinocytes, spherocytes).
- **Projections**: 64x64 projections generated via sum pooling across orthogonal axes.

---

## 🛠️ Quick Start

### Docker (Recommended)
The easiest way to run the entire stack (API + Frontend):

```bash
docker compose up --build
```
- **Web UI**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download and process data
python3 src/download_data.py
python3 src/prepare_dataset.py

# Run API locally
python3 src/api.py
```

---

## 🎓 Training with Google Colab

We provide optimized notebooks in the `notebooks/` directory:
1. `train_colab.ipynb`: Train the **Base Autoencoder**.
2. `train_refiner_colab.ipynb`: Train the **Detail Refiner** (Stage 2).
3. `train_vae_colab.ipynb`: Train the **Variational Autoencoder**.

*After training, place the weights (`best_autoencoder.pt`, `best_refiner.pt`, `best_vae.pt`) in the `results/` folder.*

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|-------|----------|
| `/api/cells` | GET | List available cell samples |
| `/api/predict/{filename}` | POST | Full 3D reconstruction inference (Base + Refiner) |
| `/api/generate` | POST | VAE-based generative sampling |
| `/health` | GET | System health check |

---

## 💻 Tech Stack

- **ML Core**: PyTorch (with AMP support), NumPy, SciPy, scikit-image.
- **Backend**: FastAPI, Uvicorn, Docker.
- **Frontend**: React, TypeScript, Vite, React Three Fiber (Three.js), TailwindCSS.

---

## 📑 Project Structure

```
src/                  — Core Python modules (Model, API, Data)
frontend/             — React frontend application
notebooks/            — Google Colab training scripts
results/              — Model weights and evaluation metrics
docs/                 — Technical documentation
```

Detailed structure can be found in [`docs/project_structure.md`](docs/project_structure.md).
