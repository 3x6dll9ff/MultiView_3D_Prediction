import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import urlopen

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autoencoder import TriViewAutoencoder
from src.classifier import LatentClassifier
from src.morphometrics import extract_all_metrics
from src.refiner import DetailRefiner
from src.llm import generate_report, verify_report, is_available as llm_available
from src.reconstruction_utils import (
    get_view_names,
    infer_in_channels_from_state_dict,
    infer_skip_channels_from_state_dict,
    lift_views_to_volume,
    project_volume_batch,
)
from src.vae import TriViewCVAE, best_of_k_generate

app = FastAPI(title="3D Cell Reconstruction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data/processed"
MODEL_PATH = "results/best_autoencoder.pt"
VAE_MODEL_PATH = "results/best_vae.pt"
REFINER_PATH = "results/best_refiner.pt"
CLASSIFIER_PATH = "results/best_classifier.pt"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_PATH = PROJECT_ROOT / "data" / "rag" / "morphology_sources.jsonl"
RAG_DISCOVERED_PATH = PROJECT_ROOT / "data" / "rag" / "morphology_discovered.jsonl"
FORBIDDEN_RAG_TERMS = (
    "cancer",
    "tumor",
    "neoplasm",
    "neoplastic",
    "preneoplastic",
    "lesion",
    "cytologic",
    "dysplastic",
    "metaplastic",
    "malignan",
    "metast",
    "oncolog",
    "patholog",
    "diagnos",
    "molecular",
    "prognos",
    "patient",
    "treatment",
)

model = None
vae_model = None
refiner_model = None
classifier_model = None
model_view_names = get_view_names("tri")
vae_view_names = get_view_names("tri")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class AgentRequest(BaseModel):
    filename: str
    classification: dict[str, Any] | None = None
    morphology: dict[str, float] | None = None
    metrics: dict[str, float] | None = None
    cell_type: str | None = None
    missing_topics: list[str] | None = None
    retrieved: list[dict[str, Any]] | None = None
    discovered: list[dict[str, Any]] | None = None
    draft_report: dict[str, Any] | None = None


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def is_safe_morphology_source(record: dict[str, Any]) -> bool:
    """Keep RAG grounding morphology-only and non-clinical for this demo."""
    text = " ".join(
        str(record.get(key, ""))
        for key in ("title", "content", "limitations", "follow_up", "source_type")
    ).lower()
    return not any(term in text for term in FORBIDDEN_RAG_TERMS)


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def morphology_topics(morphology: dict[str, float] | None, classification: dict[str, Any] | None) -> list[str]:
    topics: list[str] = []
    morphology = morphology or {}
    if classification and str(classification.get("class", "")).lower() == "anomaly":
        topics.extend(["morphology_based_anomaly", "limitations"])
    if float(morphology.get("surface_roughness", 0.0)) >= 3.0:
        topics.append("surface_roughness")
    if float(morphology.get("convexity", 1.0)) <= 0.965:
        topics.append("convexity")
        topics.append("irregular_shape")
    if float(morphology.get("sphericity", 1.0)) <= 0.84:
        topics.append("sphericity")
    if float(morphology.get("eccentricity", 1.0)) <= 0.40:
        topics.append("eccentricity")
        topics.append("asymmetry")
    if float(morphology.get("volume", 0.0)) >= 18000:
        topics.append("biological_implications")
    return list(dict.fromkeys(topics))


def retrieve_local_rag(topics: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    records = [
        record
        for record in (load_jsonl_records(RAG_PATH) + load_jsonl_records(RAG_DISCOVERED_PATH))
        if is_safe_morphology_source(record)
    ]
    if not records:
        return [], topics

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for record in records:
        record_topics = set(record.get("topics", []))
        overlap = len(record_topics.intersection(topics))
        if overlap == 0:
            continue
        priority_bonus = 2 if record.get("priority") == "high" else 1
        scored.append((overlap, priority_bonus, record))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in scored[:5]]
    covered: set[str] = set()
    for record in selected:
        covered.update(set(record.get("topics", [])).intersection(topics))
    missing = [topic for topic in topics if topic not in covered]
    return selected, missing


TOPIC_SEARCH_QUERIES = {
    "surface_roughness": 'cell morphology surface roughness cytoskeleton review',
    "convexity": 'cell shape convexity solidity morphology review',
    "irregular_shape": 'irregular cell shape morphology review',
    "sphericity": 'cell sphericity morphology shape analysis review',
    "eccentricity": 'cell elongation eccentricity morphology review',
    "asymmetry": 'cell shape asymmetry morphology review',
    "morphology_based_anomaly": 'cell morphology shape analysis anomaly review',
    "limitations": 'cell morphology analysis limitations reconstruction review',
    "biological_implications": 'cell shape morphology cytoskeleton biological implications review',
}


def search_europe_pmc(topic: str) -> dict[str, Any] | None:
    query = TOPIC_SEARCH_QUERIES.get(topic)
    if not query:
        return None
    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
        f"{quote_plus(query)}&format=json&pageSize=1&resultType=core"
    )
    try:
        with urlopen(url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    results = payload.get("resultList", {}).get("result", [])
    if not results:
        return None
    result = results[0]
    pmid = result.get("pmid")
    source_id = f"discovered_{topic}_{pmid or result.get('id', 'na')}"
    abstract = (result.get("abstractText") or "").strip()
    if not abstract:
        abstract = f"Relevant literature was found for {topic.replace('_', ' ')} but the abstract text was unavailable in the response."
    fulltext_urls = result.get("fullTextUrlList", {}).get("fullTextUrl", [])
    first_url = fulltext_urls[0].get("url") if fulltext_urls else ""
    doi_url = f"https://doi.org/{result.get('doi')}" if result.get("doi") else ""
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    record = {
        "id": source_id,
        "source_id": source_id,
        "title": result.get("title", f"Literature for {topic}"),
        "source_type": "Europe PMC",
        "url": first_url or doi_url or pubmed_url,
        "priority": "medium",
        "topics": [topic],
        "chunk_type": "discovered_literature",
        "content": abstract[:900],
        "limitations": "This record was auto-discovered at runtime and should be treated as supporting evidence until manually curated.",
        "follow_up": "Use as fallback grounding when the local RAG base lacks topic coverage.",
    }
    if not is_safe_morphology_source(record):
        return None
    return record


def sync_discovered_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = {record.get("id") for record in load_jsonl_records(RAG_DISCOVERED_PATH)}
    synced: list[dict[str, Any]] = []
    for record in records:
        if not is_safe_morphology_source(record):
            continue
        if record.get("id") in existing:
            continue
        append_jsonl_record(RAG_DISCOVERED_PATH, record)
        existing.add(record.get("id"))
        synced.append(record)
    return synced


def build_grounded_explanation(
    classification: dict[str, Any] | None,
    morphology: dict[str, float] | None,
    retrieved: list[dict[str, Any]],
    discovered: list[dict[str, Any]],
) -> str:
    classification = classification or {}
    morphology = morphology or {}
    cls = str(classification.get("class", "Unknown"))
    confidence = float(classification.get("confidence", 0.0))
    cues: list[str] = []
    if float(morphology.get("sphericity", 1.0)) <= 0.84:
        cues.append("reduced spherical regularity")
    if float(morphology.get("convexity", 1.0)) <= 0.965:
        cues.append("loss of a smooth convex envelope")
    if float(morphology.get("eccentricity", 1.0)) <= 0.40:
        cues.append("pronounced elongation or anisotropy")
    if float(morphology.get("surface_roughness", 0.0)) >= 3.0:
        cues.append("elevated surface roughness")
    if float(morphology.get("volume", 0.0)) >= 18000:
        cues.append("increased reconstructed volume")
    cue_text = ", ".join(cues[:-1]) + (" and " + cues[-1] if len(cues) > 1 else cues[0]) if cues else "the combined latent and morphometric signature"
    local_titles = [record.get("title", "") for record in retrieved[:2] if record.get("title")]
    discovered_titles = [record.get("title", "") for record in discovered[:1] if record.get("title")]
    source_text = ""
    if local_titles or discovered_titles:
        joined = "; ".join(local_titles + discovered_titles)
        source_text = f" Grounding was drawn from: {joined}."

    if cls.lower() == "normal":
        return (
            f"Classifier identified this cell as {cls} with {confidence * 100:.1f}% confidence. "
            f"The reconstructed morphology remains comparatively regular, and the decision is supported by {cue_text}. "
            f"This remains a morphology-based assessment rather than a standalone biological conclusion.{source_text}"
        )

    return (
        f"Classifier identified this cell as {cls} with {confidence * 100:.1f}% confidence. "
        f"The anomaly decision is driven by {cue_text}, which makes the reconstructed cell depart from a smoother reference morphology. "
        f"This interpretation is grounded in reconstructed 3D shape evidence and morphology literature, but it should not be treated as a diagnosis or evidence of a specific biological mechanism.{source_text}"
    )


def unwrap_state_dict(checkpoint):
    return checkpoint.get("model_state_dict", checkpoint)


def infer_view_names(state_dict):
    in_channels = infer_in_channels_from_state_dict(state_dict)
    return get_view_names("quad" if in_channels == 4 else "tri")


def infer_latent_dim(state_dict):
    for key in state_dict.keys():
        if "encoder.fc" in key and "weight" in key:
            return int(state_dict[key].shape[0])
    if "fc_mu.weight" in state_dict:
        return int(state_dict["fc_mu.weight"].shape[0])
    return 256


def load_input_views(filename: str, view_names: tuple[str, ...]) -> np.ndarray:
    projections = []
    for view_name in view_names:
        path = os.path.join(DATA_DIR, view_name, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        projections.append(np.load(path))
    return np.stack(projections, axis=0).astype(np.float32)


@torch.no_grad()
def tta_predict(model, inputs):
    pred_original = model(inputs)
    inputs_flipped = inputs.flip(-1)
    pred_flipped = model(inputs_flipped)
    pred_unflipped = pred_flipped.flip(-1)
    return (pred_original + pred_unflipped) / 2.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/status")
def get_status():
    return {
        "cnn_loaded": model is not None,
        "vae_loaded": vae_model is not None,
        "refiner_loaded": refiner_model is not None,
    }


@app.on_event("startup")
def load_resources():
    global model, vae_model, refiner_model, classifier_model, model_view_names, vae_view_names
    if os.path.exists(MODEL_PATH):
        print(f"Загрузка CNN модели на {device}...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = unwrap_state_dict(checkpoint)
        model_view_names = infer_view_names(state_dict)
        latent_dim = infer_latent_dim(state_dict)
        in_channels = infer_in_channels_from_state_dict(state_dict)
        skip_channels = infer_skip_channels_from_state_dict(state_dict)

        model = TriViewAutoencoder(
            latent_dim=latent_dim, in_channels=in_channels, skip_channels=skip_channels
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        print("CNN модель загружена.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: CNN модель не найдена.")

    if model is not None and os.path.exists(REFINER_PATH):
        print(f"Загрузка refiner модели на {device}...")
        refiner_model = DetailRefiner(view_channels=len(model_view_names)).to(device)
        refiner_model.load_state_dict(torch.load(REFINER_PATH, map_location=device))
        refiner_model.eval()
        print("Refiner модель загружена.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Refiner модель не найдена.")

    if os.path.exists(CLASSIFIER_PATH):
        print(f"Загрузка Classifier модели на {device}...")
        try:
            # Latent (256) + 5 morphometrics = 261
            classifier_model = LatentClassifier(latent_dim=261).to(device)
            classifier_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
            classifier_model.eval()
            print("Classifier модель загружена.")
        except Exception as e:
            print(f"Ошибка загрузки Classifier: {e}")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Classifier модель не найдена.")

    if os.path.exists(VAE_MODEL_PATH):
        print(f"Загрузка VAE модели на {device}...")
        checkpoint = torch.load(VAE_MODEL_PATH, map_location=device)
        state_dict = unwrap_state_dict(checkpoint)
        vae_view_names = infer_view_names(state_dict)
        latent_dim = infer_latent_dim(state_dict)
        in_channels = infer_in_channels_from_state_dict(state_dict)
        skip_channels = infer_skip_channels_from_state_dict(state_dict)
        vae_model = TriViewCVAE(
            latent_dim=latent_dim, in_channels=in_channels, skip_channels=skip_channels
        ).to(device)
        vae_model.load_state_dict(state_dict)
        vae_model.eval()
        print("VAE модель загружена.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: VAE модель не найдена (обучите через train_vae.py).")

    # LLM availability check
    if llm_available():
        print("✓ Gemini LLM доступен")
    else:
        print("⚠ Gemini LLM недоступен — будет использован шаблонный fallback")


@app.get("/api/cells")
def get_cells():
    if not os.path.exists(DATA_DIR):
        return {"cells": []}

    files = [f for f in os.listdir(os.path.join(DATA_DIR, "top_proj")) if f.endswith(".npy")]
    result = []
    for f in sorted(files):
        parts = f.replace('.npy', '').split('_')
        score = parts[0]
        c_type = "_".join(parts[1:])
        result.append({
            "filename": f,
            "score": score,
            "type": c_type
        })
    return {"cells": result}


def extract_mesh(volume, level=0.5):
    try:
        verts, faces, normals, values = marching_cubes(volume, level=level)
        return {
            "vertices": verts.flatten().tolist(),
            "indices": faces.flatten().tolist()
        }
    except ValueError:
        return None


def compute_overlap_metrics(pred_vol: np.ndarray, gt_vol: np.ndarray) -> dict:
    pred_b = (pred_vol > 0.5).astype(np.float32)
    gt_b = (gt_vol > 0.5).astype(np.float32)

    tp = float(np.sum(pred_b * gt_b))
    fp = float(np.sum(pred_b * (1.0 - gt_b)))
    fn = float(np.sum((1.0 - pred_b) * gt_b))

    dice = (2.0 * tp + 1.0) / (2.0 * tp + fp + fn + 1.0)
    iou = (tp + 1.0) / (tp + fp + fn + 1.0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    pred_voxels = float(np.sum(pred_b))
    gt_voxels = float(np.sum(gt_b))
    volume_diff_pct = (
        abs(pred_voxels - gt_voxels) / (gt_voxels + 1e-8) * 100.0
    )

    return {
        "dice": round(float(dice), 4),
        "iou": round(float(iou), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "volume_diff_pct": round(float(volume_diff_pct), 2),
    }


def compute_surface_similarity(pred_mesh: dict | None, gt_mesh: dict | None) -> dict | None:
    if not pred_mesh or not gt_mesh:
        return None

    pred_v = np.array(pred_mesh["vertices"], dtype=np.float32).reshape(-1, 3)
    gt_v = np.array(gt_mesh["vertices"], dtype=np.float32).reshape(-1, 3)

    if len(pred_v) < 10 or len(gt_v) < 10:
        return None

    max_points = 10000
    if len(pred_v) > max_points:
        idx = np.random.choice(len(pred_v), max_points, replace=False)
        pred_v = pred_v[idx]
    if len(gt_v) > max_points:
        idx = np.random.choice(len(gt_v), max_points, replace=False)
        gt_v = gt_v[idx]

    tree_pred = cKDTree(pred_v)
    tree_gt = cKDTree(gt_v)

    d_gt_to_pred, _ = tree_pred.query(gt_v, k=1)
    d_pred_to_gt, _ = tree_gt.query(pred_v, k=1)

    assd = float((d_gt_to_pred.mean() + d_pred_to_gt.mean()) / 2.0)
    hd95 = float(
        max(
            np.percentile(d_gt_to_pred, 95),
            np.percentile(d_pred_to_gt, 95),
        )
    )

    surface_similarity = float(1.0 / (1.0 + assd))

    return {
        "surface_assd": round(assd, 4),
        "surface_hd95": round(hd95, 4),
        "surface_similarity": round(float(surface_similarity), 4),
    }


from scipy.ndimage import binary_dilation, map_coordinates


def compute_diff(pred_vol: np.ndarray, gt_vol: np.ndarray, pred_mesh: dict | None) -> dict:
    pred_b = (pred_vol > 0.5).astype(np.float32)
    gt_b = (gt_vol > 0.5).astype(np.float32)
    fn_vol = (gt_b * (1.0 - pred_b)).astype(np.float32)

    fp_vertex_colors = None
    if pred_mesh is not None:
        verts = np.array(pred_mesh["vertices"], dtype=np.float32).reshape(-1, 3)
        n = len(verts)
        colors = np.ones((n, 3), dtype=np.float32) * [0.58, 0.62, 0.70]
        gt_coords = verts.T
        gt_sampled = map_coordinates(gt_b.astype(np.float64), gt_coords, order=1, mode='constant', cval=0.0)
        for i in range(n):
            if gt_sampled[i] < 0.5:
                colors[i] = [0.85, 0.45, 0.45]
        fp_vertex_colors = colors.flatten().tolist()

    fn_mesh = None
    if fn_vol.sum() > 0:
        fn_dilated = binary_dilation(fn_vol, iterations=2).astype(np.float32)
        fn_smooth = fn_dilated.copy()
        for _ in range(3):
            padded = np.pad(fn_smooth, 1, mode='constant')
            fn_smooth = (
                padded[:-2, 1:-1, 1:-1] + padded[2:, 1:-1, 1:-1] +
                padded[1:-1, :-2, 1:-1] + padded[1:-1, 2:, 1:-1] +
                padded[1:-1, 1:-1, :-2] + padded[1:-1, 1:-1, 2:] +
                padded[1:-1, 1:-1, 1:-1] * 2
            ) / 8.0
        fn_mesh = extract_mesh(fn_smooth, level=0.5)

    return {"fp_vertex_colors": fp_vertex_colors, "fn": fn_mesh}


@app.post("/api/predict/{filename}")
def predict(filename: str):
    if not model:
        raise HTTPException(status_code=500, detail="Модель не загружена.")

    try:
        multi_view_input = load_input_views(filename, model_view_names)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файлы проекций не найдены.")

    tri_tensor = torch.tensor(multi_view_input, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        coarse_logits = tta_predict(model, tri_tensor)
        coarse_tensor = torch.sigmoid(coarse_logits)
        coarse_vol = coarse_tensor[0, 0].cpu().numpy()

        pred_tensor = coarse_tensor
        if refiner_model is not None:
            lifted_views = lift_views_to_volume(tri_tensor, model_view_names)
            refined_logits = refiner_model(coarse_logits, lifted_views)
            pred_tensor = torch.sigmoid(refined_logits)

        pred_vol = pred_tensor[0, 0].cpu().numpy()
        reprojection_l1 = float(torch.abs(project_volume_batch(pred_tensor, model_view_names) - tri_tensor).mean().item())

        classification_result = None
        morphometrics_result = extract_all_metrics(pred_vol, threshold=0.5)

        if classifier_model is not None:
            z = model.encode(tri_tensor)
            MORPHO_KEYS = ["volume", "sphericity", "convexity", "eccentricity", "surface_roughness"]
            morpho_list = [[morphometrics_result[k] for k in MORPHO_KEYS]]
            morpho_tensor = torch.tensor(morpho_list, dtype=torch.float32, device=device)
            combined = torch.cat([z, morpho_tensor], dim=1)
            logits = classifier_model(combined)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = int(logits.argmax(1).item())
            classification_result = {
                "class": "Anomaly" if pred_class == 1 else "Normal",
                "confidence": float(probs[pred_class].item())
            }

    pred_mesh = extract_mesh(pred_vol)
    coarse_mesh = extract_mesh(coarse_vol) if refiner_model is not None else None

    gt_mesh = None
    diff = {}
    gt_path = os.path.join(DATA_DIR, "obj", filename)
    if os.path.exists(gt_path):
        gt_vol = np.load(gt_path)
        gt_mesh = extract_mesh(gt_vol)
        diff = compute_diff(pred_vol, gt_vol, pred_mesh)

    overlap_metrics = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "volume_diff_pct": 0.0,
    }
    surface_metrics = None
    if os.path.exists(gt_path):
        overlap_metrics = compute_overlap_metrics(pred_vol, gt_vol)
        surface_metrics = compute_surface_similarity(pred_mesh, gt_mesh)

    return {
        "dice": overlap_metrics["dice"],
        "metrics": {
            **overlap_metrics,
            **(surface_metrics or {}),
            "reprojection_l1": round(reprojection_l1, 4),
        },
        "morphology": morphometrics_result,
        "classification": classification_result,
        "pred": pred_mesh,
        "coarse": coarse_mesh,
        "gt": gt_mesh,
        "diff": diff,
    }

from PIL import Image
import io
import base64


def numpy_to_b64_png(arr):
    arr = np.nan_to_num(arr)

    min_val = arr.min()
    max_val = arr.max()
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val) * 255.0
    else:
        arr = np.zeros_like(arr)

    arr_uint8 = arr.astype(np.uint8)

    if arr_uint8.ndim == 3 and arr_uint8.shape[0] == 1:
        arr_uint8 = arr_uint8[0]

    img = Image.fromarray(arr_uint8, mode='L')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@app.get("/api/preview/{filename}")
def preview_projections(filename: str):
    try:
        response = {}
        for short_name, view_name in [
            ("top", "top_proj"),
            ("bottom", "bottom_proj"),
            ("side", "side_proj"),
            ("front", "front_proj"),
        ]:
            path = os.path.join(DATA_DIR, view_name, filename)
            if os.path.exists(path):
                response[short_name] = numpy_to_b64_png(np.load(path))
        if not response:
            raise FileNotFoundError(filename)
        return response
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Не удалось загрузить проекции: {str(e)}")


@app.post("/api/agent/retrieve")
def agent_retrieve(request: AgentRequest):
    topics = morphology_topics(request.morphology, request.classification)
    chunks, missing_topics = retrieve_local_rag(topics)
    confidence = float((request.classification or {}).get("confidence", 0.0))
    is_anomaly = str((request.classification or {}).get("class", "")).lower() == "anomaly"
    used_fallback = bool(is_anomaly and (confidence < 0.9 or len(missing_topics) > 0))
    return {
        "topics": topics,
        "chunks": chunks,
        "missing_topics": missing_topics,
        "used_fallback": used_fallback,
        "coverage": "partial" if used_fallback else "strong",
    }


@app.post("/api/agent/search")
def agent_search(request: AgentRequest):
    missing_topics = request.missing_topics or []
    discovered: list[dict[str, Any]] = []
    for topic in missing_topics[:3]:
        record = search_europe_pmc(topic)
        if record is not None:
            discovered.append(record)

    synced = sync_discovered_records(discovered)
    notice = None
    if synced:
        labels = ", ".join(record.get("title", "new literature") for record in synced[:2])
        suffix = f" +{len(synced) - 2} more" if len(synced) > 2 else ""
        notice = f"Agent found new knowledge entries: {labels}{suffix}. They were added to the local knowledge base."

    return {
        "discovered": discovered,
        "synced": synced,
        "notice": notice,
    }


@app.post("/api/agent/generate")
def agent_generate(request: AgentRequest):
    """Writer agent: structured morphology report via Gemini."""
    all_chunks = [
        chunk for chunk in ((request.retrieved or []) + (request.discovered or []))
        if is_safe_morphology_source(chunk)
    ]
    print(f"[agent/generate] chunks={len(all_chunks)}, llm_available={llm_available()}")
    report = generate_report(
        classification=request.classification or {},
        morphology=request.morphology or {},
        metrics=request.metrics or {},
        chunks=all_chunks,
        cell_type=request.cell_type or "",
    )
    if report is None:
        print("[agent/generate] LLM returned None → using fallback")
        explanation = build_grounded_explanation(
            request.classification, request.morphology,
            request.retrieved or [], request.discovered or [],
        )
        return {"report": None, "fallback_explanation": explanation, "llm_used": False}
    print(f"[agent/generate] LLM report generated, keys={list(report.keys())}")
    return {"report": report, "fallback_explanation": None, "llm_used": True}


@app.post("/api/agent/verify")
def agent_verify(request: AgentRequest):
    """Verifier agent: validate draft report against RAG sources."""
    all_chunks = [
        chunk for chunk in ((request.retrieved or []) + (request.discovered or []))
        if is_safe_morphology_source(chunk)
    ]
    result = verify_report(
        draft=request.draft_report or {},
        chunks=all_chunks,
        morphology=request.morphology or {},
    )
    if result is None:
        return {"report": request.draft_report, "corrections": [], "verified": False}
    corrections = result.pop("corrections", [])
    return {"report": result, "corrections": corrections, "verified": True}


@app.post("/api/agent/answer")
def agent_answer(request: AgentRequest):
    """Legacy fallback — template-based answer."""
    retrieved = [r for r in (request.retrieved or []) if is_safe_morphology_source(r)]
    discovered = [r for r in (request.discovered or []) if is_safe_morphology_source(r)]
    explanation = build_grounded_explanation(
        request.classification, request.morphology, retrieved, discovered,
    )
    return {
        "explanation": explanation,
        "references": [
            {"title": r.get("title"), "url": r.get("url"), "source_type": r.get("source_type")}
            for r in (retrieved + discovered)[:5]
        ],
    }


@app.get("/api/metrics")
def get_metrics():
    history_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics", "reconstruction_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    return {"error": "History file not found."}


@app.post("/api/predict-vae/{filename}")
def predict_vae(filename: str):
    if not vae_model:
        raise HTTPException(status_code=500, detail="VAE модель не загружена. Обучите через train_vae.py.")

    try:
        multi_view_input = load_input_views(filename, vae_view_names)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файлы проекций не найдены.")

    tri_tensor = torch.tensor(multi_view_input, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_logits, best_score = best_of_k_generate(vae_model, tri_tensor, vae_view_names, num_samples=8)
        pred_probs = torch.sigmoid(pred_logits)
        pred_vol = pred_probs[0, 0].cpu().numpy()

    pred_mesh = extract_mesh(pred_vol)

    gt_mesh = None
    diff = {}
    gt_path = os.path.join(DATA_DIR, "obj", filename)
    overlap_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "volume_diff_pct": 0.0}
    surface_metrics = None

    if os.path.exists(gt_path):
        gt_vol = np.load(gt_path)
        gt_mesh = extract_mesh(gt_vol)
        overlap_metrics = compute_overlap_metrics(pred_vol, gt_vol)
        surface_metrics = compute_surface_similarity(pred_mesh, gt_mesh)
        diff = compute_diff(pred_vol, gt_vol, pred_mesh)

    return {
        "dice": overlap_metrics["dice"],
        "metrics": {**overlap_metrics, **(surface_metrics or {}), "reprojection_l1": round(float(best_score.item()), 4)},
        "pred": pred_mesh,
        "gt": gt_mesh,
        "diff": diff,
    }


@app.get("/api/metrics-vae")
def get_vae_metrics():
    history_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "metrics", "vae_history.json",
    )
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    return {"error": "VAE history file not found."}


@app.post("/api/generate-custom")
async def generate_custom(
    top: UploadFile = File(...),
    bottom: UploadFile = File(...),
    side: UploadFile = File(...)
):
    """
    Генерирует 3D модель по трем загруженным фотографиям.
    Поддерживает и CNN, и CVAE реконструкцию.
    """
    if not model:
        raise HTTPException(status_code=500, detail="CNN модель не загружена.")

    import io
    from PIL import Image

    async def process_image(file: UploadFile):
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("L")
        img = img.resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    try:
        top_arr = await process_image(top)
        bottom_arr = await process_image(bottom)
        side_arr = await process_image(side)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки изображений: {e}")

    # Собираем тензор [1, 3, 64, 64]
    input_arr = np.stack([top_arr, bottom_arr, side_arr], axis=0)
    tri_tensor = torch.tensor(input_arr, dtype=torch.float32).unsqueeze(0).to(device)

    # 1. CNN + Refiner Prediction
    cnn_mesh = None
    with torch.no_grad():
        coarse_logits = tta_predict(model, tri_tensor)
        if refiner_model is not None:
            lifted_views = lift_views_to_volume(tri_tensor, model_view_names)
            refined_logits = refiner_model(coarse_logits, lifted_views)
            pred_tensor = torch.sigmoid(refined_logits)
        else:
            pred_tensor = torch.sigmoid(coarse_logits)
        
        pred_vol = pred_tensor[0, 0].cpu().numpy()
        cnn_mesh = extract_mesh(pred_vol)

    # 2. CVAE Prediction
    vae_mesh = None
    if vae_model is not None:
        with torch.no_grad():
            from src.vae import best_of_k_generate
            pred_logits, _ = best_of_k_generate(vae_model, tri_tensor, vae_view_names, num_samples=8)
            out_vol = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()
            vae_mesh = extract_mesh(out_vol)

    return {
        "cnn_mesh": cnn_mesh,
        "vae_mesh": vae_mesh
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
