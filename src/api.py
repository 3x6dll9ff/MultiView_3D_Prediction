import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from skimage.measure import marching_cubes
import sys

# Добавляем корень проекта для правильных импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoencoder import TriViewAutoencoder

app = FastAPI(title="3D Cell Reconstruction API")

# Настройка CORS для свободного общения с фронтендом на React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data/processed"
MODEL_PATH = "results/best_autoencoder.pt"

model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

@app.on_event("startup")
def load_resources():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Загрузка модели на {device}...")
        model = TriViewAutoencoder(latent_dim=256).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Готово!")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Файл модели не найден.")

@app.get("/api/cells")
def get_cells():
    """Получает список всех клеток и парсит их метаданные из имени."""
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
    """Превращает воксели в эффективный 3D-меш для передачи в браузер."""
    try:
        verts, faces, normals, values = marching_cubes(volume, level=level)
        # Для React Three Fiber нам нужен плоский массив координат
        # vertices: [x1,y1,z1, x2,y2,z2...]  и индексы граней
        return {
            "vertices": verts.flatten().tolist(),
            "indices": faces.flatten().tolist()
        }
    except ValueError:
        return None

@app.post("/api/predict/{filename}")
def predict(filename: str):
    """Принимает имя файла, прогоняет через PyTorch автоэнкодер и отдает обратно две 3D-модели."""
    if not model:
        raise HTTPException(status_code=500, detail="Модель не загружена.")
        
    try:
        top = np.load(os.path.join(DATA_DIR, "top_proj", filename))
        bottom = np.load(os.path.join(DATA_DIR, "bottom_proj", filename))
        side = np.load(os.path.join(DATA_DIR, "side_proj", filename))
    except Exception as e:
        raise HTTPException(status_code=404, detail="Файлы проекций не найдены.")
        
    # Инференс (предсказание 3D формы по 3-м плоским фото)
    tri_input = np.stack([top, bottom, side], axis=0)
    tri_tensor = torch.tensor(tri_input, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_vol = model(tri_tensor)[0, 0].cpu().numpy()
        
    pred_mesh = extract_mesh(pred_vol)
    
    # Чтение Ground Truth для сравнения
    gt_mesh = None
    gt_path = os.path.join(DATA_DIR, "obj", filename)
    if os.path.exists(gt_path):
        gt_vol = np.load(gt_path)
        gt_mesh = extract_mesh(gt_vol)
        
    # Вычисление Dice на лету
    dice = 0.0
    if os.path.exists(gt_path):
        pred_b = (pred_vol > 0.5).astype(np.float32)
        gt_b = (gt_vol > 0.5).astype(np.float32)
        inter = np.sum(pred_b * gt_b)
        union = np.sum(pred_b) + np.sum(gt_b)
        dice = (2.0 * inter + 1) / (union + 1)
        
    return {
        "dice": round(dice, 4),
        "pred": pred_mesh,
        "gt": gt_mesh
    }

import json

from PIL import Image
import io
import base64

def numpy_to_b64_png(arr):
    # Убираем NaN
    arr = np.nan_to_num(arr)
    
    # Автоконтраст: растягиваем гистограмму на полный диапазон [0, 255]
    min_val = arr.min()
    max_val = arr.max()
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val) * 255.0
    else:
        arr = np.zeros_like(arr)
    
    # Конвертируем в uint8
    arr_uint8 = arr.astype(np.uint8)
    
    # Для SHAPR это скорее всего одноканальные 64x64. Убедимся, что размерность [64,64]
    if arr_uint8.ndim == 3 and arr_uint8.shape[0] == 1:
        arr_uint8 = arr_uint8[0]
        
    img = Image.fromarray(arr_uint8, mode='L')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

@app.get("/api/preview/{filename}")
def preview_projections(filename: str):
    """Возвращает 3 проекции в виде base64 png для предпросмотра на UI."""
    try:
        top = np.load(os.path.join(DATA_DIR, "top_proj", filename))
        bottom = np.load(os.path.join(DATA_DIR, "bottom_proj", filename))
        side = np.load(os.path.join(DATA_DIR, "side_proj", filename))
        
        return {
            "top": numpy_to_b64_png(top),
            "bottom": numpy_to_b64_png(bottom),
            "side": numpy_to_b64_png(side)
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Не удалось загрузить проекции: {str(e)}")

@app.get("/api/metrics")
def get_metrics():
    """Возвращает историю лоссов и метрик из JSON для вкладки Metrics."""
    history_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics", "reconstruction_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    return {"error": "History file not found."}

# Точка входа для запуска через `python src/api.py`
if __name__ == "__main__":
    import uvicorn
    # Запускаем локальный веб-сервер на 8000 порту
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
