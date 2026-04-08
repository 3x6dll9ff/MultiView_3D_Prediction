# MultiView 3D Cell Shape Prediction & Classification

Пайплайн предсказания 3D-формы клетки из трёх 2D-проекций (top, bottom, side)
с последующей морфологической классификацией (норма / аномалия).

## Модель

### Архитектура: Tri-View Convolutional Autoencoder

```
Input [3, 64, 64]                     Output [1, 64, 64, 64]
 (top + bottom + side)                 (predicted 3D shape)
        │                                      ▲
        ▼                                      │
┌─── 2D Encoder (CNN) ───┐    ┌─── 3D Decoder (CNN) ────────┐
│                         │    │                              │
│ Conv2d  3→32   /2  64→32│    │ ConvT3d 256→128  ×2  4→8    │
│ Conv2d  32→64  /2  32→16│    │ ConvT3d 128→64   ×2  8→16   │
│ Conv2d  64→128 /2  16→8 │    │ ConvT3d 64→32    ×2  16→32  │
│ Conv2d  128→256/2  8→4  │    │ ConvT3d 32→1     ×2  32→64  │
│ Flatten → FC → 256      │    │ FC 256→16384 → reshape       │
└────────────┬────────────┘    └──────────────┬───────────────┘
             │                                │
             └──── latent vector (256) ───────┘
```

Каждый свёрточный блок: `Conv → BatchNorm → ReLU`. Декодер завершается `Sigmoid`.

### Параметры модели

| Параметр | Значение |
|----------|----------|
| Общее кол-во параметров | **8,404,129** |
| Encoder | 1,438,208 (17%) |
| Decoder | 6,965,921 (83%) |
| Размер файла весов | 32 MB |
| Latent dimension | 256 |
| Input shape | `[batch, 3, 64, 64]` |
| Output shape | `[batch, 1, 64, 64, 64]` |

### Вход / Выход

| Что | Формат | Описание |
|-----|--------|----------|
| **Input** | `[3, 64, 64]` float32 | 3 канала: top projection, bottom projection, side projection. Каждый канал — sum projection из 3D ground truth. Значения нормализованы в `[0, 1]` |
| **Output** | `[1, 64, 64, 64]` float32 | Бинарный 3D voxel grid. Значения в `[0, 1]` (Sigmoid). Порог бинаризации: `> 0.5` |
| **Latent** | `[256]` float32 | Сжатое представление 3D-формы. Используется для классификации |

### Обучение

| Параметр | Значение |
|----------|----------|
| Loss | `0.5 × BCE + 0.5 × Dice` |
| Optimizer | Adam, lr=1e-3 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Epochs | 50 |
| Batch size | 8 |
| Train/Test split | 80/20, seed=42 |

### Результаты

| Метрика | Значение |
|---------|----------|
| **Dice score** | **0.900** |
| **IoU** | **0.825** |
| Train loss (final) | 0.048 |
| Test loss (final) | 0.106 |

## Данные

### SHAPR Dataset

- **Источник**: [Zenodo — SHAPR](https://zenodo.org/records/7031924) (3.6 ГБ)
- **Клеток**: 602 RBC (Red Blood Cells)
- **Типы**: discocyte (norm), stomatocyte, echinocyte, spherocyte
- **Paper**: [SHAPR (iScience 2022)](https://doi.org/10.1016/j.isci.2022.104523)

### Проекции (Input)

Из каждого 3D ground truth (`obj/*.npy`, shape `64×64×64`) генерируются 3 проекции:

| Проекция | Как создаётся | Что показывает |
|----------|--------------|---------------|
| **Top** | `sum(voxels[:32], axis=0)` | Вид сверху (верхняя половина Z) |
| **Bottom** | `sum(voxels[32:], axis=0)` | Вид снизу (нижняя половина Z) |
| **Side** | `sum(voxels, axis=1)` | Профиль сбоку (сумма по Y) |

Каждая проекция нормализуется в `[0, 1]`.

## Быстрый старт

### Docker (рекомендуется)

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

### Локально (Python)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Скачать и подготовить данные
python3 src/download_data.py
python3 src/prepare_dataset.py

# Обучение (GPU рекомендуется)
python3 src/train_reconstruction.py --epochs 50

# Запуск API
python3 src/api.py
```

### Google Colab

Используй `notebooks/train_colab.ipynb`.

## API

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Health check |
| `/api/cells` | GET | Список всех клеток (602 шт) |
| `/api/preview/{filename}` | GET | 3 проекции как base64 PNG |
| `/api/predict/{filename}` | POST | Инференс: 3D mesh + метрики (Dice, IoU, ASSD, HD95) |
| `/api/metrics` | GET | История обучения (loss, dice, iou по эпохам) |

### Пример ответа `/api/predict`

```json
{
  "dice": 0.92,
  "metrics": {
    "dice": 0.92,
    "iou": 0.86,
    "precision": 0.94,
    "recall": 0.90,
    "volume_diff_pct": 3.2,
    "surface_assd": 0.85,
    "surface_hd95": 2.1,
    "surface_similarity": 0.54
  },
  "pred": { "vertices": [...], "indices": [...] },
  "gt": { "vertices": [...], "indices": [...] }
}
```

## Структура проекта

```
src/                  — Python-код (12 файлов)
frontend/             — Web-приложение (React + Vite + Three.js)
data/                 — Данные (не в Git)
results/              — Модель + метрики + графики
notebooks/            — Google Colab ноутбук
docs/                 — Документация
```

Подробная структура: [`docs/project_structure.md`](docs/project_structure.md)

## Стек

| Компонент | Технологии |
|-----------|-----------|
| ML | PyTorch, NumPy, SciPy, scikit-image |
| API | FastAPI, Uvicorn |
| Frontend | React, TypeScript, Vite, React Three Fiber, TailwindCSS |
| Deploy | Docker Compose |
| Training | Google Colab (T4 GPU) |
