# Структура проекта

## Файлы src/

| Файл | Назначение | Статус |
|------|-----------|--------|
| `download_data.py` | Скачивание SHAPR с Zenodo | ✅ Готов |
| `prepare_dataset.py` | Подготовка tri-view пар + метки из имён файлов | ✅ Готов (602 RBC) |
| `dataset.py` | PyTorch Dataset (tri-view + single-view baseline) | ✅ Готов |
| `autoencoder.py` | Tri-View Autoencoder (2D encoder + 3D decoder) | ✅ Готов |
| `classifier.py` | MLP на latent + Random Forest на метриках | ✅ Готов |
| `morphometrics.py` | 6 морфометрических метрик из 3D | ✅ Готов |
| `train_reconstruction.py` | Training loop для autoencoder | ✅ Готов |
| `train_classifier.py` | Training loop для классификатора | ✅ Готов |
| `evaluate.py` | Метрики: Dice, IoU, accuracy, AUROC, графики | ✅ Готов |
| `visualize.py` | Plotly 3D, срезы, сравнение predicted vs GT | ✅ Готов |
| `app.py` | Интерактивный дашборд Streamlit для защиты | ✅ Готов |
| `Train_Colab.ipynb` | Ноутбук для обучения модели в Google Colab | ✅ Готов |

## Данные

```
data/
├── raw/
│   └── shapr/
│       └── dataset_for_3D_reconstruction/
│           ├── image/   — 825 файлов .tif (2D, 64×64, ~4.4 KB)
│           ├── mask/    — 825 файлов .tif (2D маски)
│           ├── obj/     — 825 файлов .tif (3D, 64×64×64, ~273 KB)
│           ├── Cylinder_fit/
│           └── Ellipse_fit/
└── processed/           — подготовленные .npy тензоры + metadata.csv
```

## Пайплайн запуска

```bash
# 1. Скачать данные
python3 src/download_data.py

# 2. Подготовить датасет
python3 src/prepare_dataset.py

# 3. Обучить autoencoder (на Colab с GPU)
python3 src/train_reconstruction.py --epochs 50

# 4. Обучить классификатор
python3 src/train_classifier.py --mode rf
python3 src/train_classifier.py --mode latent --autoencoder results/best_autoencoder.pt

# 5. Интерактивная демонстрация (Web App)
streamlit run src/app.py

# 6. Оценка метрик
python3 src/evaluate.py
```
