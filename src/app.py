import os
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import sys

# Добавляем корень проекта в sys.path для импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoencoder import TriViewAutoencoder

# ======= НАСТРОЙКИ СТРАНИЦЫ =======
st.set_page_config(page_title="3D Cell Reconstruction", page_icon="🧬", layout="wide")

st.title("🧬 Multi-View 3D Cell Shape Reconstruction")
st.markdown("Пошаговая демонстрация: от 2D снимков до интерактивной 3D-модели (Autoencoder).")

# ======= ЗАГРУЗКА ДАННЫХ =======
DATA_DIR = "data/processed"
MODEL_PATH = "results/best_autoencoder.pt"

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = TriViewAutoencoder(latent_dim=256).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

model, device = get_model()

if not os.path.exists(DATA_DIR):
    st.error("Папка data/processed не найдена.")
    st.stop()

if not model:
    st.warning(f"Файл модели {MODEL_PATH} не найден. Проведите обучение.")
    st.stop()

# Получаем список файлов
all_files = [f for f in os.listdir(os.path.join(DATA_DIR, "top_proj")) if f.endswith(".npy")]
if not all_files:
    st.error("Нет подготовленных npy-файлов проекций.")
    st.stop()

# ======= ШАГ 0: ВЫБОР КЛЕТКИ =======
st.sidebar.header("🕹 Управление")
selected_file = st.sidebar.selectbox("Выберите клетку для анализа:", sorted(all_files))

cell_score = selected_file.split('_')[0]
cell_type = "_".join(selected_file.split('_')[1:]).replace('.npy', '')

st.sidebar.markdown(f"**Тип:** {cell_type}")
st.sidebar.markdown(f"**Score:** {cell_score}")

# ======= ШАГ 1: ВХОДНЫЕ ПРОЕКЦИИ =======
st.header("Шаг 1: Извлечение 2D проекций")
st.markdown("Микроскоп снимает слои клетки. Мы разбиваем этот стек на три плоские картинки (Sum Projection), чтобы ИИ мог увидеть клетку с разных физических сторон.")

def load_proj(folder, fname):
    return np.load(os.path.join(DATA_DIR, folder, fname))

def apply_hot_cmap(img_2d):
    """Превращает 2D массив 0-1 в цветную термо-картинку RGB (как на графиках)."""
    # Нормализуем на всякий случай, если есть выбивающиеся значения
    img_norm = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
    cmap = plt.get_cmap("hot")
    rgba_img = cmap(img_norm) # [H, W, 4]
    return rgba_img

top_img = load_proj("top_proj", selected_file)
bottom_img = load_proj("bottom_proj", selected_file)
side_img = load_proj("side_proj", selected_file)

col1, col2, col3 = st.columns(3)
with col1:
    st.image(apply_hot_cmap(top_img), caption="Top (Верхняя полусфера)", use_container_width=True, clamp=True)
with col2:
    st.image(apply_hot_cmap(bottom_img), caption="Bottom (Нижняя полусфера)", use_container_width=True, clamp=True)
with col3:
    st.image(apply_hot_cmap(side_img), caption="Side (Боковой профиль)", use_container_width=True, clamp=True)

# ======= ШАГ 2: ИНФЕРЕНС (ПОСТРОЕНИЕ) =======
st.header("Шаг 2 и 3: Построение 3D-модели нейросетью")
st.markdown("Подаём эти 3 картинки в наш Tri-View Autoencoder. Нейросеть предсказывает 3D-куб размером 64x64x64.")

if st.button("🚀 Сгенерировать 3D-модель (Inference)", type="primary"):
    with st.spinner("Нейросеть генерирует 3D воксели..."):
        # Подготовка тензора
        tri_input = np.stack([top_img, bottom_img, side_img], axis=0) # [3, 64, 64]
        # Используем torch.tensor(..., dtype) вместо from_numpy, чтобы избежать проблем с типами
        tri_tensor = torch.tensor(tri_input, dtype=torch.float32).unsqueeze(0).to(device) # [1, 3, 64, 64]
        
        with torch.no_grad():
            pred = model(tri_tensor)[0, 0].cpu().numpy() # [64, 64, 64]
        
        # Загрузка Ground Truth для сравнения
        gt_path = os.path.join(DATA_DIR, "obj", selected_file)
        if os.path.exists(gt_path):
            gt = np.load(gt_path)
            
            # Подсчёт Dice на лету
            pred_bin = (pred > 0.5).astype(np.float32)
            gt_bin = (gt > 0.5).astype(np.float32)
            intersection = np.sum(pred_bin * gt_bin)
            union = np.sum(pred_bin) + np.sum(gt_bin)
            dice = (2.0 * intersection + 1) / (union + 1)
            st.success(f"Генерация завершена за доли секунды! **Точность (Dice) с оригиналом: {dice:.3f}**")
        else:
            st.success("Генерация завершена!")

        # Отрисовка интерактивных 3D Plotly (Слева Предсказание, Справа GT)
        col_pred, col_gt = st.columns(2)
        
        def plot_voxel(volume, title, color_scale='Viridis'):
            z, y, x = np.where(volume > 0.5)
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=3, color=z, colorscale=color_scale, opacity=0.8)
            )])
            fig.update_layout(
                title=title,
                margin=dict(l=0, r=0, b=0, t=30),
                scene=dict(
                    xaxis=dict(range=[0, 64], title="X"),
                    yaxis=dict(range=[0, 64], title="Y"),
                    zaxis=dict(range=[0, 64], title="Z"),
                    aspectmode='cube'
                )
            )
            return fig

        with col_pred:
            st.plotly_chart(plot_voxel(pred, "Pred: Сгенерированная 3D Модель", 'Plasma'), use_container_width=True)
            
        with col_gt:
            if os.path.exists(gt_path):
                st.plotly_chart(plot_voxel(gt, "GT: Оригинальная 3D Модель (Ground Truth)", 'Viridis'), use_container_width=True)
            else:
                st.info("Ground Truth отсутствует.")
