import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé - THÈME CLAIR
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    
    /* Variables CSS - THEME CLAIR */
    :root {
        --primary: #0284c7;
        --primary-light: #0ea5e9;
        --primary-bg: #e0f2fe;
        --success: #059669;
        --success-bg: #d1fae5;
        --danger: #dc2626;
        --danger-bg: #fee2e2;
        --warning: #d97706;
        --bg-main: #f8fafc;
        --bg-card: #ffffff;
        --bg-card-hover: #f1f5f9;
        --text-primary: #0f172a;
        --text-secondary: #64748b;
        --border: #e2e8f0;
        --border-hover: #cbd5e1;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
    }
    
    /* Global styles */
    .stApp {
        background: var(--bg-main);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 50%, #38bdf8 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 60%);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 0.5rem 0 0;
        position: relative;
        z-index: 1;
    }
    
    /* Cards */
    .custom-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
    }
    
    .custom-card:hover {
        border-color: var(--primary-light);
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    .card-title {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border);
    }
    
    .card-title span {
        font-size: 1.25rem;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px dashed var(--primary-light);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: var(--primary);
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
    }
    
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .upload-text {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    .upload-subtext {
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    /* Model selector */
    .model-selector {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
    }
    
    .model-selector-title {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    /* Result badges */
    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        width: 100%;
        justify-content: center;
    }
    
    .result-normal {
        background: var(--success-bg);
        border: 2px solid var(--success);
        color: var(--success);
    }
    
    .result-pneumonia {
        background: var(--danger-bg);
        border: 2px solid var(--danger);
        color: var(--danger);
    }
    
    /* Confidence meter */
    .confidence-container {
        margin: 1.25rem 0;
    }
    
    .confidence-label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .confidence-bar-bg {
        background: var(--border);
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
    }
    
    .confidence-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
    }
    
    .confidence-value {
        color: var(--text-primary);
        font-size: 1.75rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        margin-top: 0.5rem;
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        background: var(--bg-main);
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
        border: 1px solid var(--border);
    }
    
    .stat-value {
        font-size: 1.1rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.7rem;
        margin-top: 0.15rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Image container */
    .image-container {
        background: var(--bg-main);
        border-radius: 12px;
        padding: 0.75rem;
        border: 1px solid var(--border);
    }
    
    .image-container img {
        border-radius: 8px;
        width: 100%;
        height: auto;
    }
    
    /* Gradcam info */
    .gradcam-info {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.75rem;
        font-size: 0.8rem;
        color: #92400e;
    }
    
    /* Model info grid */
    .model-info-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .model-info-item {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: var(--shadow);
    }
    
    .model-info-value {
        color: var(--primary);
        font-size: 1rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
    }
    
    .model-info-label {
        color: var(--text-secondary);
        font-size: 0.75rem;
        margin-top: 0.25rem;
        text-transform: uppercase;
    }
    
    /* Footer */
    .custom-footer {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        margin-top: 2rem;
        box-shadow: var(--shadow);
    }
    
    .footer-text {
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    .footer-brand {
        color: var(--primary);
        font-weight: 600;
    }
    
    /* Model status badges */
    .model-status-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .model-badge.success {
        background: var(--success-bg);
        color: var(--success);
        border: 1px solid var(--success);
    }
    
    .model-badge.error {
        background: var(--danger-bg);
        color: var(--danger);
        border: 1px solid var(--danger);
    }
    
    .model-badge.warning {
        background: #fef3c7;
        color: #b45309;
        border: 1px solid #f59e0b;
    }
    
    /* Info box */
    .info-box {
        background: #f0f9ff;
        border: 1px solid #0ea5e9;
        border-radius: 10px;
        padding: 1rem;
        color: #0369a1;
        font-size: 0.875rem;
    }
    
    /* Performance section */
    .performance-section {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: var(--shadow);
    }
    
    .section-title {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Chemin vers les modeles
MODEL_DIR = "models"

@st.cache_resource
def load_models():
    """Charge tous les modeles sauvegardes"""
    models = {}
    model_files = {
        "Custom CNN": "best_model_custom_cnn.h5",
        "DenseNet121": "best_model_densenet121.h5",
        "DenseNet121 Fine-tuned": "best_model_densenet121_finetuned.h5",
        "VGG16": "best_model_vgg16.h5",
        "VGG16 Fine-tuned": "best_model_vgg16_finetuned.h5"
    }
    
    loaded_status = []
    
    for name, filename in model_files.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            try:
                models[name] = load_model(filepath)
                loaded_status.append((name, "success"))
            except Exception as e:
                loaded_status.append((name, "error"))
        else:
            loaded_status.append((name, "warning"))
    
    return models, loaded_status

@st.cache_data
def load_comparison_data():
    """Charge les donnees de comparaison"""
    comparison_file = os.path.join(MODEL_DIR, "models_comparison.csv")
    if os.path.exists(comparison_file):
        return pd.read_csv(comparison_file)
    return None

def preprocess_image(img, model_name, target_size):
    """Pretraite l'image pour la prediction"""
    img_array = np.array(img)
    img_resized = cv2.resize(img_array, target_size)
    
    if model_name == "Custom CNN":
        if len(img_resized.shape) == 3:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            img_resized = np.expand_dims(img_resized, axis=-1)
    else:
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 4:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
    
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_image(img, model, model_name):
    """Fait une prediction sur l'image"""
    target_size = (150, 150) if model_name == "Custom CNN" else (224, 224)
    img_processed = preprocess_image(img, model_name, target_size)
    
    prediction = model.predict(img_processed, verbose=0)
    pred_class = np.argmax(prediction[0])
    confidence = float(prediction[0][pred_class])
    
    label = "NORMAL" if pred_class == 0 else "PNEUMONIA"
    
    return label, confidence, prediction[0]

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Genere une heatmap Grad-CAM"""
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except:
        return None

def find_last_conv_layer(model):
    """Trouve la derniere couche convolutionnelle"""
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return f"{layer.name}/{sublayer.name}"
        elif isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def display_gradcam(img, model, model_name, pred_class):
    """Affiche le Grad-CAM"""
    target_size = (150, 150) if model_name == "Custom CNN" else (224, 224)
    img_processed = preprocess_image(img, model_name, target_size)
    last_conv = find_last_conv_layer(model)
    
    if last_conv is None:
        return None
    
    heatmap = make_gradcam_heatmap(img_processed, model, last_conv, pred_class)
    
    if heatmap is None:
        return None
    
    img_array = np.array(img)
    img_resized = cv2.resize(img_array, target_size)
    
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    heatmap_resized = cv2.resize(heatmap, target_size)
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
    
    return superimposed

# Chargement des modeles
models, model_status = load_models()

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🫁 PneumoScan AI</h1>
    <p>Système de Détection de Pneumonie par Intelligence Artificielle</p>
</div>
""", unsafe_allow_html=True)

# Afficher le statut des modèles
st.markdown('<div class="model-status-container">', unsafe_allow_html=True)
status_html = ""
for name, status in model_status:
    icon = "✓" if status == "success" else "✗" if status == "error" else "○"
    status_html += f'<span class="model-badge {status}">{icon} {name}</span>'
st.markdown(status_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if not models:
    st.error("⚠️ Aucun modèle chargé. Vérifiez le dossier 'models/'")
    st.stop()

# Zone unique de sélection du modèle et upload
col_upload, col_model = st.columns([2, 1])

with col_upload:
    st.markdown("""
    <div class="upload-section">
        <div class="upload-icon">📤</div>
        <div class="upload-text">Déposez votre radiographie thoracique ici</div>
        <div class="upload-subtext">Formats acceptés : JPG, JPEG, PNG</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choisir une image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col_model:
    st.markdown("""
    <div class="model-selector">
        <div class="model-selector-title">🎯 Sélection du Modèle</div>
    """, unsafe_allow_html=True)
    
    model_name = st.selectbox(
        "Modèle",
        list(models.keys()),
        label_visibility="collapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Info sur le modèle sélectionné
    st.markdown(f"""
    <div class="info-box">
        <strong>💡 Modèle actif :</strong> {model_name}<br>
        <small>Taille d'entrée : {"150×150" if model_name == "Custom CNN" else "224×224"}</small>
    </div>
    """, unsafe_allow_html=True)

# Contenu principal - si une image est uploadée
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title"><span>🔬</span> Image Originale</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction
    model = models[model_name]
    label, confidence, probs = predict_image(img, model, model_name)
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title"><span>🎯</span> Résultat de l'Analyse</div>
        </div>
        """, unsafe_allow_html=True)
        
        result_class = "result-normal" if label == "NORMAL" else "result-pneumonia"
        result_icon = "✓" if label == "NORMAL" else "⚠"
        
        st.markdown(f"""
        <div class="result-badge {result_class}">
            <span>{result_icon}</span>
            <span>{label}</span>
        </div>
        """, unsafe_allow_html=True)
        
        bar_color = "#059669" if label == "NORMAL" else "#dc2626"
        bar_width = confidence * 100
        
        st.markdown(f"""
        <div class="confidence-container">
            <div class="confidence-label">Niveau de Confiance</div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar" style="width: {bar_width}%; background: {bar_color};"></div>
            </div>
            <div class="confidence-value">{confidence:.1%}</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value" style="color: #059669;">{probs[0]:.1%}</div>
                <div class="stat-label">Normal</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color: #dc2626;">{probs[1]:.1%}</div>
                <div class="stat-label">Pneumonie</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Informations sur le modèle
    st.markdown(f"""
    <div class="model-info-grid">
        <div class="model-info-item">
            <div class="model-info-value">{model_name}</div>
            <div class="model-info-label">Architecture</div>
        </div>
        <div class="model-info-item">
            <div class="model-info-value">{model.count_params():,}</div>
            <div class="model-info-label">Paramètres</div>
        </div>
        <div class="model-info-item">
            <div class="model-info-value">{"150×150" if model_name == "Custom CNN" else "224×224"}</div>
            <div class="model-info-label">Taille d'entrée</div>
        </div>
        <div class="model-info-item">
            <div class="model-info-value">{"CNN" if model_name == "Custom CNN" else "Transfer"}</div>
            <div class="model-info-label">Type</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Section performances quand pas d'image
    st.markdown("""
    <div class="performance-section">
        <div class="section-title">📊 Performances des Modèles</div>
    </div>
    """, unsafe_allow_html=True)
    
    comparison_df = load_comparison_data()
    
    if comparison_df is not None:
        # Afficher le dataframe (compatible toutes versions Streamlit)
        st.dataframe(comparison_df)
        
        # Graphique comparatif avec thème clair
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#f8fafc')
        ax.set_facecolor('#ffffff')
        
        x = np.arange(len(comparison_df))
        width = 0.18
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#0284c7', '#7c3aed', '#059669', '#d97706']
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                offset = width * (i - 1.5)
                bars = ax.bar(x + offset, comparison_df[metric], width, 
                             label=metric, color=colors[i], alpha=0.85,
                             edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Modèles', fontweight='600', color='#0f172a', fontsize=11)
        ax.set_ylabel('Score', fontweight='600', color='#0f172a', fontsize=11)
        ax.set_title('Comparaison des Performances', fontsize=14, fontweight='700', 
                    color='#0f172a', pad=15)
        ax.set_xticks(x)
        
        if 'Modele' in comparison_df.columns:
            ax.set_xticklabels(comparison_df['Modele'], rotation=45, ha='right', color='#64748b')
        
        ax.tick_params(colors='#64748b')
        ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#0f172a')
        ax.grid(axis='y', alpha=0.3, color='#e2e8f0')
        ax.set_ylim([0.7, 1.0])
        
        for spine in ax.spines.values():
            spine.set_color('#e2e8f0')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("📊 Le fichier de comparaison des modèles n'a pas été trouvé.")

# Footer
st.markdown("""
<div class="custom-footer">
    <p class="footer-text">
        <span class="footer-brand">PneumoScan AI</span> — Système de Détection de Pneumonie par Deep Learning<br>
        <small style="color: #94a3b8;">Basé sur des radiographies thoraciques pédiatriques • À usage éducatif uniquement</small>
    </p>
</div>
""", unsafe_allow_html=True)