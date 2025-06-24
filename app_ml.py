import streamlit as st
import tensorflow as tf
import requests
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import gdown
import base64

# Path a tu imagen de fondo
background_image_path = "medical_laboratory.jpg"

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def set_background(image_path):
    b64_image = get_base64_image(image_path)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{b64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main-title {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .prediction-box {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }}
    .stFileUploader > div {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 8px;
    }}
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.95) !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Establece el fondo
set_background(background_image_path)

# Función para cargar el modelo con cache (más eficiente)
@st.cache_resource
def load_model_from_url_cached(file_id, model_filename="downloaded_model.h5"):
    if not os.path.exists(model_filename):
        # Puedes mantener la descarga sin mensajes adicionales si quieres
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_filename, quiet=False)
    model = load_model(model_filename)
    return model

# ID del modelo en Google Drive
file_id = '1qLS6t1a5R3hk5PtpvUxuQQjAHnDcUFeU'

# Carga del modelo (sin mensajes)
model = load_model_from_url_cached(file_id)

# Clases posibles
class_labels = ["Healthy", "Tumor"]

# Estado de sesión
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

# Función para preprocesar la imagen
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    from tensorflow.keras.applications.resnet import preprocess_input
    return preprocess_input(image_array)

# Título con estilos
st.markdown("""
<div class="main-title">
    <h1>🧠 Deep Learning para Detectar Tumor Cerebral 🔎</h1>
</div>
""", unsafe_allow_html=True)

# Diseño en columnas para subir y mostrar imagen
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

    predict_button = st.button("Predecir")

    # Si se sube una nueva imagen, reiniciar predicción
    if uploaded_file:
        if st.session_state.get('uploaded_image') != uploaded_file:
            st.session_state['uploaded_image'] = uploaded_file
            st.session_state['prediction'] = None
    elif st.session_state.get('uploaded_image'):
        uploaded_file = st.session_state['uploaded_image']
    else:
        uploaded_file = None

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Imagen subida.', width=240)

# Lógica para predecir
if predict_button and uploaded_file:
    # Preprocesar y predecir
    image = Image.open(uploaded_file).convert('RGB')
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    # Interpretar predicciones
    if predictions.shape[1] == 1:
        pred_value = predictions[0][0]
        if pred_value >= 0.5:
            predicted_class = "Tumor"
            confidence = pred_value
        else:
            predicted_class = "Healthy"
            confidence = 1 - pred_value
    elif predictions.shape[1] == 2:
        probs = predictions[0]
        predicted_index = np.argmax(probs)
        predicted_class = class_labels[predicted_index]
        confidence = probs[predicted_index]
    else:
        st.write("Forma inesperada de salida del modelo:", predictions.shape)
        predicted_class = "Desconocido"
        confidence = 0.0

    # Guardar predicción en el estado
    st.session_state['prediction'] = {
        'class': predicted_class,
        'confidence': confidence
    }

# Mostrar resultados
if st.session_state['prediction']:
    pred = st.session_state['prediction']
    st.markdown(f"""
    <div class="prediction-box">
        <h3>Resultado de la Predicción:</h3>
        <p><strong>{pred['class']}</strong> con confianza <strong>{pred['confidence']*100:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)
