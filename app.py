import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.applications.resnet50 import preprocess_input

# Cache para no cargar el modelo en cada interaccion
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('transport_classifier.keras')
    with open('clases.json') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

modelo, idx_to_class = load_model()

st.title('Clasificador Medio de Transporte')
st.write('Sube una imagen para clasificar')

uploaded_file = st.file_uploader('Imagen', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Imagen subida', use_column_width=True)
    
    # Preprocesar
    IMG_SIZE = 224  # el mismo que usaste en entrenamiento
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predecir
    pred = modelo.predict(img_array, verbose=0)[0]
    
    # Mostrar resultados
    st.subheader('Predicciones:')
    for idx in pred.argsort()[::-1]:
        nombre = idx_to_class[idx].replace('_', ' ').title()
        prob = pred[idx]
        st.write(f'**{nombre}**: {prob:.1%}')
        st.progress(float(prob))
