from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import time

# Configuration de la page
st.set_page_config(page_title="Détection de voitures - YOLOv8n", layout="centered")

# Titre stylisé
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚗 Détection de voitures avec YOLOv8n</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Téléversez une image pour détecter les voitures présentes avec un modèle entraîné.</p>",
    unsafe_allow_html=True)

# Barre latérale
with st.sidebar:
    st.markdown("### Paramètres")
    st.info("Modèle utilisé : `yolov8n.pt`", icon="📦")
    st.write("💡 Essayez avec une photo de stationnement ou de rue.")

# Chargement du modèle
with st.spinner("Chargement du modèle YOLOv8n..."):
    model = YOLO("yolov8n.pt")

# Téléversement
uploaded_file = st.file_uploader("🔼 Téléversez une image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.markdown("---")
    st.subheader("🖼️ Image téléversée")
    st.image(image, use_container_width=True)

    # Traitement avec spinner
    with st.spinner("🔍 Détection en cours..."):
        time.sleep(1)
        image_array = np.array(image)
        results = model(image_array)

    st.success("Détection terminée ✅")

    # Image annotée
    st.markdown("---")
    st.subheader("📍 Résultat avec détection")
    annotated_image = results[0].plot()
    st.image(annotated_image, use_container_width=True)

    # Résultats bruts
    st.markdown("---")
    st.subheader("📊 Résultats bruts")
    boxes = results[0].boxes.data.cpu().numpy()

    if boxes.size > 0:
        df = pd.DataFrame(
            boxes,
            columns=["x1", "y1", "x2", "y2", "Confiance", "Classe"]
        )
        st.dataframe(df, use_container_width=True)

        # Phrase explicative
        st.markdown(
            "<p style='margin-top:10px;'>Chaque ligne représente une voiture détectée dans l’image. "
            "Le score de <strong>confiance</strong> de la détection, ainsi que la classe prédite "
            "(Classe = 2 correspond généralement à une voiture).</p>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Aucune voiture détectée dans l'image.")
else:
    st.info("Veuillez téléverser une image pour commencer.")
