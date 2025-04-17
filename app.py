from ultralytics import YOLO
from PIL import Image
import numpy as np
import streamlit as st
import time

# Configuration de page
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
        time.sleep(1)  # Simulation courte
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
        st.dataframe(boxes, use_container_width=True)
    else:
        st.warning("Aucune voiture détectée dans l'image.")
else:
    st.info("Veuillez téléverser une image pour commencer.")
