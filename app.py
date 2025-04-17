from ultralytics import YOLO
from PIL import Image
import numpy as np
import streamlit as st

# Titre
st.title("Détection de voitures avec YOLOv8n")

# Chargement du modèle
model = YOLO("yolov8n.pt")

# Téléversement de l’image
uploaded_file = st.file_uploader("Téléversez une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléversée", use_column_width=True)

    # Conversion
    image_array = np.array(image)

    # Inférence
    results = model(image_array)

    # Affichage image annotée
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Résultat : voitures détectées", use_column_width=True)

    # Résultats bruts
    st.subheader("Résultats bruts")
    st.dataframe(results[0].boxes.data.cpu().numpy())
