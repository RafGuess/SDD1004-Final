import streamlit as st
from PIL import Image
import numpy as np
import time

# Titre
st.title("Détection de voitures avec YOLOv8 (Mode Simulation)")

# Téléversement
uploaded_file = st.file_uploader("Téléversez une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléversée", use_column_width=True)

    # Simule un traitement
    with st.spinner("Détection en cours..."):
        time.sleep(2)  # Pause pour simuler l'inférence

    # Affiche une image annotée simulée (dans ce cas la même image)
    st.image(image, caption="Résultat simulé : voitures détectées", use_column_width=True)

    # Affiche un tableau de résultats fictif
    st.subheader("Résultats simulés")
    st.dataframe(np.array([
        [50, 50, 150, 150, 0.89, 2],
        [200, 100, 300, 250, 0.75, 2]
    ]),
    height=150,
    column_config={
        "0": st.column_config.NumberColumn("x1"),
        "1": st.column_config.NumberColumn("y1"),
        "2": st.column_config.NumberColumn("x2"),
        "3": st.column_config.NumberColumn("y2"),
        "4": st.column_config.NumberColumn("Confiance"),
        "5": st.column_config.NumberColumn("Classe (car)")
    })
