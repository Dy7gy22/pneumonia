import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Charger le modèle
model = load_model('my_model.h5')

# Fonction de prédiction
def predict(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 1)
    img = img / 255.0

    prediction = model.predict(img)
    predicted_class = (prediction >= 0.5).astype(int)
    return 'Normal' if predicted_class == 1 else 'Malade'

# Titre de l'application
st.title("Prédiction Pneumonie")

# Télécharger l'image
uploaded_file = st.file_uploader("Choisir une image de radiographie pulmonaire", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Sauvegarder l'image téléchargée
    with open("uploaded_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Afficher l'image téléchargée
    st.image(uploaded_file, caption='Image téléchargée', use_column_width=True)
    
    # Effectuer la prédiction
    prediction = predict("uploaded_image.png", model)
    
    # Afficher le résultat
    st.write(f"Prédiction: {prediction}")
