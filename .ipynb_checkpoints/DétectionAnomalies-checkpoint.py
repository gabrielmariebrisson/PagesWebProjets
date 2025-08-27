import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configuration de la page
st.set_page_config(
    page_title="Anomaly Detection Lab",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS
st.markdown("""
<style>
    .reportview-container {
        background-color: #F9F9F9;
        font-family: 'Arial', sans-serif;
    }

    h1, h2, h3 {
        color: #2C3E50;
        font-weight: 700;
    }

    .stButton>button {
        color: white;
        background-color: #e74c3c;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #c0392b;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Charger le modèle
@st.cache_resource
def load_anomaly_model():
    model = load_model("path_to_saved_model/anomaly_model.h5")  # Remplacez par le chemin de votre modèle
    return model

model = load_anomaly_model()

# Fonction pour prédire les anomalies
def predict_anomalies(data, model, threshold):
    """Prédit les anomalies en fonction du seuil défini."""
    predictions = model.predict(data)
    anomalies = (predictions > threshold).astype(int)
    return anomalies

# Interface utilisateur
st.title("🚨 Anomaly Detection Lab")
st.markdown("""
Cette application permet de détecter des anomalies à partir de données grâce à un modèle d'apprentissage automatique. Vous pouvez :
1. **Uploader vos propres données.**
2. **Configurer le seuil d'anomalie.**
3. **Visualiser les résultats.**
""")

# Uploader un fichier
uploaded_file = st.file_uploader("Uploader un fichier CSV contenant vos données :", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("**Aperçu des données :**")
    st.dataframe(data.head())

    # Sélection du seuil
    threshold = st.slider("Sélectionnez le seuil d'anomalie :", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.write(f"**Seuil sélectionné :** {threshold}")

    # Prédire les anomalies
    if st.button("🔍 Détecter les anomalies"):
        # Prétraitement des données
        features = data.values  # Adaptez en fonction de vos données
        anomalies = predict_anomalies(features, model, threshold)
        data['Anomaly'] = anomalies

        st.success("Détection terminée!")
        st.write("**Résultats avec anomalies :**")
        st.dataframe(data)

        # Graphiques
        st.subheader("📊 Visualisation des anomalies")
        st.bar_chart(data['Anomaly'].value_counts())

else:
    st.info("Veuillez uploader un fichier CSV pour commencer.")

# Footer
st.markdown("---")
st.markdown("© 2024 - Anomaly Detection Lab | Powered by Streamlit")
