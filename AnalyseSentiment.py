import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Définir la configuration de la page en premier
st.set_page_config(
    page_title="Analyse de Sentiments",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Charger le modèle
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('./templates/assets/AnalyseSentiment.h5')

model = load_model()

@st.cache_resource
def load_tokenizer():
    with open('./templates/assets/tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)

tokenizer = load_tokenizer()

# Fonction pour traiter les séquences
def seq_pad_and_trunc(sequences, tokenizer, padding='post', truncating='post', maxlen=100):
    sequences = tokenizer.texts_to_sequences([sequences])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)
    return padded_sequences

# Bouton de redirection
st.markdown(
    """
    <a href="https://gabriel.mariebrisson.fr" target="_blank" style="text-decoration:none;">
    <div style="
    display:inline-block;
    background-color: grey;
    color:white;
    padding:10px 20px;
    border-radius:5px;
    text-align:center;
    font-size:16px;
    cursor:pointer;">
    Retour
    </div>
    </a>
    """,
    unsafe_allow_html=True
)

# Titre et introduction
st.title("Projet NLP : Surajustement et Classification des Sentiments")
st.markdown(
    """
    Ce projet utilise un modèle de réseau de neurones développé avec TensorFlow/Keras pour analyser 
    les sentiments dans des tweets, en exploitant une architecture basée sur les embeddings GloVe.
    """
)

# Section Présentation
st.header("Présentation")
st.markdown(
    """
    Ce projet vise à classifier les tweets en fonction des sentiments exprimés par les utilisateurs. Les réseaux sociaux jouent un rôle prépondérant dans la communication moderne, permettant à chacun d'exprimer librement son opinion. Les applications de cette technologie sont nombreuses :

    **Applications potentielles :**
    - **Marketing :** Analyser la perception d'une marque sur les réseaux sociaux.
    - **Service client :** Surveiller en temps réel les commentaires des clients.
    - **Prévisions de tendances :** Aider les entreprises à anticiper les besoins des consommateurs.
    - **Ressources humaines :** Évaluer le moral des employés.
    - **Finance :** Prédire l'évolution des cours boursiers.

    Pour cela, nous avons utilisé le jeu de données Sentiment140, qui contient 1,6 million de tweets étiquetés par sentiment (0 pour négatif, 4 pour positif). Contrairement à un déploiement industriel, cette approche ne nécessite pas une ingénierie des données complète, incluant l'extraction, le nettoyage et la gestion des données manquantes. Le code a été développé lors de la certification TensorFlow, Cours 3, semaine 3, et vous pouvez le retrouver ici : [GitHub](https://github.com).
    """
)

# Section Architecture du Modèle
st.header("Architecture du Modèle")
st.markdown(
    """
    Pour classifier un texte, il est essentiel de le transformer en un format compréhensible par la machine. Nous avons utilisé la méthode d'embedding préentraînée GloVe de Stanford. Cette technique tient compte de la fréquence à laquelle des paires de mots apparaissent ensemble dans les textes, permettant ainsi de capturer des relations sémantiques subtiles entre les mots. Par exemple, une opération vectorielle possible est : roi - homme + femme ≈ reine. Une approche plus moderne consisterait à utiliser une architecture de type transformateur.

    Le modèle de classification des sentiments se compose de plusieurs couches :
    - **Couche d'embedding :** Cette couche intègre la matrice d'embedding GloVe, permettant d'ajuster les poids pour gérer les mots non présents dans le vocabulaire de GloVe et de réduire la taille de la matrice.
    - **Couche Conv1D :** Elle capture les relations temporelles au sein des séquences textuelles. Bien que j'aurais pu utiliser des couches bidirectionnelles ou LSTM, qui sont plus adaptées, cela aurait prolongé le temps d'entraînement.
    - **Couche Dense :** Cette couche fournit la sortie finale en agrégeant les informations contextuelles extraites des couches précédentes.

    Les hyperparamètres, tels que le nombre de neurones et le taux d'apprentissage, jouent un rôle crucial dans les performances du modèle. Parmi ces hyperparamètres figurent : le nombre de neurones, le taux d'apprentissage, la longueur maximale des séquences et la dimension de l'embedding.
    """
)
st.image("./templates/assets/images/Architecture.png", caption="Structure du modèle de classification des sentiments", use_container_width=True)

# Section Résultats
st.header("Résultats")
st.markdown(
    """
    Les techniques les plus efficaces pour limiter le surajustement incluent la régularisation par dropout et la réduction de la complexité du modèle. Ces approches ont permis d'obtenir des résultats significatifs, comme en témoignent les courbes ci-dessous.

    En observant les courbes d'apprentissage, nous notons une convergence stable avec une précision atteignant 75 % sur les données de test (répartition 90/10) et 79 % sur les données d'apprentissage. Cela démontre une bonne capacité du modèle à généraliser sans trop s'adapter aux spécificités des données d'entraînement.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.image("./templates/assets/images/accurancy.png", caption="Courbes de précision", use_container_width=True)

with col2:
    st.image("./templates/assets/images/loss.png", caption="Courbes de perte", use_container_width=True)

st.markdown(
    """
    **Performances du modèle :**
    - Précision de 75 % sur les données de test.
    - Précision de 79 % sur les données d'apprentissage.
    - Bonne généralisation sans surajustement.

    En plus de la régularisation, des techniques telles que l'augmentation des données et l'ajustement des hyperparamètres ont été envisagées pour améliorer davantage les performances du modèle. Cela permettrait non seulement d'optimiser la précision, mais également d'accroître la robustesse face à des données variées.
    """
)

# Section Coût et Maintenance
st.header("Coût de Développement")
st.markdown(
    """
    Pour entraîner ce modèle, nous avons utilisé Google Colab, où l'entraînement à duré 13 minutes. Voici les spécifications matérielles utilisées :

    - **Processeur :** Intel Xeon (simple cœur) cadencé à 2,2 GHz.
    - **RAM :** 12,67 GiB.

    Les performances obtenues montrent une précision de 75 % sur les données de test et de 79 % sur les données d'apprentissage. Le poids du modèle chargé est de 52 Mo, et le temps d'exécution pour un test est de seulement 0,21 seconde.

    **Analyse des coûts :**
    - Le coût d'entraînement est raisonnable grâce à l'utilisation de Google Colab, qui offre des ressources GPU gratuites pour des projets de petite à moyenne envergure.
    - Les coûts de maintenance incluront principalement les mises à jour des données et l'optimisation des hyperparamètres.

    **Perspectives d'amélioration :**
    - Tester des architectures plus complexes comme les réseaux de neurones récurrents (RNN) ou les Transformers.
    - Utiliser des techniques d'augmentation de données pour enrichir l'ensemble d'apprentissage.
    - Implémenter une validation croisée pour mieux évaluer la robustesse du modèle.
    """
)

# Analyse de texte
st.header("Analyse de Sentiment en Temps Réel")
user_input = st.text_area("Entrez un texte en anglais pour tester le modèle :", 
                           placeholder="Tapez votre texte ici...")

if st.button("Analyser"):
    if user_input:
        # Prétraitement et prédiction
        processed_input = seq_pad_and_trunc(user_input, tokenizer)
        prediction = model.predict(processed_input)
        
        # Affichage du résultat
        sentiment = "positif" if prediction[0][0] > 0.5 else "négatif"
        st.success(f"La phrase est **{sentiment}** avec une probabilité de {prediction[0][0]:.2f}")
    else:
        st.error("Veuillez entrer un texte avant de lancer l'analyse.")

# Footer
st.markdown(
    """
    ---
    Développé par [Gabriel Marie-Brisson](https://gabriel.mariebrisson.fr)
    """
)
