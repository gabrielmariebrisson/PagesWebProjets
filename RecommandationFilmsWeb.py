# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import requests # Pour les requêtes API
from collections import defaultdict
import logging


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger('CineReco')

# Vérification des fichiers
logger.info("=" * 60)
logger.info("🎬 DÉMARRAGE DE CINÉ-RECO")
logger.info("=" * 60)




# --- Fonctions de l'API TMDb ---
@st.cache_data(ttl=3600 ) # Mettre en cache les résultats de l'API pendant 1h
def get_movie_data(title):
    """Récupère les infos du film depuis OMDb"""
    # Essaye d’extraire une année s’il y en a à la fin du titre
    year = None
    clean_title = title
    if title[-1] == ")" and "(" in title:
        try:
            year = title.split("(")[-1][:-1]  # extrait l’année entre parenthèses
            clean_title = title.rsplit("(", 1)[0].strip()
        except:
            pass

    url = f"http://www.omdbapi.com/?t={clean_title}&apikey=667fd579"
    if year:
        url += f"&y={year}"

    try:
        response = requests.get(url).json()
        if response.get("Response") == "True":
            return {
                "title": response["Title"],
                "year": response["Year"],
                "genre": response.get("Genre", "N/A"),
                "director": response.get("Director", "N/A"),
                "actors": response.get("Actors", "N/A"),
                "plot": response.get("Plot", "N/A"),
                "rating": response.get("imdbRating", "N/A"),
                "votes": response.get("imdbVotes", "0"),
                "poster": response.get("Poster", None),
            }
    except requests.RequestException:
        return None

    return None


# --- Fonctions de chargement (inchangées) ---
@st.cache_resource
def load_model():
    try:
        def l2_norm(x):
            return tf.linalg.l2_normalize(x, axis=1)

        def diff_abs(x):
            return tf.abs(x[0] - x[1])

        def prod_mul(x):
            return x[0] * x[1]

        return tf.keras.models.load_model(
            "./templates/assets/film/best_model.keras",
            custom_objects={
                'l2_norm': l2_norm,
                'diff_abs': diff_abs,
                'prod_mul': prod_mul
            },
            safe_mode=False
        )
    
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

@st.cache_data
def load_objects():
    try:
        with open('./templates/assets/film/scalerUser.pkl', 'rb') as f: scalerUser = pickle.load(f)
        with open('./templates/assets/film/scalerItem.pkl', 'rb') as f: scalerItem = pickle.load(f)
        with open('./templates/assets/film/scalerTarget.pkl', 'rb') as f: scalerTarget = pickle.load(f)
        with open('./templates/assets/film/movie_dict.pkl', 'rb') as f: movie_dict = pickle.load(f)
        with open('./templates/assets/film/item_vecs_finder.pkl', 'rb') as f: item_vecs_finder = pickle.load(f)
        with open('./templates/assets/film/unique_genres.pkl', 'rb') as f: unique_genres = pickle.load(f)
        return scalerUser, scalerItem, scalerTarget, movie_dict, item_vecs_finder, unique_genres
    except FileNotFoundError:
        st.error("Fichiers .pkl manquants. Assurez-vous d'avoir exécuté le script de sauvegarde.")
        return (None,) * 6

# --- Fonction de recommandation (inchangée) ---
def generate_recommendations(model, user_ratings, scalers, data):
    scalerUser, scalerItem, scalerTarget = scalers
    movie_dict, item_vecs, unique_genres = data

    if not all([model, scalerUser, scalerItem, scalerTarget, movie_dict, item_vecs is not None, unique_genres]):
        return pd.DataFrame()

    num_ratings = len(user_ratings)
    avg_rating = np.mean(list(user_ratings.values()))
    genre_ratings = defaultdict(list)
    for movie_id, rating in user_ratings.items():
        genres_str = movie_dict[movie_id]['genres']
        if pd.notna(genres_str) and genres_str != "(no genres listed)":
            for genre in genres_str.split('|'):
                genre_ratings[genre].append(rating)

    user_prefs = {f'pref_{g}': np.mean(genre_ratings.get(g, [avg_rating])) for g in unique_genres}
    user_vec = np.array([[num_ratings, avg_rating, 0] + list(user_prefs.values())])
    
    num_items = len(item_vecs)
    user_vecs_repeated = np.tile(user_vec, (num_items, 1))

    suser_vecs = scalerUser.transform(user_vecs_repeated)
    sitem_vecs = scalerItem.transform(item_vecs[:, 1:])

    predictions = model.predict([suser_vecs, sitem_vecs])
    predictions_rescaled = scalerTarget.inverse_transform(predictions)

    recommendations = []
    for i, item_id in enumerate(item_vecs[:, 0]):
        if int(item_id) not in user_ratings:
            recommendations.append({
                'Movie ID': int(item_id),
                'Titre': movie_dict[int(item_id)]['title'],
                'Genres': movie_dict[int(item_id)]['genres'],
                'Note Prédite': predictions_rescaled[i][0]
            })

    reco_df = pd.DataFrame(recommendations)
    return reco_df.sort_values(by='Note Prédite', ascending=False)

# --- Interface Streamlit ---
logger.info(" --- Interface Streamlit ---")
st.set_page_config(layout="wide", page_title="Ciné-Reco")
st.title("🎬 Ciné-Reco : Votre Guide Cinéma Personnalisé")

logger.info("--- Chargement des données ---")
logger.info("- Modèle")
# Chargement
model = load_model()
logger.info("- Scalers")
scalerUser, scalerItem, scalerTarget, movie_dict, item_vecs_finder, unique_genres = load_objects()
logger.info("--- Données chargées avec succès ---")


if model and movie_dict:
    # --- Barre latérale pour la notation ---
    st.sidebar.header("📝 Notez des films")
    
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}

    # AMÉLIORATION: Recherche de film
    movie_list = sorted([info['title'] for info in movie_dict.values()])
    search_term = st.sidebar.text_input("Rechercher un film à noter :")
    if search_term:
        # Filtrer la liste des films en fonction de la recherche
        filtered_movie_list = [m for m in movie_list if search_term.lower() in m.lower()]
    else:
        filtered_movie_list = movie_list

    with st.sidebar.form("rating_form"):
        selected_movie_title = st.selectbox("Choisissez un film", filtered_movie_list[:1000]) # Limiter pour la performance
        rating = st.slider("Votre note", 1.0, 5.0, 3.0, 0.5)
        submitted = st.form_submit_button("Ajouter la note")

        if submitted and selected_movie_title:
            movie_id = next((mid for mid, info in movie_dict.items() if info['title'] == selected_movie_title), None)
            if movie_id:
                st.session_state.user_ratings[movie_id] = rating
                st.success(f"Note ajoutée pour : {selected_movie_title}")

    if st.session_state.user_ratings:
        st.sidebar.subheader("Vos notes :")
        for movie_id, rating in st.session_state.user_ratings.items():
            st.sidebar.write(f"- {movie_dict[movie_id]['title']}: **{rating} / 5.0**")
        if st.sidebar.button("🗑️ Vider les notes"):
            st.session_state.user_ratings = {}
            st.experimental_rerun()

    # --- Affichage principal des recommandations ---
    st.header("🌟 Vos Recommandations Personnalisées")
    if len(st.session_state.user_ratings) >= 3:
        with st.spinner("Nous préparons votre sélection personnalisée..."):
            recommendations_df = generate_recommendations(
                model, 
                st.session_state.user_ratings,
                (scalerUser, scalerItem, scalerTarget),
                (movie_dict, item_vecs_finder, unique_genres)
            )
        
        # AMÉLIORATION: Filtre par genre
        all_genres = sorted(list(set(g for genres in recommendations_df['Genres'].str.split('|') for g in genres if g != "(no genres listed)")))
        selected_genres = st.multiselect("Filtrer par genre :", all_genres)
        
        if selected_genres:
            # Filtrer le dataframe pour ne garder que les films qui ont AU MOINS UN des genres sélectionnés
            filtered_df = recommendations_df[recommendations_df['Genres'].apply(lambda x: any(g in x for g in selected_genres))]
        else:
            filtered_df = recommendations_df

        # AMÉLIORATION: Affichage avec affiches
        st.subheader(f"Top {min(20, len(filtered_df))} des films pour vous :")
        
        cols = st.columns(5) # Créer 5 colonnes pour les affiches
        for i, row in enumerate(filtered_df.head(20).itertuples()):
            col = cols[i % 5]
            with col:
                movie_data = get_movie_data(row.Titre)
                
                if movie_data and movie_data["poster"] and movie_data["poster"] != "N/A":
                    st.image(movie_data["poster"], caption=f"{row.Note_Prédite:.1f} ★")
                else:
                    st.image("https://via.placeholder.com/200x300.png?text=Pas+d'affiche", caption=f"{row.Note_Prédite:.1f} ★")
                
                with st.expander(f"_{row.Titre}_"):
                    st.write(f"**Genres :** {movie_data['genre'] if movie_data else row.Genres}")
                    st.write(f"**Note prédite :** {row.Note_Prédite:.2f}")
                    if movie_data:
                        st.write(f"**Acteurs :** {movie_data['actors']}")
                        st.write(f"**Résumé :** {movie_data['plot']}")
                        st.write(f"**Note IMDb :** {movie_data['rating']} ⭐")
                        st.write(f"**Année :** {movie_data['year']}")

    else:
        st.info("👋 Bienvenue ! Veuillez noter au moins 3 films dans la barre latérale pour débloquer vos recommandations.")

else:
    st.error("L'application n'a pas pu démarrer. Vérifiez les fichiers du modèle et des données.")

