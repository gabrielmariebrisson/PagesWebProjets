# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import requests
from collections import defaultdict

# --- Fonctions de l'API TMDb ---
@st.cache_data(ttl=3600)
def get_movie_data(title):
    """Récupère les infos du film depuis OMDb"""
    year = None
    clean_title = title
    if title[-1] == ")" and "(" in title:
        try:
            year = title.split("(")[-1][:-1]
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

# --- Fonctions de chargement ---
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

# --- Fonction de recommandation ---
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

# Bouton de redirection
st.markdown(
    """
    <a href="https://gabriel.mariebrisson.fr" target="_blank" style="text-decoration:none;">
    <div style="
    display: inline-block;
    background: linear-gradient(135deg, #6A11CB 0%, #2575FC 100%);
    color: white;
    padding: 12px 25px;
    border-radius: 30px;
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(37, 117, 252, 0.3);
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 2px solid transparent;
    position: relative;
    overflow: hidden;
    ">
    Retour
    <span style="
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255,255,255,0.2);
    transform: scaleX(0);
    transform-origin: right;
    transition: transform 0.3s ease;
    z-index: 1;
    "></span>
    </div>
    </a>
    """,
    unsafe_allow_html=True
)

# --- Interface Streamlit ---
st.set_page_config(layout="wide", page_title="Ciné-Reco")
st.title("🎬 Ciné-Reco : Votre Guide Cinéma Personnalisé")


model = load_model()
scalerUser, scalerItem, scalerTarget, movie_dict, item_vecs_finder, unique_genres = load_objects()


if model and movie_dict:
    # --- Barre latérale pour la notation ---
    st.sidebar.header("🔍 Notez des films")
    
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}

    # CORRECTION 1: Recherche HORS du formulaire
    movie_list = sorted([info['title'] for info in movie_dict.values()])
    search_term = st.sidebar.text_input("Rechercher un film à noter :")
    
    if search_term:
        filtered_movie_list = [m for m in movie_list if search_term.lower() in m.lower()]
    else:
        filtered_movie_list = movie_list[:1000]  # Limiter par défaut

    # Le formulaire contient uniquement la sélection et la note
    with st.sidebar.form("rating_form"):
        if filtered_movie_list:
            selected_movie_title = st.selectbox("Choisissez un film", filtered_movie_list)
        else:
            st.warning("Aucun film trouvé avec cette recherche")
            selected_movie_title = None
            
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
            st.rerun()

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
        
        # CORRECTION 2: Gestion des NaN dans les genres
        all_genres = set()
        for genres_str in recommendations_df['Genres'].dropna():
            if genres_str and genres_str != "(no genres listed)":
                for genre in str(genres_str).split('|'):
                    if genre.strip():
                        all_genres.add(genre.strip())
        
        all_genres = sorted(list(all_genres))
        selected_genres = st.multiselect("Filtrer par genre :", all_genres)
        
        if selected_genres:
            def has_selected_genre(genres_str):
                if pd.isna(genres_str) or not genres_str:
                    return False
                return any(g in str(genres_str) for g in selected_genres)
            
            filtered_df = recommendations_df[recommendations_df['Genres'].apply(has_selected_genre)]
        else:
            filtered_df = recommendations_df

        # Affichage avec affiches
        st.subheader(f"Top {min(20, len(filtered_df))} des films pour vous :")
        
        cols = st.columns(5)
        for i, (idx, row) in enumerate(filtered_df.head(20).iterrows()):
            col = cols[i % 5]
            with col:
                movie_data = get_movie_data(row['Titre'])
                
                if movie_data and movie_data["poster"] and movie_data["poster"] != "N/A":
                    st.image(movie_data["poster"], caption=f"{row['Note Prédite']:.1f} ⭐")
                else:
                    st.image("./templates/assets/images/no-poster.jpg", caption=f"{row['Note Prédite']:.1f} ⭐")
                
                with st.expander(f"_{row['Titre']}_"):
                    st.write(f"**Genres :** {movie_data['genre'] if movie_data else row['Genres']}")
                    st.write(f"**Note prédite :** {row['Note Prédite']:.2f}")
                    if movie_data:
                        st.write(f"**Acteurs :** {movie_data['actors']}")
                        st.write(f"**Résumé :** {movie_data['plot']}")
                        st.write(f"**Note IMDb :** {movie_data['rating']} ⭐")
                        st.write(f"**Année :** {movie_data['year']}")

    else:
        st.info("👋 Bienvenue ! Veuillez noter au moins 3 films dans la barre latérale pour débloquer vos recommandations.")


    # Section Présentation
    st.header("Présentation")
    st.markdown(
        """
        Ce projet vise à recommander des films en fonction des notes attribuées par les utilisateurs. À l'ère du numérique, 
        les algorithmes de recommandation sont omniprésents et jouent un rôle crucial dans nos choix quotidiens, en suggérant 
        du contenu aligné avec nos préférences.

        **Domaines d'application :**
        - **E-commerce & Marketing :** Suggestion de produits similaires aux achats précédents pour augmenter les conversions
        - **Services client :** Recommandation de services adaptés aux besoins identifiés de l'utilisateur
        - **Divertissement :** Proposition de films, séries ou musiques correspondant aux goûts de chacun
        - **Analyse de tendances :** Identification de tendances émergentes basées sur les comportements collectifs
        - **Ressources humaines :** Mise en relation de profils compatibles (recrutement, networking)
        - **Éducation :** Parcours d'apprentissage personnalisés selon le niveau et les centres d'intérêt

        **Données utilisées :**
        
        Ce système s'appuie sur le jeu de données MovieLens, qui contient :
        - **20 763 films** couvrant la période de 1874 à 2024
        - **8 493 utilisateurs** actifs
        - **2 864 752 notes** au total (échelle de 0.5 à 5 étoiles)
        - Données à jour jusqu'au 1er mai 2024
        - Multiples genres cinématographiques pour affiner les recommandations
        """
    )

    # Section Architecture du Modèle
    st.header("Architecture du Modèle")
    st.markdown(
        """
        Notre système repose sur un **réseau de neurones siamois à deux branches**, une architecture particulièrement 
        adaptée à l'apprentissage de similarités entre entités hétérogènes.

        **Structure du modèle :**
        
        Le modèle est composé de deux sous-réseaux parallèles :
        
        1. **Branche utilisateur :** Transforme le profil utilisateur (historique de notes, préférences de genres) 
        en une représentation vectorielle dense
        
        2. **Branche film :** Encode les caractéristiques des films (genres, popularité, patterns de notation) 
        dans un espace latent commun
        
        **Composants techniques :**
        - **Couches denses successives** (256 → 128 → 64 neurones) pour l'extraction de features hiérarchiques
        - **Activation GELU** : Fonction d'activation continue favorisant une meilleure propagation du gradient
        - **Normalisation par batch** : Stabilise l'apprentissage et accélère la convergence
        - **Dropout (30%)** : Prévient le surapprentissage en désactivant aléatoirement certains neurones
        - **Régularisation L2 (1e-6)** : Pénalise les poids élevés pour favoriser la généralisation
        - **Normalisation L2 finale** : Projette les embeddings sur une hypersphère unitaire pour des comparaisons stables
        
        **Couche de fusion :**
        
        Les vecteurs normalisés sont combinés via deux opérations complémentaires :
        - **Différence absolue** : Capture la dissimilarité entre utilisateur et film
        - **Produit élément par élément** : Modélise la similarité bilinéaire et les interactions fines
        
        **Prédiction finale :**
        
        Une couche Dense avec activation sigmoïde produit un score de compatibilité entre 0 et 1, 
        facilement convertible en note prédite sur l'échelle 0.5-5 étoiles.

        **Limitations connues :**
        - Pas de prise en compte des données textuelles (synopsis, critiques)
        - L'ordre chronologique des notes n'est pas exploité
        - Les évolutions temporelles des préférences ne sont pas modélisées
        
        Malgré ces limitations, le modèle offre des recommandations fiables et pertinentes.
        """
    )
    st.image("./templates/assets/film/architecture_model.png", caption="Architecture du modèle neuronal", use_container_width=True)

    # Section Résultats
    st.header("Performances du Modèle")
    st.markdown(
        """
        **Capacités du système :**
        
        Le modèle entraîné permet deux types d'utilisation :
        1. **Prédiction de note** : Estimer la note qu'un utilisateur attribuerait à un film non vu
        2. **Recommandation personnalisée** : Suggérer les films avec les meilleures notes prédites pour un utilisateur donné
        
        **Métriques de performance :**
        - **RMSE (Root Mean Square Error) : 0.35** - Erreur moyenne de prédiction
        - **MSE (Mean Square Error) : 0.12** - Métrique d'optimisation du modèle
        
        Ces résultats sont satisfaisants pour un système de recommandation : une erreur de ~0.35 étoile 
        représente une précision acceptable dans la prédiction des préférences cinématographiques.
        
        À titre de comparaison, les systèmes de recommandation professionnels atteignent généralement des RMSE 
        entre 0.25 et 0.40 sur MovieLens, positionnant notre modèle dans une fourchette compétitive.
        """
    )

    # Section Coût et Maintenance
    st.header("Développement et Déploiement")
    st.markdown(
        """
        **Infrastructure d'entraînement :**
        - Matériel utilisé : MacBook M1 (sans GPU dédié)
        - Temps de préparation des données : ~30 minutes
        - Durée d'entraînement : 35 minutes
        - Coût total : 0€ (aucune ressource cloud nécessaire)
        
        **Caractéristiques du modèle en production :**
        - Taille du modèle : 1.8 Mo (déploiement léger)
        - Temps d'inférence : < 1 seconde pour générer des recommandations
        - Scalabilité : Compatible avec des environnements à ressources limitées
        
        **Coûts opérationnels :**
        - **Entraînement** : Gratuit (CPU standard suffisant)
        - **Hébergement** : Minimal (faible empreinte mémoire)
        - **Maintenance** : Mise à jour périodique du dataset et réentraînement occasionnel
        
        **Axes d'amélioration futurs :**
        - Intégration de données textuelles (NLP sur synopsis et critiques)
        - Prise en compte de la dimension temporelle (évolution des goûts)
        - Ajout de features contextuelles (heure, dispositif, météo)
        - Modèle hybride combinant filtrage collaboratif et approche content-based
        - A/B testing pour optimiser les hyperparamètres en production
        - Explainability : visualisation des facteurs influençant chaque recommandation
        """
    )

else:
    st.error("L'application n'a pas pu démarrer. Vérifiez les fichiers du modèle et des données.")