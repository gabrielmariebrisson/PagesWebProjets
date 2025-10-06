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
        Ce projet vise à recommender des films enfonction des notes par utilisateur par les utilisateurs. Les informations sont partout aujourd'hui, et des algorithme de recommendation choissent pour nous ce que nous avons le plus de chance d'aimer. Nous retrouvons ses algorithmes partout :

        **Applications potentielles :**
        - **Marketing :** Dans la recommendation de produits, les utilisateurs sont plus susceptibles de trouver des produits similaires à ceux qu'ils ont déjà achetés.
        - **Service client :** Dans la recommandation de services, les utilisateurs sont plus susceptibles de trouver des services similaires à ceux qu'ils ont déjà utilisés
        - **Divertissement :** Dans la recommendation de films, les utilisateurs sont plus susceptibles de trouver des films similaires à ceux qu'ils ont déjà regardés.
        - **Prévisions de tendances :** Dans la recommandation de tendances, les utilisateurs sont plus susceptibles de trouver des tendances similaires à ceux qu'ils ont déjà suivis.
        - **Ressources humaines :** Dans la recommandation de personnes, les utilisateurs sont plus susceptibles de trouver des personnes similaires à ceux qu'ils ont déjà rencontrés.
        - **Éducation :** Dans la recommandation de cours, les utilisateurs sont plus susceptibles de trouver des cours similaires à ceux qu'ils ont déjà suivis.


        Pour cela, nous avons utilisé le jeu de données movielensBeliefs afin d'avoir les films notes des films jusqu'au 1er mai 2024, la liste des genres que nous avons en filtre, avec des notes allant de 0.5 a 5 étoiles, les films vont de 1874 a 2024 pour 20 763 films et 8 493 utilisateurs donc soit un totale de 2 864 752 notes.
        """
    )

    # Section Architecture du Modèle
    st.header("Architecture du Modèle")
    st.markdown(
        """
        Pour cette algorithme de recommandation, nous avons choisit une architecture de réseau de neurones à deux branches. Une branche traite les données utilisateur, tandis que l'autre gère les données des films. Chaque branche comprend plusieurs couches denses, avec des fonctions d'activation ReLU et des techniques de régularisation telles que le dropout pour prévenir le surajustement. Les sorties des deux branches sont ensuite combinées à l'aide d'une couche Lambda, qui calcule le score de recommandation en fonction des notes des utilisateurs et des notes des films.
        Cette approche permet de capturer les interactions complexes entre les préférences des utilisateurs et les caractéristiques des films, améliorant ainsi la précision des recommandations.
        Attention, ce modèle ne prend pas en compte les données textuelles, l'ordre des notes, ni les données temporelles. Ainsi les performances sont limitées mais restent fiables.

        Le modèle de prédiction des préférences utilisateur–film est composé de plusieurs blocs :
        - **Sous-réseau utilisateur et sous-réseau item :** chacun est constitué de couches entièrement connectées (Dense) avec normalisation (BatchNormalization) et activation GELU, permettant d’extraire des représentations denses et non linéaires des utilisateurs et des films.
        - **Couches de régularisation :** des couches Dropout (0.3) et une régularisation L2 légère sur les poids limitent le surapprentissage tout en préservant la richesse de l’espace latent.
        - **Normalisation L2 :** les vecteurs issus des deux réseaux sont normalisés pour contraindre les embeddings sur une hypersphère unitaire, ce qui stabilise la comparaison entre utilisateurs et items.
        - **Couche de fusion :** les représentations sont combinées via la différence absolue et le produit élément par élément, permettant au modèle de capturer à la fois la dissimilarité et la similarité bilinéaire.
        - **Couche de sortie Dense :** une couche sigmoïde fournit la probabilité de correspondance entre un utilisateur et un film.

        Les principaux hyperparamètres incluent la taille des couches (256, 128, 64), le taux de dropout (0.3), la régularisation L2 (1e-6) et l’activation GELU, choisie pour sa continuité et sa meilleure propagation du gradient par rapport à ReLU.
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

else:
    st.error("L'application n'a pas pu démarrer. Vérifiez les fichiers du modèle et des données.")