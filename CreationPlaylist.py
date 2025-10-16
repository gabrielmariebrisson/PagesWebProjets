import streamlit as st
from spotipy.oauth2 import SpotifyOAuth
import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
import tempfile
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import subprocess
import warnings
from templates.assets.music.architecture import SimpleCNN
warnings.filterwarnings('ignore')

load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Music Playlist Generator",
    page_icon="🎵",
    layout="wide"
)


# Mapping des genres
label_mapping = {
    0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
    5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
}

genre_colors = {
    'blues': '#1f77b4', 'classical': '#ff7f0e', 'country': '#2ca02c',
    'disco': '#d62728', 'hiphop': '#9467bd', 'jazz': '#8c564b',
    'metal': '#e377c2', 'pop': '#7f7f7f', 'reggae': '#bcbd22', 'rock': '#17becf'
}


if 'model' not in st.session_state:
    st.session_state.model = None
if 'analyzed_tracks' not in st.session_state:
    st.session_state.analyzed_tracks = []
if 'spotify_top_tracks' not in st.session_state:
    st.session_state.spotify_top_tracks = []
if 'spotify_recent_tracks' not in st.session_state:
    st.session_state.spotify_recent_tracks = []
if 'spotify_saved_tracks' not in st.session_state:
    st.session_state.spotify_saved_tracks = []
if 'spotify_client' not in st.session_state:
    st.session_state.spotify_client = None

# --- INTERFACE STREAMLIT PRINCIPALE ---
st.title("🎵 Music Playlist Generator")
st.markdown("Créez des playlists personnalisées avec l'IA - Analyse de genres musicaux par CNN")

# Configuration des variables d'environnement (à définir dans vos secrets Streamlit)
CLIENT_ID =  os.getenv('CLIENT_ID_SPOTIFY')
CLIENT_SECRET = os.getenv('CLIENT_SECRET_SPOTIFY')
REDIRECT_URI = os.getenv('REDIRECT_URI_SPOTIFY')

# Initialisation de la session state
if 'spotify_client' not in st.session_state:
    st.session_state.spotify_client = None
if 'analyzed_tracks' not in st.session_state:
    st.session_state.analyzed_tracks = []
if 'user_top_tracks' not in st.session_state:
    st.session_state.user_top_tracks = []

def get_spotify_client():
    """Obtient le client Spotify authentifié"""
    
    # Si déjà connecté
    if st.session_state.spotify_client:
        return st.session_state.spotify_client
    
    # Configuration OAuth
    sp_oauth = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="user-library-read user-top-read playlist-modify-public playlist-modify-private user-read-recently-played user-read-email",
        cache_path=".spotify_cache"
    )
    
    # Vérifier le token dans les paramètres d'URL (après redirection)
    if st.query_params.get("code"):
        try:
            token_info = sp_oauth.get_access_token(st.query_params["code"])
            if token_info:
                spotify_client = spotipy.Spotify(auth=token_info['access_token'])
                st.session_state.spotify_client = spotify_client
                # Nettoyer l'URL
                st.query_params.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Erreur d'authentification: {e}")
    
    # Vérifier le cache
    try:
        token_info = sp_oauth.get_cached_token()
        if token_info and not sp_oauth.is_token_expired(token_info):
            spotify_client = spotipy.Spotify(auth=token_info['access_token'])
            st.session_state.spotify_client = spotify_client
    except Exception:
        pass
    
    return st.session_state.spotify_client

def get_user_top_tracks(spotify_client, limit=20, time_range='medium_term'):
    """Récupère les musiques les plus écoutées de l'utilisateur"""
    try:
        results = spotify_client.current_user_top_tracks(
            limit=limit, 
            time_range=time_range  # short_term, medium_term, long_term
        )
        
        tracks = []
        for track in results['items']:
            track_data = {
                'id': track['id'],
                'name': track['name'],
                'artists': [artist['name'] for artist in track['artists']],
                'preview_url': track.get('preview_url'),
                'uri': track['uri'],
                'external_urls': track['external_urls'],
                'album': track['album']['name'],
                'popularity': track['popularity']
            }
            tracks.append(track_data)
        
        return tracks
    except Exception as e:
        st.error(f"Erreur lors de la récupération des musiques: {e}")
        return []

def get_user_recent_tracks(spotify_client, limit=20):
    """Récupère les musiques récemment écoutées"""
    try:
        results = spotify_client.current_user_recently_played(limit=limit)
        
        tracks = []
        for item in results['items']:
            track = item['track']
            track_data = {
                'id': track['id'],
                'name': track['name'],
                'artists': [artist['name'] for artist in track['artists']],
                'preview_url': track.get('preview_url'),
                'uri': track['uri'],
                'external_urls': track['external_urls'],
                'album': track['album']['name'],
                'played_at': item['played_at']
            }
            tracks.append(track_data)
        
        return tracks
    except Exception as e:
        st.error(f"Erreur lors de la récupération des musiques récentes: {e}")
        return []

def convertSongToMatrice(audio_path, size=599):
    y, sr = librosa.load(audio_path)
    n_fft = (sr/10) / 2 + 3
    D = np.abs(librosa.stft(y, hop_length=int(n_fft)))
    spectrogram = librosa.feature.melspectrogram(S=D, sr=sr)
    S = librosa.util.fix_length(spectrogram, size=size)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
    return S_db_norm

def get_tracks_with_previews(tracks):
    """Filtre les tracks qui ont des previews disponibles"""
    if not tracks:
        return []
    
    tracks_with_previews = []
    for track in tracks:
        if track.get('preview_url'):
            track_data = {
                'id': track.get('id'),
                'name': track.get('name', 'Titre inconnu'),
                'artists': track.get('artists', []),
                'preview_url': track.get('preview_url'),
                'uri': track.get('uri'),
                'external_urls': track.get('external_urls', {})
            }
            tracks_with_previews.append(track_data)
    
    st.info(f"🎵 {len(tracks_with_previews)}/{len(tracks)} titres avec extraits audio")
    return tracks_with_previews

def analyze_spotify_track(track, track_index, source_type="top"):
    """Analyse une track Spotify avec gestion d'erreurs complète"""
    with st.spinner(f"Analyse de '{track['name'][:30]}'..."):
        # Créer un répertoire temporaire
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"spotify_{source_type}_{track_index}.mp3")
        
        # Télécharger et convertir l'audio
        downloaded_path, status = download_spotify_preview(track['preview_url'], audio_path)
        
        if not downloaded_path:
            st.error(f"❌ Échec du téléchargement: {status}")
            return None
        
        st.info(f"✅ Audio téléchargé: {status}")
        
        # Analyser le genre
        genre, confidence, probs = analyze_audio_genre(
            downloaded_path, st.session_state.model
        )
        
        if genre:
            # Formater les artistes
            artists = track.get('artists', 'Artiste inconnu')
            if isinstance(artists, list):
                artists = ', '.join([a if isinstance(a, str) else str(a) for a in artists])
            
            track_data = {
                'name': track['name'],
                'artists': artists,
                'source': 'spotify',
                'spotify_id': track.get('id'),
                'uri': track.get('uri'),
                'preview_url': track.get('preview_url'),
                'genre': genre,
                'confidence': confidence,
                'probabilities': probs
            }
            
            st.session_state.analyzed_tracks.append(track_data)
            st.success(f"✅ {track['name'][:40]} - {genre} ({confidence:.1%})")
            return track_data
        else:
            st.error(f"❌ Échec de l'analyse pour: {track['name'][:40]}")
            return None

def convert_audio_format(input_path, output_path):
    """Convertit un fichier audio en format WAV compatible avec librosa"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path, 
            '-acodec', 'pcm_s16le',
            '-ac', '1', 
            '-ar', '22050',
            '-y',  # Overwrite
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        else:
            st.warning(f"Conversion FFmpeg échouée: {result.stderr}")
            return None
    except Exception as e:
        st.warning(f"Erreur conversion audio: {e}")
        return None

def download_spotify_preview(preview_url, output_path):
    """Télécharge et convertit l'aperçu audio d'un titre Spotify"""
    try:
        st.info(f"📥 Téléchargement de l'extrait audio...")
        response = requests.get(preview_url, timeout=20)
        
        if response.status_code == 200:
            # Sauvegarder le fichier original
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Vérifier la taille du fichier
            file_size = os.path.getsize(output_path)
            if file_size < 1000:  # Fichier trop petit
                return None, "Extrait audio trop court ou corrompu"
            
            # Essayer la conversion en WAV
            wav_path = output_path.replace('.mp3', '.wav')
            converted_path = convert_audio_format(output_path, wav_path)
            
            if converted_path and os.path.exists(converted_path):
                return converted_path, "Conversion réussie"
            else:
                # Utiliser le fichier MP3 original
                return output_path, "Fichier MP3 original"
        else:
            return None, f"Erreur HTTP {response.status_code}"
    except Exception as e:
        return None, f"Erreur téléchargement: {str(e)}"

# Le reste des fonctions (load_model, analyze_audio_genre, perform_pca, generate_playlist_line_pytorch, export_playlist_to_spotify) reste identique

# Barre latérale
with st.sidebar:
    st.title("🎵 Spotify Analyzer")
    
    # Authentification
    spotify_client = get_spotify_client()
    
    if spotify_client:
        try:
            user_info = spotify_client.current_user()
            st.success(f"✅ Connecté en tant que **{user_info['display_name']}**")
            st.image(user_info['images'][0]['url'] if user_info.get('images') else "🎵", width=80)
        except Exception as e:
            st.error(f"Erreur de connexion: {e}")
            st.session_state.spotify_client = None
    else:
        # Bouton de connexion
        sp_oauth = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="user-library-read user-top-read playlist-modify-public playlist-modify-private user-read-recently-played user-read-email"
        )
        auth_url = sp_oauth.get_authorize_url()
        
        st.markdown(f"""
        **Étapes de connexion:**
        1. Cliquez sur le bouton ci-dessous
        2. Autorisez l'application sur Spotify  
        3. Vous serez redirigé vers cette page
        
        <a href="{auth_url}" target="_self">
            <button style="
                background-color: #1DB954; 
                color: white; 
                padding: 12px 24px; 
                border-radius: 25px; 
                text-align: center; 
                text-decoration: none; 
                display: inline-block;
                font-weight: bold;
                width: 100%;
                border: none;
                cursor: pointer;">
            🎵 Se connecter avec Spotify
            </button>
        </a>
        """, unsafe_allow_html=True)

    # Bouton de déconnexion
    if spotify_client:
        if st.button("🚪 Déconnexion", type="secondary"):
            st.session_state.spotify_client = None
            st.session_state.analyzed_tracks = []
            st.session_state.user_top_tracks = []
            if os.path.exists(".spotify_cache"):
                os.remove(".spotify_cache")
            st.rerun()

# Tabs principaux
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Accueil", "🎧 Mes Musiques", "📊 Analyse", "🎨 Playlist"])

# Tab 1: Accueil
with tab1:
    st.header("🎵 Analyseur de Musique Spotify")
    st.markdown("""
    Bienvenue dans votre analyseur de musique personnel !
    
    **Fonctionnalités :**
    - 📊 Analyse des genres de vos musiques préférées
    - 🎧 Récupération de vos musiques les plus écoutées
    - 🔍 Classification automatique par IA
    - 🎨 Création de playlists intelligentes
    
    **Commencez par :**
    1. Vous connecter à Spotify dans la barre latérale
    2. Aller dans l'onglet "🎧 Mes Musiques"
    3. Charger et analyser vos musiques
    """)

# Tab 2: Mes Musiques (NOUVEAU)
with tab2:
    st.header("🎧 Mes Musiques Spotify")
    
    if spotify_client:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎵 Mes titres les plus écoutés")
            time_range = st.selectbox(
                "Période :",
                ["short_term", "medium_term", "long_term"],
                format_func=lambda x: {"short_term": "4 dernières semaines", "medium_term": "6 derniers mois", "long_term": "Tout le temps"}[x]
            )
            
            if st.button("🔄 Charger mes tops titres"):
                with st.spinner("Récupération de vos musiques..."):
                    top_tracks = get_user_top_tracks(spotify_client, limit=20, time_range=time_range)
                    st.session_state.user_top_tracks = top_tracks
                    
                    if top_tracks:
                        tracks_with_previews = get_tracks_with_previews(top_tracks)
                        st.session_state.spotify_top_tracks = tracks_with_previews
                        st.success(f"✅ {len(tracks_with_previews)} titres chargés")
                    else:
                        st.error("❌ Aucun titre récupéré")
        
        with col2:
            st.subheader("🕐 Récemment écoutés")
            if st.button("🔄 Charger l'historique"):
                with st.spinner("Récupération de l'historique..."):
                    recent_tracks = get_user_recent_tracks(spotify_client, limit=20)
                    
                    if recent_tracks:
                        tracks_with_previews = get_tracks_with_previews(recent_tracks)
                        st.session_state.spotify_recent_tracks = tracks_with_previews
                        st.success(f"✅ {len(tracks_with_previews)} titres récents chargés")
                    else:
                        st.error("❌ Aucun historique récupéré")
        
        # Affichage et analyse des musiques
        if st.session_state.get('spotify_top_tracks'):
            st.subheader("📊 Analyse des musiques")
            
            tracks_to_analyze = st.session_state.spotify_top_tracks
            st.write(f"**{len(tracks_to_analyze)}** titres disponibles pour l'analyse")
            
            if st.button("🔍 Analyser tous les titres", type="primary"):
                analyzed_count = 0
                progress_bar = st.progress(0)
                
                for i, track in enumerate(tracks_to_analyze[:10]):  # Limiter à 10 pour les performances
                    result = analyze_spotify_track(track, i, "top")
                    if result:
                        analyzed_count += 1
                    progress_bar.progress((i + 1) / min(10, len(tracks_to_analyze)))
                
                st.success(f"✅ {analyzed_count} titres analysés avec succès!")
        
        # Affichage des musiques chargées
        if st.session_state.get('user_top_tracks'):
            st.subheader("🎼 Vos musiques les plus écoutées")
            for i, track in enumerate(st.session_state.user_top_tracks[:10]):
                with st.expander(f"{i+1}. {track['name']} - {', '.join(track['artists'])}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Album:** {track.get('album', 'N/A')}")
                        st.write(f"**Popularité:** {track.get('popularity', 'N/A')}/100")
                        if track.get('preview_url'):
                            st.audio(track['preview_url'], format="audio/mp3")
                        else:
                            st.warning("❌ Aucun extrait disponible")
                    with col2:
                        if track.get('external_urls', {}).get('spotify'):
                            st.markdown(f"[🎵 Ouvrir]({track['external_urls']['spotify']})")
    
    else:
        st.warning("⚠️ Veuillez vous connecter à Spotify pour voir vos musiques")


# Tab 3: Analyse
with tab3:
    st.header("📊 Analyse des genres")
    
    if st.session_state.analyzed_tracks:
        # Filtrer les tracks avec des données valides
        valid_tracks = [t for t in st.session_state.analyzed_tracks if t.get('genre') != 'Non analysé']
        
        if valid_tracks:
            df = pd.DataFrame(valid_tracks)
            
            # Statistiques
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total tracks", len(df))
            col2.metric("Genres uniques", df['genre'].nunique())
            col3.metric("Confiance moy.", f"{df['confidence'].mean():.1%}")
            
            # Distribution des genres
            st.subheader("Distribution des genres")
            genre_counts = df['genre'].value_counts()
            
            fig = px.bar(
                x=genre_counts.index, 
                y=genre_counts.values,
                color=genre_counts.index,
                color_discrete_map=genre_colors,
                labels={'x': 'Genre', 'y': 'Nombre de tracks'}
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # PCA Visualization
            if len(valid_tracks) >= 2:
                st.subheader("Visualisation PCA")
                
                features_list = [t['features'] for t in valid_tracks if t.get('features') is not None]
                
                if len(features_list) >= 2:
                    pca_result, pca_model, scaler = perform_pca(features_list)
                    
                    if pca_result is not None:
                        pca_df = pd.DataFrame({
                            'PC1': pca_result[:, 0],
                            'PC2': pca_result[:, 1],
                            'name': [t['name'][:30] for t in valid_tracks if t.get('features') is not None],
                            'genre': [t['genre'] for t in valid_tracks if t.get('features') is not None]
                        })
                        
                        fig = px.scatter(
                            pca_df, 
                            x='PC1', 
                            y='PC2',
                            color='genre',
                            color_discrete_map=genre_colors,
                            hover_data=['name'],
                            title='Espace PCA des tracks'
                        )
                        fig.update_traces(marker=dict(size=12))
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Stocker pour la génération de playlist
                        st.session_state.pca_df = pca_df
            
            # Tableau détaillé
            st.subheader("Détails des tracks")
            display_df = df[['name', 'genre', 'confidence', 'source']].copy()
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("👆 Analysez des tracks pour voir les statistiques")
    else:
        st.info("👆 Ajoutez des tracks pour voir l'analyse")

# Tab 4: Génération de playlist
with tab4:
    st.header("🎨 Générateur de Playlist")
    
    if 'pca_df' in st.session_state and len(st.session_state.pca_df) >= 2:
        st.markdown("Sélectionnez deux tracks pour créer une playlist progressive entre elles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            track1_idx = st.selectbox(
                "Track de départ:",
                range(len(st.session_state.pca_df)),
                format_func=lambda x: f"{st.session_state.pca_df.iloc[x]['name']} ({st.session_state.pca_df.iloc[x]['genre']})"
            )
        
        with col2:
            track2_idx = st.selectbox(
                "Track d'arrivée:",
                range(len(st.session_state.pca_df)),
                index=min(1, len(st.session_state.pca_df)-1),
                format_func=lambda x: f"{st.session_state.pca_df.iloc[x]['name']} ({st.session_state.pca_df.iloc[x]['genre']})"
            )
        
        num_tracks = st.slider("Nombre de tracks dans la playlist:", 5, 20, 10)
        
        if st.button("🎯 Générer la playlist", type="primary"):
            if track1_idx != track2_idx:
                with st.spinner("Génération..."):
                    playlist, line_points, p1, p2 = generate_playlist_line(
                        st.session_state.pca_df, track1_idx, track2_idx, num_tracks
                    )
                    
                    st.session_state.generated_playlist = playlist
                    st.session_state.line_points = line_points
                    st.session_state.p1 = p1
                    st.session_state.p2 = p2
                    st.success(f"✅ Playlist de {len(playlist)} tracks générée!")
            else:
                st.warning("⚠️ Sélectionnez deux tracks différentes")
        
        # Afficher la playlist générée
        if 'generated_playlist' in st.session_state:
            st.subheader("📝 Playlist générée")
            
            # Visualisation
            fig = go.Figure()
            
            # Tous les points
            for genre in st.session_state.pca_df['genre'].unique():
                mask = st.session_state.pca_df['genre'] == genre
                genre_data = st.session_state.pca_df[mask]
                fig.add_trace(go.Scatter(
                    x=genre_data['PC1'],
                    y=genre_data['PC2'],
                    mode='markers',
                    name=genre,
                    marker=dict(size=8, color=genre_colors.get(genre, '#999'), opacity=0.3),
                    text=genre_data['name'],
                    hovertemplate='%{text}<br>Genre: ' + genre
                ))
            
            # Ligne de la playlist
            fig.add_trace(go.Scatter(
                x=st.session_state.line_points[:, 0],
                y=st.session_state.line_points[:, 1],
                mode='lines',
                name='Trajectoire',
                line=dict(color='black', width=2, dash='dash')
            ))
            
            # Points de départ et d'arrivée
            fig.add_trace(go.Scatter(
                x=[st.session_state.p1[0]],
                y=[st.session_state.p1[1]],
                mode='markers',
                name='Départ',
                marker=dict(size=20, color='blue', symbol='star')
            ))
            
            fig.add_trace(go.Scatter(
                x=[st.session_state.p2[0]],
                y=[st.session_state.p2[1]],
                mode='markers',
                name='Arrivée',
                marker=dict(size=20, color='green', symbol='star')
            ))
            
            # Tracks de la playlist
            playlist_x = [t['PC1'] for t in st.session_state.generated_playlist]
            playlist_y = [t['PC2'] for t in st.session_state.generated_playlist]
            playlist_names = [t['name'] for t in st.session_state.generated_playlist]
            playlist_genres = [t['genre'] for t in st.session_state.generated_playlist]
            
            fig.add_trace(go.Scatter(
                x=playlist_x,
                y=playlist_y,
                mode='markers+text',
                name='Playlist',
                marker=dict(size=15, color='red', line=dict(width=2, color='black')),
                text=[str(i+1) for i in range(len(playlist_x))],
                textposition='middle center',
                textfont=dict(color='white', size=10, family='Arial Black'),
                hovertext=playlist_names,
                hovertemplate='%{hovertext}<br>Genre: ' + '%{customdata}',
                customdata=playlist_genres
            ))
            
            fig.update_layout(
                title='Visualisation de la playlist dans l\'espace PCA',
                xaxis_title='Composante Principale 1',
                yaxis_title='Composante Principale 2',
                height=700,
                showlegend=True,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Liste détaillée
            st.subheader("🎵 Ordre de lecture")
            
            for track in st.session_state.generated_playlist:
                col1, col2, col3 = st.columns([1, 5, 2])
                
                with col1:
                    st.markdown(f"**#{track['position']}**")
                
                with col2:
                    st.markdown(f"**{track['name']}**")
                    st.caption(f"Genre: {track['genre']}")
                
                with col3:
                    st.metric("Distance", f"{track['distance']:.3f}")
            
            # Métriques de qualité
            st.subheader("📈 Métriques de qualité")
            
            genres_in_playlist = [t['genre'] for t in st.session_state.generated_playlist]
            unique_genres = len(set(genres_in_playlist))
            avg_distance = np.mean([t['distance'] for t in st.session_state.generated_playlist])
            
            # Calculer la fluidité (distance entre tracks consécutives)
            smoothness_distances = []
            for i in range(len(st.session_state.generated_playlist) - 1):
                p1 = np.array([
                    st.session_state.generated_playlist[i]['PC1'],
                    st.session_state.generated_playlist[i]['PC2']
                ])
                p2 = np.array([
                    st.session_state.generated_playlist[i+1]['PC1'],
                    st.session_state.generated_playlist[i+1]['PC2']
                ])
                smoothness_distances.append(np.linalg.norm(p2 - p1))
            
            avg_smoothness = np.mean(smoothness_distances) if smoothness_distances else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tracks", len(st.session_state.generated_playlist))
            col2.metric("Genres uniques", unique_genres)
            col3.metric("Distance moy.", f"{avg_distance:.3f}")
            col4.metric("Fluidité", f"{avg_smoothness:.3f}")
            
            # Distribution des genres dans la playlist
            genre_dist = pd.Series(genres_in_playlist).value_counts()
            
            fig_dist = px.bar(
                x=genre_dist.index,
                y=genre_dist.values,
                color=genre_dist.index,
                color_discrete_map=genre_colors,
                title='Distribution des genres dans la playlist',
                labels={'x': 'Genre', 'y': 'Nombre'}
            )
            fig_dist.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Export vers Spotify
            if spotify_client:
                st.subheader("💾 Exporter la playlist")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    playlist_name = st.text_input("Nom de la playlist:", 
                                                value=f"Playlist IA - {datetime.now().strftime('%d/%m/%Y')}")
                    playlist_description = st.text_area("Description:",
                                                      value="Playlist générée automatiquement par l'IA")
                
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("📤 Créer sur Spotify", type="primary", use_container_width=True):
                        # Préparer les données pour l'export
                        export_tracks = []
                        for playlist_track in st.session_state.generated_playlist:
                            # Trouver la track originale dans analyzed_tracks
                            for original_track in st.session_state.analyzed_tracks:
                                if original_track.get('name') == playlist_track['name']:
                                    export_tracks.append(original_track)
                                    break
                        
                        if export_tracks:
                            playlist = export_playlist_to_spotify(
                                spotify_client,
                                export_tracks,
                                playlist_name,
                                playlist_description
                            )
                        else:
                            st.warning("⚠️ Aucune track Spotify trouvée pour l'export")
    
    else:
        st.info("👆 Analysez au moins 2 tracks dans l'onglet Analyse pour générer une playlist")
