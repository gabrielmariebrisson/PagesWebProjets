import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
import tempfile
import yt_dlp
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
from templates.assets.music.architecture import SimpleCNN
load_dotenv()


from deep_translator import GoogleTranslator

# --- Configuration de la traduction automatique ---
LANGUAGES = {
    "fr": "🇫🇷 Français",
    "en": "🇬🇧 English",
    "es": "🇪🇸 Español",
    "de": "🇩🇪 Deutsch",
    "it": "🇮🇹 Italiano",
    "pt": "🇵🇹 Português",
    "ja": "🇯🇵 日本語",
    "zh-CN": "🇨🇳 中文",
    "ar": "🇸🇦 العربية",
    "ru": "🇷🇺 Русский"
}

# Initialisation de la langue
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

# Sélecteur de langue
lang = st.sidebar.selectbox(
    "🌐 Language / Langue", 
    options=list(LANGUAGES.keys()),
    format_func=lambda x: LANGUAGES[x],
    index=list(LANGUAGES.keys()).index(st.session_state.language)
)

st.session_state.language = lang

# Cache pour les traductions (évite de retranduire à chaque fois)
if 'translations_cache' not in st.session_state:
    st.session_state.translations_cache = {}

def _(text):
    """Fonction de traduction automatique avec cache"""
    if lang == 'fr':
        return text
    
    # Vérifier le cache
    cache_key = f"{lang}_{text}"
    if cache_key in st.session_state.translations_cache:
        return st.session_state.translations_cache[cache_key]
    
    # Traduire
    try:
        translated = GoogleTranslator(source='fr', target=lang).translate(text)
        st.session_state.translations_cache[cache_key] = translated
        return translated
    except:
        return text
    

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Music Playlist Generator",
    page_icon="🎵",
    layout="wide"
)

# --- CONSTANTES ET CONFIGURATION ---
LABEL_MAPPING = {
    0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
    5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
}

GENRE_COLORS = {
    'blues': '#1f77b4', 'classical': '#ff7f0e', 'country': '#2ca02c',
    'disco': '#d62728', 'hiphop': '#9467bd', 'jazz': '#8c564b',
    'metal': '#e377c2', 'pop': '#7f7f7f', 'reggae': '#bcbd22', 'rock': '#17becf'
}

SPOTIFY_SCOPE = "user-library-read user-top-read playlist-modify-public playlist-modify-private user-read-recently-played"
DEEZER_BASE_URL = "https://api.deezer.com"

# --- FONCTIONS DEEZER ---
def search_deezer_tracks(query, limit=10):
    """Recherche des tracks sur Deezer"""
    try:
        url = f"{DEEZER_BASE_URL}/search"
        params = {
            'q': query,
            'limit': limit
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            tracks = []
            
            for item in data.get('data', []):
                track = {
                    'id': item.get('id'),
                    'name': item.get('title'),
                    'artists': item.get('artist', {}).get('name', ''),
                    'preview_url': item.get('preview'),
                    'album': item.get('album', {}).get('title', ''),
                    'duration': item.get('duration'),
                    'deezer_id': item.get('id')
                }
                tracks.append(track)
            
            return tracks
        else:
            st.error(f"Erreur API Deezer: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Erreur recherche Deezer: {str(e)}")
        return []

def download_deezer_preview(preview_url, output_path):
    """Télécharge l'extrait audio de 30s depuis Deezer"""
    try:
        if not preview_url:
            return False
            
        response = requests.get(preview_url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Erreur téléchargement Deezer: {str(e)}")
        return False

def get_deezer_track_info(track_id):
    """Récupère les informations d'une track Deezer"""
    try:
        url = f"{DEEZER_BASE_URL}/track/{track_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'id': data.get('id'),
                'name': data.get('title'),
                'artists': data.get('artist', {}).get('name', ''),
                'preview_url': data.get('preview'),
                'album': data.get('album', {}).get('title', ''),
                'duration': data.get('duration'),
                'deezer_id': data.get('id')
            }
        return None
    except Exception as e:
        st.error(f"Erreur récupération track Deezer: {str(e)}")
        return None

# --- ARCHITECTURE DU MODÈLE ---
def convert_song_to_matrix(audio_path, size=599):
    """Convertit un fichier audio en spectrogramme normalisé"""
    try:
        y, sr = librosa.load(audio_path, duration=30)
        n_fft = int((sr/10) / 2 + 3)
        D = np.abs(librosa.stft(y, hop_length=int(n_fft)))
        spectrogram = librosa.feature.melspectrogram(S=D, sr=sr)
        S = librosa.util.fix_length(spectrogram, size=size)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
        return S_db_norm
    except Exception as e:
        st.error(f"Erreur conversion audio: {str(e)}")
        return None

@st.cache_resource
def load_model(model_path):
    """Charge le modèle CNN pré-entraîné"""
    try:
        model = SimpleCNN()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle: {str(e)}")
        return None

def extract_features(model, spectrogram_tensor):
    """Extrait les features avant la couche de classification"""
    with torch.no_grad():
        x = spectrogram_tensor
        x = model.norm1(x)
        x = model.conv1(x)
        x = model.relu(x)
        x = model.cbam1(x)
        x = model.pool1(x)
        x = torch.permute(x,(0,2,1,3))
        
        x = model.norm2(x)
        x = model.conv2(x)
        x = model.relu(x)
        x = model.cbam1(x)
        x = model.pool2(x)
        x = torch.permute(x,(0,2,1,3))
        
        x = model.norm3(x)
        x = model.conv3(x)
        x = model.relu(x)
        x = model.cbam2(x)
        x = model.pool2(x)
        x = torch.permute(x,(0,2,1,3))
        
        x = model.norm4(x)
        x = model.conv4(x)
        x = model.cbam2(x)
        x = model.relu(x)
        x = torch.permute(x,(0,2,1,3))
        
        mean_values = torch.mean(x, dim=3, keepdim=True)
        max_values, _ = torch.max(x, dim=3, keepdim=True)
        l2_norm = torch.linalg.norm(x, dim=3, ord=2, keepdim=True)
        
        x = torch.cat([max_values, mean_values, l2_norm], dim=1)
        x = x.view(-1, 1536)
        
        x = model.normfc2(x)
        x = model.fc2(x)
        features = F.relu(x)
        
        return features.cpu().numpy()

def analyze_audio_genre(audio_path, model):
    """Analyse un fichier audio et prédit son genre"""
    spectrogram = convert_song_to_matrix(audio_path)
    
    if spectrogram is None:
        return None, None, None, None
    
    if model is None:
        genre_id = np.random.randint(0, 10)
        confidence = np.random.uniform(0.6, 0.95)
        return LABEL_MAPPING[genre_id], confidence, None, None
    
    try:
        spectrogram_tensor = torch.tensor(spectrogram).unsqueeze(0).unsqueeze(0).float()
        features = extract_features(model, spectrogram_tensor)
        
        with torch.no_grad():
            output = model(spectrogram_tensor)
            probabilities = F.softmax(output, dim=1)
            genre_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][genre_id].item()
            all_probs = probabilities[0].cpu().numpy()
        
        return LABEL_MAPPING[genre_id], confidence, features[0], all_probs
    except Exception as e:
        st.error(f"Erreur prédiction: {str(e)}")
        return None, None, None, None

def perform_pca(features_list):
    """Effectue une PCA sur les features extraites"""
    if len(features_list) < 2:
        return None, None, None
    
    features_array = np.array(features_list)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    return features_pca, pca, scaler

def generate_playlist_line(pca_df, track1_idx, track2_idx, num_tracks=10):
    """Génère une playlist linéaire entre deux tracks"""
    p1 = np.array([pca_df.iloc[track1_idx]['PC1'], pca_df.iloc[track1_idx]['PC2']])
    p2 = np.array([pca_df.iloc[track2_idx]['PC1'], pca_df.iloc[track2_idx]['PC2']])
    
    t_values = np.linspace(0, 1, num_tracks)
    line_points = np.array([p1 + t * (p2 - p1) for t in t_values])
    
    playlist_tracks = []
    used_tracks = {track1_idx, track2_idx}
    
    for i, target_point in enumerate(line_points):
        distances = []
        for idx, row in pca_df.iterrows():
            if idx not in used_tracks:
                track_point = np.array([row['PC1'], row['PC2']])
                distance = np.linalg.norm(track_point - target_point)
                distances.append((distance, idx, row))
        
        if distances:
            distances.sort(key=lambda x: x[0])
            closest_distance, closest_idx, closest_row = distances[0]
            
            playlist_tracks.append({
                'position': i + 1,
                'index': closest_idx,
                'name': closest_row['name'],
                'genre': closest_row['genre'],
                'distance': closest_distance,
                'PC1': closest_row['PC1'],
                'PC2': closest_row['PC2'],
                'uri': closest_row.get('uri'),
                'spotify_id': closest_row.get('spotify_id')
            })
            used_tracks.add(closest_idx)
    
    return playlist_tracks, line_points, p1, p2

# --- FONCTIONS SPOTIFY ---
def get_spotify_oauth():
    """Retourne l'objet OAuth configuré pour Spotify"""
    CLIENT_ID = os.getenv('CLIENT_ID_SPOTIFY')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET_SPOTIFY')
    REDIRECT_URI = os.getenv('REDIRECT_URI_SPOTIFY', 'http://localhost:8501')
    
    if not CLIENT_ID or not CLIENT_SECRET:
        return None
    
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SPOTIFY_SCOPE,
        cache_path=".spotify_cache",
        show_dialog=True
    )

def get_spotify_client():
    """Obtient un client Spotify authentifié"""
    try:
        sp_oauth = get_spotify_oauth()
        if not sp_oauth:
            return None
        
        # Vérifier le token en cache
        token_info = sp_oauth.get_cached_token()
        
        # Gérer la redirection après authentification
        query_params = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
        
        if "code" in query_params:
            code = query_params["code"]
            if isinstance(code, list):
                code = code[0]
            
            token_info = sp_oauth.get_access_token(code, as_dict=False)
            # Nettoyer l'URL
            if hasattr(st, "query_params"):
                st.query_params.clear()
            else:
                st.experimental_set_query_params()
            st.rerun()
        
        # Si token expiré, rafraîchir
        if token_info and sp_oauth.is_token_expired(token_info):
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        
        if token_info:
            return spotipy.Spotify(auth=token_info['access_token'])
        
        return None
    except Exception as e:
        st.error(_(f"Erreur d'authentification Spotify: {e}"))
        return None

def export_playlist_to_spotify(spotify_client, playlist_tracks, playlist_name, playlist_description=""):
    """Exporte une playlist vers Spotify"""
    try:
        if not spotify_client:
            st.error(_("❌ Client Spotify non disponible"))
            return None
        
        user_info = spotify_client.current_user()
        user_id = user_info['id']
        
        playlist = spotify_client.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=False,
            description=playlist_description
        )
        
        track_uris = []
        for track in playlist_tracks:
            if track.get('uri'):
                track_uris.append(track['uri'])
            elif track.get('spotify_id'):
                track_uris.append(f"spotify:track:{track['spotify_id']}")
        
        if track_uris:
            for i in range(0, len(track_uris), 100):
                batch = track_uris[i:i+100]
                spotify_client.playlist_add_items(playlist['id'], batch)
            
            st.success(_(f"✅ Playlist '{playlist_name}' créée avec {len(track_uris)} titres!"))
            st.markdown(_(f"[🎵 Ouvrir dans Spotify]{playlist['external_urls']['spotify']})"))
            return playlist
        else:
            st.warning(_("⚠️ Aucun URI Spotify trouvé pour les tracks sélectionnées"))
            return None
    except Exception as e:
        st.error(_(f"❌ Erreur lors de la création de la playlist: {e}"))
        return None

def get_user_saved_tracks(spotify_client, limit=50):
    """Récupère les titres sauvegardés de l'utilisateur"""
    try:
        results = spotify_client.current_user_saved_tracks(limit=limit)
        tracks = []
        
        for item in results['items']:
            track = item['track']
            
            tracks.append({
                'id': track['id'],
                'name': track['name'],
                'artists': ', '.join([artist['name'] for artist in track['artists']]),
                'preview_url': track.get('preview_url'),
                'uri': track['uri'],
                'external_url': track['external_urls']['spotify']
            })
        
        return tracks
    except Exception as e:
        st.error(f"Erreur lors de la récupération des titres sauvegardés: {e}")
        return []
    
def get_user_top_tracks(spotify_client, limit=20):
    """Récupère les top tracks de l'utilisateur"""
    try:
        top_tracks = spotify_client.current_user_top_tracks(limit=limit, time_range='medium_term')
        tracks = []
        
        for track in top_tracks['items']:
            tracks.append({
                'id': track['id'],
                'name': track['name'],
                'artists': ', '.join([artist['name'] for artist in track['artists']]),
                'preview_url': track.get('preview_url'),
                'uri': track['uri'],
                'external_url': track['external_urls']['spotify']
            })
        
        return tracks
    except Exception as e:
        st.error(_(f"Erreur lors de la récupération des top tracks: {e}"))
        return []

def search_tracks_with_preview(spotify_client, query, limit=10):
    """Recherche des tracks Spotify avec preview"""
    try:
        results = spotify_client.search(q=query, type='track', limit=limit)
        tracks = []
        
        for track in results['tracks']['items']:
            if track.get('preview_url'):
                tracks.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artists': ', '.join([artist['name'] for artist in track['artists']]),
                    'preview_url': track.get('preview_url'),
                    'uri': track['uri'],
                    'external_url': track['external_urls']['spotify']
                })
        
        return tracks
    except Exception as e:
        st.error(_(f"Erreur recherche Spotify: {e}"))
        return []

# --- FONCTIONS D'ANALYSE ---

def process_track_analysis(track, track_data):
    """Traite l'analyse d'une track"""
    try:
        if st.session_state.model:
            with st.spinner("Analyse en cours..."):
                temp_dir = tempfile.mkdtemp()
                audio_path = os.path.join(temp_dir, f"{track_data['type']}_{track_data['index']}.mp3")
                
                # Télécharger l'extrait selon la source
                download_success = False
                download_success = download_deezer_preview(track['preview_url'], audio_path)
                
                
                if download_success:
                    genre, confidence, features, probs = analyze_audio_genre(
                        audio_path, st.session_state.model
                    )
                    
                    if genre:
                        artists = track.get('artists', '')
                        analyzed_track = {
                            'name': track['name'],
                            'artists': artists,
                            'spotify_id': track.get('id'),
                            'deezer_id': track.get('deezer_id'),
                            'uri': track.get('uri'),
                            'preview_url': track.get('preview_url'),
                            'genre': genre,
                            'confidence': confidence,
                            'features': features,
                            'probabilities': probs
                        }
                        st.session_state.analyzed_tracks.append(analyzed_track)
                        st.success(f"✅ {track['name']} - {genre} ({confidence:.1%})")
                        return True
                else:
                    st.info(_("ℹ️ Aperçu audio non disponible pour l'analyse"))
        else:
            st.warning(_("Modèle non chargé"))
        return False
    except Exception as e:
        st.error(_(f"Erreur lors de l'analyse: {e}"))
        return False

def process_track_addition(track):
    """Traite l'ajout d'une track sans analyse"""
    try:
        artists = track.get('artists', '')
        track_data = {
            'name': track['name'],
            'artists': artists,
            'spotify_id': track.get('id'),
            'deezer_id': track.get('deezer_id'),
            'uri': track.get('uri'),
            'preview_url': track.get('preview_url'),
            'genre': 'Non analysé',
            'confidence': 0.0
        }
        st.session_state.analyzed_tracks.append(track_data)
        st.success(_(f"✅ {track['name']} ajouté!"))
        return True
    except Exception as e:
        st.error(_(f"Erreur lors de l'ajout: {e}"))
        return False


# Bouton de redirection
st.markdown(
    f"""
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
    {_("Retour")}
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


# --- INITIALISATION SESSION STATE ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'analyzed_tracks' not in st.session_state:
    st.session_state.analyzed_tracks = []
if 'spotify_top_tracks' not in st.session_state:
    st.session_state.spotify_top_tracks = []
if 'spotify_saved_tracks' not in st.session_state:
    st.session_state.spotify_saved_tracks = []
if 'current_playing_track' not in st.session_state:
    st.session_state.current_playing_track = None
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'deezer_search_results' not in st.session_state:
    st.session_state.deezer_search_results = []

# --- INTERFACE STREAMLIT PRINCIPALE ---
st.title("🎵 Music Playlist Generator")
st.markdown("Créez des playlists personnalisées avec l'IA - Analyse de genres musicaux par CNN")

# Barre latérale
with st.sidebar:
    st.header(_("⚙️ Configuration"))
    
    # Charger le modèle
    st.subheader(_("🧠 Modèle CNN"))
    
    # Essayer plusieurs chemins possibles
    model_path = "templates/assets/music/best_model_original_loss.pth"
    
    model_loaded = False
    with st.spinner(_("Chargement...")):
        st.session_state.model = load_model(model_path)
    
    if not model_loaded:
        st.error(_("❌ Aucun modèle trouvé"))

    # Connexion Spotify
    st.subheader(_("🔗 Connexion Spotify"))
    
    CLIENT_ID_SPOTIFY = os.getenv('CLIENT_ID_SPOTIFY')
    CLIENT_SECRET_SPOTIFY = os.getenv('CLIENT_SECRET_SPOTIFY')
    
    spotify_client = get_spotify_client()


    sp_oauth = get_spotify_oauth()
    if sp_oauth:
        auth_url = sp_oauth.get_authorize_url()
        st.markdown(_(f"""
        **Étapes de connexion:** 
        1. Cliquez sur le bouton ci-dessous
        2. Autorisez l'application sur Spotify  
        3. Vous serez redirigé vers cette page"""), unsafe_allow_html=True)
        st.markdown(f"""
        
        <a href="{auth_url}" target="_self">
            <div style="
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
            {_("🎵 Se connecter avec Spotify")}
            </div>
        </a>
        """, unsafe_allow_html=True)

    # Bouton de reset
    st.markdown("---")
    if st.button(_("🔄 Réinitialiser tout"), type="secondary"):
        for key in list(st.session_state.keys()):
            if key != 'model':
                del st.session_state[key]
        st.success(_("✅ Réinitialisé!"))
        st.rerun()

# Tabs principaux
tab1, tab2, tab3, tab4, tab5 = st.tabs([_("🏠 Accueil"), _("🎧 Mes Musiques"), _("🔍 Recherche Deezer"), _("📊 Analyse"), _("🎨 Playlist")])

# Tab 1: Accueil
with tab1:
    st.header(_("🎵 Analyseur de Musique"))
    st.markdown(_("""
    Bienvenue dans votre analyseur de musique personnel !
    
    **Fonctionnalités :**
    - 📊 Analyse des genres de vos musiques préférées
    - 🎧 Récupération de vos musiques depuis Spotify
    - 🔍 Recherche et analyse depuis Deezer
    - 🔍 Classification automatique par IA
    - 🎨 Création de playlists intelligentes
    
    **Commencez par :**
    1. Vous connecter à Spotify dans la barle latérale (optionnel)
    2. Aller dans l'onglet "🎧 Mes Musiques" ou "🔍 Recherche Deezer"
    3. Charger et analyser vos musiques
    """))

# Tab 2: Mes Musiques (Spotify)
with tab2:
    st.header(_("🎧 Importer depuis Spotify"))
    
    if spotify_client:
        # Section Titres Sauvegardés
        st.subheader(_("💾 Mes titres sauvegardés"))
        
        col1, col2 = st.columns([3, 1])
        with col1:
            limit_tracks = st.slider(_("Nombre de titres à importer:"), 10, 100, 20, key="saved_slider")
        
        with col2:
            if st.button(_("🔄 Charger mes titres"), key="load_saved"):
                with st.spinner(_("Récupération de vos titres...")):
                    saved_tracks = get_user_saved_tracks(spotify_client, limit=limit_tracks)
                    st.session_state.spotify_saved_tracks = saved_tracks
                    st.success(_(f"✅ {len(saved_tracks)} titres chargés!"))
        
        # Afficher les titres sauvegardés
        if st.session_state.spotify_saved_tracks:
            st.subheader(_(f"Vos titres ({len(st.session_state.spotify_saved_tracks)})"))
            
            for i, track in enumerate(st.session_state.spotify_saved_tracks[:20]):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{track['name']}**")
                    st.caption(_(f"Artiste: {track['artists']}"))
                
                with col3:
                    if st.button(_("🔍 Analyser"), key=f"analyze_saved_{i}"):
                        if process_track_analysis(track, {"type": "saved", "index": i}):
                            st.rerun()
                
                with col4:
                    if st.button(_("➕ Ajouter"), key=f"add_saved_{i}"):
                        if process_track_addition(track):
                            st.rerun()
        
        st.markdown("---")
        
        # Section Top Tracks
        st.subheader(_("🎯 Top tracks"))
        
        if st.button(_("📊 Charger top tracks")):
            with st.spinner(_("Récupération...")):
                top_tracks = get_user_top_tracks(spotify_client, limit=20)
                st.session_state.spotify_top_tracks = top_tracks
                st.success(_(f"✅ {len(top_tracks)} top tracks chargés"))
                st.rerun()
        
        # Afficher les top tracks
        if st.session_state.spotify_top_tracks:
            st.markdown(_("### Vos top tracks"))
            
            for i, track in enumerate(st.session_state.spotify_top_tracks[:20]):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"**{track['name']}**")
                    st.caption(_(f"Artiste: {track['artists']}"))
                
                with col3:
                    if st.button(_("🔍 Analyser"), key=f"analyze_top_{i}"):
                        if process_track_analysis(track, {"type": "top", "index": i}):
                            st.rerun()
                
                with col4:
                    if st.button(_("➕ Ajouter"), key=f"add_top_{i}"):
                        if process_track_addition(track):
                            st.rerun()
        
        st.markdown("---")
        
        # Section Recherche Spotify
        st.subheader(_("🔍 Rechercher des titres Spotify"))
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(_("Rechercher un titre:"), placeholder=_("Nom de la chanson ou artiste..."), key="spotify_search")
        with col2:
            search_limit = st.slider(_("Résultats"), 5, 20, 10, key="search_slider")
        
        if search_query:
            with st.spinner(_("Recherche en cours...")):
                search_results = search_tracks_with_preview(spotify_client, search_query, search_limit)
                
                if search_results:
                    st.success(_(f"✅ {len(search_results)} titres trouvés!"))
                    
                    for i, track in enumerate(search_results):
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**{track['name']}**")
                            st.caption(_(f"Artiste: {track['artists']}"))
                        
                        with col3:
                            if st.button(_("🔍 Analyser"), key=f"analyze_search_{i}"):
                                if process_track_analysis(track, {"type": "search", "index": i}):
                                    st.rerun()
                        
                        with col4:
                            if st.button(_("➕ Ajouter"), key=f"add_search_{i}"):
                                if process_track_addition(track):
                                    st.rerun()
                else:
                    st.warning(_("Aucun titre trouvé pour cette recherche."))
    else:
        st.warning(_("⚠️ Connectez-vous à Spotify pour importer vos musiques"))

# Nouveau Tab: Recherche Deezer
with tab3:
    st.header(_("🔍 Recherche Deezer"))
    st.markdown(_("Recherchez et analysez des morceaux directement depuis Deezer (extraits de 30 secondes)"))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        deezer_query = st.text_input(_("Rechercher un titre sur Deezer:"), placeholder="Nom de la chanson ou artiste...")
    with col2:
        deezer_limit = st.slider(_("Nombre de résultats"), 5, 20, 10, key="deezer_slider")
    
    if deezer_query:
        if st.button(_("🔍 Rechercher sur Deezer"), type="primary"):
            with st.spinner(_("Recherche en cours...")):
                deezer_results = search_deezer_tracks(deezer_query, deezer_limit)
                st.session_state.deezer_search_results = deezer_results
                
                if deezer_results:
                    st.success(_(f"✅ {len(deezer_results)} titres trouvés sur Deezer!"))
                else:
                    st.warning(_("Aucun titre trouvé sur Deezer."))
    
    # Afficher les résultats Deezer
    if st.session_state.deezer_search_results:
        st.subheader(_("Résultats Deezer"))
        
        for i, track in enumerate(st.session_state.deezer_search_results):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{track['name']}**")
                st.caption(_(f"Artiste: {track['artists']}"))
                if track.get('album'):
                    st.caption(_(f"Album: {track['album']}"))
            
            with col2:
                if track.get('preview_url'):
                    st.markdown(_("🎵 **30s**"))
                else:
                    st.markdown(_("❌ **No preview**"))
            
            with col3:
                if st.button(_("🎵 Écouter"), key=f"preview_deezer_{i}"):
                    if track.get('preview_url'):
                        st.audio(track['preview_url'], format="audio/mp3")
                    else:
                        st.warning("Aucun extrait disponible")
            
            with col4:
                if st.button(_("🔍 Analyser"), key=f"analyze_deezer_{i}"):
                    if process_track_analysis(track, {"type": "deezer", "index": i}):
                        st.rerun()
            
            with col5:
                if st.button(_("➕ Ajouter"), key=f"add_deezer_{i}"):
                    if process_track_addition(track):
                        st.rerun()

# Tab 4: Analyse
with tab4:
    st.header(_("📊 Analyse des genres"))
    
    if st.session_state.analyzed_tracks:
        valid_tracks = [t for t in st.session_state.analyzed_tracks if t.get('genre') != 'Non analysé']
        
        if valid_tracks:
            df = pd.DataFrame(valid_tracks)
            
            # Statistiques
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total tracks", len(df))
            col2.metric("Genres uniques", df['genre'].nunique())
            col3.metric("Confiance moy.", f"{df['confidence'].mean():.1%}")

            # Distribution des genres
            st.subheader(_("Distribution des genres"))
            genre_counts = df['genre'].value_counts()
            
            fig = px.bar(
                x=genre_counts.index, 
                y=genre_counts.values,
                color=genre_counts.index,
                color_discrete_map=GENRE_COLORS,
                labels={'x': 'Genre', 'y': 'Nombre de tracks'}
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # PCA Visualization
            if len(valid_tracks) >= 2:
                st.subheader(_("Visualisation PCA"))
                
                features_list = [t['features'] for t in valid_tracks if t.get('features') is not None]
                
                if len(features_list) >= 2:
                    pca_result, pca_model, scaler = perform_pca(features_list)
                    
                    if pca_result is not None:
                        pca_df = pd.DataFrame({
                            'PC1': pca_result[:, 0],
                            'PC2': pca_result[:, 1],
                            'name': [t['name'][:30] for t in valid_tracks if t.get('features') is not None],
                            'genre': [t['genre'] for t in valid_tracks if t.get('features') is not None],
                            'artists': [t['artists'] for t in valid_tracks if t.get('features') is not None],
                            'uri': [t.get('uri') for t in valid_tracks if t.get('features') is not None],
                            'spotify_id': [t.get('spotify_id') for t in valid_tracks if t.get('features') is not None],
                        })
                        
                        fig = px.scatter(
                            pca_df, 
                            x='PC1', 
                            y='PC2',
                            color='genre',
                            color_discrete_map=GENRE_COLORS,
                            hover_data=['name', 'artists'],
                            title='Espace PCA des tracks'
                        )
                        fig.update_traces(marker=dict(size=12))
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.session_state.pca_df = pca_df
            
            st.subheader(_("Détails des tracks"))
            
            for idx, track in enumerate(valid_tracks[:20]):
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{track['name']}**")
                    st.caption(f"{track['artists']} - {track['genre']} ({track['confidence']:.1%})")
                
                with col3:
                    if track.get('preview_url'):
                        if st.button("🎵", key=f"preview_{idx}"):
                            st.audio(track['preview_url'], format="audio/mp3")
                
                with col4:
                    st.markdown(f"🎯 {track['confidence']:.1%}")
                
                with col5:
                    st.markdown(f"**{track['genre']}**")
        else:
            st.info(_("👆 Analysez des tracks pour voir les statistiques"))
    else:
        st.info(_("👆 Ajoutez des tracks pour voir l'analyse"))

# Tab 5: Playlist
with tab5:
    st.header(_("🎨 Générateur de Playlist"))
    
    # Vérifier les prérequis
    prerequisites_ok = True
    
    if len(st.session_state.analyzed_tracks) < 2:
        st.warning(_("⚠️ **Prérequis**: Analysez au moins 2 morceaux"))
        prerequisites_ok = False
    
    if 'pca_df' not in st.session_state:
        st.warning(_("⚠️ **Prérequis**: Effectuez d'abord l'analyse PCA dans l'onglet '📊 Analyse'"))
        prerequisites_ok = False
    
    if not prerequisites_ok:
        st.info(_("""
        **Étapes pour créer une playlist:**
        1. 🎧 **Mes Musiques** ou 🔍 **Recherche Deezer**: Importez et analysez vos morceaux
        2. 📊 **Analyse**: Laissez le système analyser les caractéristiques audio  
        3. 🎨 **Playlist**: Créez votre playlist personnalisée ici
        """))
        
        # Afficher l'état actuel
        st.subheader(_("État actuel"))
        col1, col2 = st.columns(2)
        with col1:
            st.metric(_("Morceaux analysés"), len(st.session_state.analyzed_tracks))
        with col2:
            valid_tracks = len([t for t in st.session_state.analyzed_tracks if t.get('features') is not None])
            st.metric(_("Avec features"), valid_tracks)
    
    elif len(st.session_state.pca_df) >= 2:
        st.markdown(_("Sélectionnez deux tracks pour créer une playlist progressive entre elles"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            track1_idx = st.selectbox(
                _("Track de départ:"),
                range(len(st.session_state.pca_df)),
                format_func=lambda x: f"{st.session_state.pca_df.iloc[x]['name']} ({st.session_state.pca_df.iloc[x]['genre']})"
            )
        
        with col2:
            track2_idx = st.selectbox(
                _("Track d'arrivée:"),
                range(len(st.session_state.pca_df)),
                index=min(1, len(st.session_state.pca_df)-1),
                format_func=lambda x: f"{st.session_state.pca_df.iloc[x]['name']} ({st.session_state.pca_df.iloc[x]['genre']})"
            )
        
        num_tracks = st.slider(_("Nombre de tracks dans la playlist:"), 5, 20, 10)
        
        if st.button(_("🎯 Générer la playlist"), type="primary"):
            if track1_idx != track2_idx:
                with st.spinner(_("Génération...")):
                    playlist, line_points, p1, p2 = generate_playlist_line(
                        st.session_state.pca_df, track1_idx, track2_idx, num_tracks
                    )
                    
                    st.session_state.generated_playlist = playlist
                    st.session_state.line_points = line_points
                    st.session_state.p1 = p1
                    st.session_state.p2 = p2
                    st.success(_(f"✅ Playlist de {len(playlist)} tracks générée!"))
            else:
                st.warning(_("⚠️ Sélectionnez deux tracks différentes"))
        
        # Afficher la playlist générée
        if 'generated_playlist' in st.session_state and st.session_state.generated_playlist:
            st.subheader(_("📋 Playlist Générée"))
            
            # Visualisation
            fig = go.Figure()
            
            # Points PCA
            fig.add_trace(go.Scatter(
                x=st.session_state.pca_df['PC1'],
                y=st.session_state.pca_df['PC2'],
                mode='markers',
                marker=dict(size=8, color='lightgray'),
                name='Autres tracks',
                hovertext=st.session_state.pca_df['name']
            ))
            
            # Ligne de la playlist
            if 'line_points' in st.session_state:
                fig.add_trace(go.Scatter(
                    x=st.session_state.line_points[:, 0],
                    y=st.session_state.line_points[:, 1],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trajectoire'
                ))
            
            # Points de la playlist
            playlist_df = pd.DataFrame(st.session_state.generated_playlist)
            fig.add_trace(go.Scatter(
                x=playlist_df['PC1'],
                y=playlist_df['PC2'],
                mode='markers+text',
                marker=dict(size=12, color='red'),
                text=playlist_df['position'],
                textposition='top center',
                name='Playlist',
                hovertext=playlist_df['name']
            ))
            
            fig.update_layout(
                title=_('Visualisation de la Playlist'),
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Liste des tracks
            st.subheader(_("🎵 Ordre de lecture"))
            
            spotify_tracks_count = 0
            deezer_tracks_count = 0
            
            for track_info in st.session_state.generated_playlist:
                col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**#{track_info['position']}**")
                
                with col2:
                    st.markdown(f"**{track_info['name']}**")
                    st.caption(f"{track_info['genre']}")
                
                with col3:
                    deezer_tracks_count += 1
                
                with col4:
                    st.markdown(f"📏 {track_info['distance']:.2f}")
                
                with col5:
                    if track_info['position'] == 1:
                        st.markdown(_("🚀 **Départ**"))
                    elif track_info['position'] == len(st.session_state.generated_playlist):
                        st.markdown(_("🎯 **Arrivée**"))
            
            # Export vers Spotify
            st.markdown("---")
            st.subheader(_("📤 Exporter vers Spotify"))
            
            st.info(_(f"ℹ️ {spotify_tracks_count} tracks Spotify peuvent être exportées, {deezer_tracks_count} tracks Deezer ne peuvent pas être exportées"))
            
            if spotify_tracks_count > 0:
                col1, col2 = st.columns(2)
                with col1:
                    playlist_name = st.text_input(_("Nom de la playlist:"), value="Ma Playlist IA")
                with col2:
                    playlist_desc = st.text_input(_("Description:"), value="Générée par IA")
                
                if st.button(_("🎵 Créer la playlist sur Spotify"), type="primary"):
                    if spotify_client:
                        # Récupérer les URIs des tracks Spotify uniquement
                        playlist_tracks_data = []
                        for track_info in st.session_state.generated_playlist:
                            track_idx = track_info['index']
                            track_uri = st.session_state.pca_df.iloc[track_idx]['uri']
                            if track_uri:  # Uniquement les tracks Spotify
                                playlist_tracks_data.append({
                                    'uri': track_uri,
                                    'name': track_info['name'],
                                    'spotify_id': st.session_state.pca_df.iloc[track_idx]['spotify_id']
                                })
                        
                        if playlist_tracks_data:
                            result = export_playlist_to_spotify(
                                spotify_client, 
                                playlist_tracks_data, 
                                playlist_name, 
                                playlist_desc
                            )
                        else:
                            st.warning(_("Aucune track Spotify dans la playlist à exporter"))
                    else:
                        st.error(_("❌ Client Spotify non disponible"))
            else:
                st.warning(_("Aucune track Spotify dans la playlist sélectionnée pour l'export"))

# Footer
st.markdown(_(
    """
    ---
    Développé par [Gabriel Marie-Brisson](https://gabriel.mariebrisson.fr)
    """
))