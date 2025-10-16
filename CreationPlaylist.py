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
import subprocess  # ⬅️ AJOUT IMPORT MANQUANT
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Music Playlist Generator",
    page_icon="🎵",
    layout="wide"
)

# --- DÉBUT DES CLASSES CNN (COMPLÈTES) ---
class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max_pool = F.adaptive_max_pool2d(x, output_size=1)
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max_pool.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg_pool.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x
        return output

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max_pool = torch.max(x,1)[0].unsqueeze(1)
        avg_pool = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max_pool,avg_pool), dim=1)
        output = self.conv(concat)
        output = torch.sigmoid(output) * x 
        return output     

class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x

class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super(SimpleCNN, self).__init__()
        
        self.norm1 = nn.BatchNorm2d(1) 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(128,4), padding=(0,1))
        self.relu = nn.ReLU()
        self.cbam1 = CBAM(256, r=16)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,4))
        
        self.norm2 = nn.BatchNorm2d(1) 
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(256,4)) 
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2))
        
        self.norm3 = nn.BatchNorm2d(1)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(256,4))
        self.cbam2 = CBAM(512, r=16)
        
        self.norm4 = nn.BatchNorm2d(1) 
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(512,4))
        
        self.normfc2 = nn.BatchNorm1d(1536)
        self.fc2 = nn.Linear(1536, 2048)
        self.normfc3 = nn.BatchNorm1d(2048)
        self.drop1 = nn.Dropout2d(p=dropout)  
        self.fc3 = nn.Linear(2048, 1024)
        self.normfc4 = nn.BatchNorm1d(1024) 
        self.drop2 = nn.Dropout2d(p=dropout) 
        self.fc4 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.cbam1(x)
        x = self.pool1(x)
        x = torch.permute(x,(0,2,1,3))
        
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.cbam1(x)
        x = self.pool2(x)
        x = torch.permute(x,(0,2,1,3))
        
        x = self.norm3(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.cbam2(x)
        x = self.pool2(x)
        x = torch.permute(x,(0,2,1,3))
        
        x = self.norm4(x)
        x = self.conv4(x)
        x = self.cbam2(x)
        x = self.relu(x)
        x = torch.permute(x,(0,2,1,3))
        
        mean_values = torch.mean(x, dim=3, keepdim=True)
        max_values, _ = torch.max(x, dim=3, keepdim=True)
        l2_norm = torch.linalg.norm(x, dim=3, ord=2, keepdim=True)
        
        x = torch.cat([max_values, mean_values, l2_norm], dim=1)
        x = x.view(-1, 1536)
        
        x = self.normfc2(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.normfc3(x)
        x = self.fc3(x)
        x = self.drop1(x)
        x = F.relu(x)
        
        x = self.normfc4(x)
        x = self.drop2(x)
        x = self.fc4(x)
        
        return x


def convertSongToMatrice(audio_path, size=599):
    """Convertit un fichier audio en spectrogramme normalisé - Version robuste"""
    try:
        st.info("🎵 Conversion de l'audio en spectrogramme...")
        
        # Essayer de charger avec différentes configurations
        try:
            y, sr = librosa.load(audio_path, duration=30, mono=True, sr=22050)
        except Exception as e:
            st.warning(f"Premier essai échoué: {e}. Nouvel essai...")
            try:
                y, sr = librosa.load(audio_path, duration=30, mono=True)
            except Exception as e2:
                st.error(f"❌ Impossible de charger le fichier audio: {e2}")
                return None
        
        # Vérifier la durée audio
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 5:  # Moins de 5 secondes
            st.warning(f"⚠️ Extrait trop court: {duration:.1f}s")
            return None
        
        st.info(f"✅ Audio chargé: {duration:.1f}s, {sr}Hz")
        
        # Créer le spectrogramme avec paramètres adaptatifs
        n_fft = min(2048, int(sr * 0.025))  # Taille FFT adaptative
        hop_length = int(n_fft / 4)
        
        try:
            D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            spectrogram = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=128, fmin=50, fmax=8000)
            
            # Normaliser la taille
            S = librosa.util.fix_length(spectrogram, size=size, axis=1)
            S_db = librosa.power_to_db(S, ref=np.max)
            S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
            
            st.info("✅ Spectrogramme généré avec succès")
            return S_db_norm
            
        except Exception as e:
            st.error(f"❌ Erreur création spectrogramme: {e}")
            return None
        
    except Exception as e:
        st.error(f"❌ Erreur conversion audio: {str(e)}")
        return None
    
def get_tracks_with_previews(tracks):
    """Filtre les tracks qui ont des previews disponibles"""
    tracks_with_previews = [track for track in tracks if track.get('preview_url')]
    st.info(f"🎵 {len(tracks_with_previews)}/{len(tracks)} titres avec extraits audio")
    return tracks_with_previews

def analyze_spotify_track(track, track_index, source_type="saved"):
    """Analyse une track Spotify avec gestion d'erreurs complète"""
    if not st.session_state.model:
        st.warning("⚠️ Veuillez charger le modèle d'abord")
        return None
    
    if not track.get('preview_url'):
        st.warning(f"⚠️ Aucun extrait disponible pour: {track['name']}")
        return None
    
    try:
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
            genre, confidence, features, probs = analyze_audio_genre(
                downloaded_path, st.session_state.model
            )
            
            if genre:
                # Formater les artistes
                artists = track.get('artists', 'Artiste inconnu')
                if isinstance(artists, list):
                    artists = ', '.join([a['name'] if isinstance(a, dict) else str(a) for a in artists])
                
                track_data = {
                    'name': track['name'],
                    'artists': artists,
                    'source': 'spotify',
                    'spotify_id': track.get('id'),
                    'uri': track.get('uri'),
                    'preview_url': track.get('preview_url'),
                    'genre': genre,
                    'confidence': confidence,
                    'features': features,
                    'probabilities': probs
                }
                
                st.session_state.analyzed_tracks.append(track_data)
                st.success(f"✅ {track['name'][:40]} - {genre} ({confidence:.1%})")
                return track_data
            else:
                st.error(f"❌ Échec de l'analyse pour: {track['name'][:40]}")
                return None
                
    except Exception as e:
        st.error(f"❌ Erreur analyse '{track['name'][:30]}': {str(e)}")
        return None


def convert_audio_format(input_path, output_path):
    """Convertit un fichier audio en format WAV compatible avec librosa"""
    try:
        # Vérifier si ffmpeg est disponible
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            ffmpeg_available = True
        except:
            ffmpeg_available = False
            st.warning("⚠️ FFmpeg n'est pas installé. L'installation est recommandée pour la conversion audio.")
        
        if ffmpeg_available:
            # Conversion en WAV avec FFmpeg
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
        
        # Fallback: utiliser le fichier original
        return input_path
        
    except Exception as e:
        st.warning(f"Erreur conversion: {e}")
        return input_path

def download_spotify_preview(preview_url, output_path):
    """Télécharge et convertit l'aperçu audio d'un titre Spotify"""
    try:
        if not preview_url:
            return None, "Aucun extrait disponible pour cette piste"
            
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
            
            if converted_path != output_path and os.path.exists(converted_path):
                return converted_path, "Conversion réussie"
            else:
                # Utiliser le fichier MP3 original
                return output_path, "Fichier MP3 original"
        else:
            return None, f"Erreur HTTP {response.status_code}"
            
    except Exception as e:
        return None, f"Erreur téléchargement: {str(e)}"


# --- FONCTIONS SPOTIFY AMÉLIORÉES ---

def get_tracks_with_previews(tracks):
    """Filtre les tracks qui ont des previews disponibles"""
    return [track for track in tracks if track.get('preview_url')]

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

@st.cache_resource
def load_model(model_path, weights_only=True):
    """Charge le modèle CNN pré-entraîné"""
    try:
        model = SimpleCNN()
        
        if weights_only:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
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
    spectrogram = convertSongToMatrice(audio_path)
    
    if spectrogram is None:
        return None, None, None, None
    
    if model is None:
        genre_id = np.random.randint(0, 10)
        confidence = np.random.uniform(0.6, 0.95)
        return label_mapping[genre_id], confidence, None, None
    
    try:
        spectrogram_tensor = torch.tensor(spectrogram).unsqueeze(0).unsqueeze(0).float()
        
        features = extract_features(model, spectrogram_tensor)
        
        with torch.no_grad():
            output = model(spectrogram_tensor)
            probabilities = F.softmax(output, dim=1)
            genre_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][genre_id].item()
            all_probs = probabilities[0].cpu().numpy()
        
        return label_mapping[genre_id], confidence, features[0], all_probs
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
    used_tracks = set([track1_idx, track2_idx])
    
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
                'PC2': closest_row['PC2']
            })
            
            used_tracks.add(closest_idx)
    
    return playlist_tracks, line_points, p1, p2

# --- FONCTIONS SPOTIFY CORRIGÉES ---
# Configuration Spotify améliorée
def init_spotify_auth():
    """Initialise l'authentification Spotify avec gestion d'erreurs améliorée"""
    try:
        CLIENT_ID = os.getenv('CLIENT_ID_SPOTIFY')
        CLIENT_SECRET = os.getenv('CLIENT_SECRET_SPOTIFY')
        REDIRECT_URI = os.getenv('REDIRECT_URI_SPOTIFY', 'http://localhost:8501')
        
        # Validation des credentials
        if not CLIENT_ID or CLIENT_ID == 'votre_client_id_ici':
            st.error("❌ CLIENT_ID_SPOTIFY manquant ou non configuré")
            st.info("💡 Ajoutez votre vrai CLIENT_ID_SPOTIFY dans le fichier .env")
            return None
            
        if not CLIENT_SECRET or CLIENT_SECRET == 'votre_client_secret_ici':
            st.error("❌ CLIENT_SECRET_SPOTIFY manquant ou non configuré")
            st.info("💡 Ajoutez votre vrai CLIENT_SECRET_SPOTIFY dans le fichier .env")
            return None
        
        # Configuration OAuth
        sp_oauth = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="user-library-read user-top-read playlist-modify-public playlist-modify-private user-read-recently-played",
            cache_path=".spotify_cache",
            show_dialog=True  # Force la reconnexion si besoin
        )
        
        return sp_oauth
        
    except Exception as e:
        st.error(f"❌ Erreur configuration Spotify: {str(e)}")
        return None

def get_spotify_client():
    """Obtient un client Spotify authentifié"""
    try:
        CLIENT_ID = os.getenv('CLIENT_ID_SPOTIFY')
        CLIENT_SECRET = os.getenv('CLIENT_SECRET_SPOTIFY')
        REDIRECT_URI = os.getenv('REDIRECT_URI_SPOTIFY', 'http://localhost:8501')
        
        if not CLIENT_ID or not CLIENT_SECRET:
            return None

        scope = "user-library-read user-top-read playlist-modify-public playlist-modify-private user-read-recently-played user-read-email"
        
        sp_oauth = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope=scope,
            cache_path=".spotify_cache",
            show_dialog=True
        )
        
        # Vérifier le token en cache
        token_info = sp_oauth.get_cached_token()
        
        # Gérer la redirection après authentification
        query_params = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
        
        if "code" in query_params:
            code = query_params["code"]
            if isinstance(code, list):
                code = code[0]
            
            try:
                token_info = sp_oauth.get_access_token(code, as_dict=False)
                # Nettoyer l'URL
                if hasattr(st, "query_params"):
                    st.query_params.clear()
                else:
                    st.experimental_set_query_params()
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de l'échange du code: {e}")
                return None
        
        # Si token expiré, rafraîchir
        if token_info and sp_oauth.is_token_expired(token_info):
            try:
                token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            except Exception as e:
                st.error(f"Erreur rafraîchissement token: {e}")
                return None
        
        if token_info:
            return spotipy.Spotify(auth=token_info['access_token'])
        
        return None
        
    except Exception as e:
        st.error(f"Erreur d'authentification Spotify: {e}")
        return None

def export_tracks_to_csv(tracks_data, filename="ma_musique_export.csv"):
    """Exporte les tracks analysés en CSV"""
    if not tracks_data:
        st.warning("Aucune donnée à exporter")
        return None
    
    try:
        export_data = []
        for track in tracks_data:
            track_info = {
                'nom': track.get('name', 'Inconnu'),
                'artistes': track.get('artists', 'Inconnu'),
                'genre_prediction': track.get('genre', 'Inconnu'),
                'confiance': f"{track.get('confidence', 0):.2%}",
                'source': track.get('source', 'Inconnu'),
                'url': track.get('url', ''),
                'spotify_id': track.get('spotify_id', ''),
                'date_analyse': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            export_data.append(track_info)
        
        df = pd.DataFrame(export_data)
        return df.to_csv(index=False, encoding='utf-8-sig')
    except Exception as e:
        st.error(f"Erreur lors de l'export: {e}")
        return None

def export_playlist_to_spotify(spotify_client, playlist_tracks, playlist_name, playlist_description=""):
    """Exporte une playlist vers Spotify"""
    try:
        if not spotify_client:
            st.error("❌ Client Spotify non disponible")
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
            
            st.success(f"✅ Playlist '{playlist_name}' créée avec {len(track_uris)} titres!")
            st.markdown(f"[🎵 Ouvrir dans Spotify]({playlist['external_urls']['spotify']})")
            return playlist
        else:
            st.warning("⚠️ Aucun URI Spotify trouvé pour les tracks sélectionnées")
            return None
            
    except Exception as e:
        st.error(f"❌ Erreur lors de la création de la playlist: {e}")
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

# --- INITIALISATION SESSION STATE ---
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

# --- INTERFACE STREAMLIT PRINCIPALE ---
st.title("🎵 Music Playlist Generator")
st.markdown("Créez des playlists personnalisées avec l'IA - Analyse de genres musicaux par CNN")

# Barre latérale
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Charger le modèle
    st.subheader("🧠 Modèle CNN")
    model_path = "templates/assets/music/best_model_original_loss.pth"
    
    if os.path.exists(model_path):
        with st.spinner("Chargement..."):
            st.session_state.model = load_model(model_path, weights_only=True)
    else:
        st.error(f"❌ Fichier introuvable: {model_path}")

    # Connexion Spotify - VERSION CORRIGÉE
    st.subheader("🔗 Connexion Spotify")
    
    CLIENT_ID_SPOTIFY = os.getenv('CLIENT_ID_SPOTIFY')
    CLIENT_SECRET_SPOTIFY = os.getenv('CLIENT_SECRET_SPOTIFY')
    
    if not CLIENT_ID_SPOTIFY or not CLIENT_SECRET_SPOTIFY:
        st.error("🔒 CLIENT_ID_SPOTIFY ou CLIENT_SECRET_SPOTIFY manquant")
        st.info("""
        **Pour configurer Spotify:**
        1. Allez sur [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
        2. Créez une application
        3. Ajoutez dans votre fichier .env:
        ```
        CLIENT_ID_SPOTIFY=votre_client_id_ici
        CLIENT_SECRET_SPOTIFY=votre_client_secret_ici
        REDIRECT_URI_SPOTIFY=http://localhost:8501
        ```
        """)
        spotify_client = None
    else:
        # Configuration OAuth
        scope = "user-library-read user-top-read playlist-modify-public playlist-modify-private user-read-recently-played user-read-email"
        
        sp_oauth = SpotifyOAuth(
            client_id=CLIENT_ID_SPOTIFY,
            client_secret=CLIENT_SECRET_SPOTIFY,
            redirect_uri=os.getenv('REDIRECT_URI_SPOTIFY', 'http://localhost:8501'),
            scope=scope,
            cache_path=".spotify_cache",
            show_dialog=True
        )
        
        # Obtenir le client Spotify
        spotify_client = get_spotify_client()
        
        if spotify_client:
            try:
                user_info = spotify_client.current_user()
                st.success(f"✅ Connecté en tant que **{user_info['display_name']}**")
                
                # Options d'export
                st.subheader("📤 Export")
                
                if st.session_state.analyzed_tracks:
                    if st.button("💾 Exporter en CSV"):
                        csv_data = export_tracks_to_csv(st.session_state.analyzed_tracks)
                        if csv_data:
                            st.download_button(
                                label="📥 Télécharger CSV",
                                data=csv_data,
                                file_name="musique_export.csv",
                                mime="text/csv"
                            )
                
            except Exception as e:
                st.error(f"Erreur de connexion: {e}")
        else:
            # Afficher le bouton de connexion
            auth_url = sp_oauth.get_authorize_url()
            st.markdown(f"""
            **Étapes de connexion:**
            1. Cliquez sur le bouton ci-dessous
            2. Autorisez l'application sur Spotify  
            3. Vous serez redirigé vers cette page
            
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
                🎵 Se connecter avec Spotify
                </div>
            </a>
            """, unsafe_allow_html=True)
            
            st.info("👆 Cliquez sur le bouton pour vous connecter à Spotify")

    # Bouton de reset
    st.markdown("---")
    if st.button("🔄 Réinitialiser tout", type="secondary"):
        for key in list(st.session_state.keys()):
            if key != 'model':
                del st.session_state[key]
        st.success("✅ Réinitialisé!")
        st.rerun()

# Tabs principaux
tab2, tab3, tab4 = st.tabs(["🎧 Spotify", "📊 Analyse", "🎨 Playlist"])

# Tab 2: Spotify (CORRIGÉ)
with tab2:
    st.header("🎧 Importer depuis Spotify")
    
    if spotify_client:
        # Nouvelle section : Mes titres sauvegardés
        st.subheader("💾 Mes titres sauvegardés")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            limit_tracks = st.slider("Nombre de titres à importer:", 10, 100, 20)
        
        with col2:
            if st.button("🔄 Charger mes titres", key="load_saved"):
                with st.spinner("Récupération de vos titres..."):
                    saved_tracks = get_user_saved_tracks(spotify_client, limit=limit_tracks)
                    tracks_with_previews = get_tracks_with_previews(saved_tracks)
                    st.session_state.spotify_saved_tracks = tracks_with_previews
                    
                    if tracks_with_previews:
                        st.success(f"✅ {len(tracks_with_previews)} titres avec extraits chargés (sur {len(saved_tracks)} total)")
                    else:
                        st.warning("⚠️ Aucun titre avec extrait audio disponible")
        
        with col3:
            if st.button("🔍 Tout analyser", key="analyze_all_saved"):
                if 'spotify_saved_tracks' in st.session_state and st.session_state.spotify_saved_tracks:
                    for i, track in enumerate(st.session_state.spotify_saved_tracks[:5]):  # Limiter à 5 pour éviter les timeouts
                        analyze_spotify_track(track, i, "saved")
                else:
                    st.warning("⚠️ Chargez d'abord vos titres")
        
        # Afficher et analyser les titres sauvegardés
        if 'spotify_saved_tracks' in st.session_state and st.session_state.spotify_saved_tracks:
            st.subheader(f"Vos titres avec extraits ({len(st.session_state.spotify_saved_tracks)})")
            
            # Statistiques des previews
            total_tracks = len(st.session_state.spotify_saved_tracks)
            st.info(f"🎵 {total_tracks} titres avec extraits audio disponibles")
            
            for i, track in enumerate(st.session_state.spotify_saved_tracks[:8]):  # Limiter l'affichage
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{track['name']}**")
                    artists = track.get('artists', 'Artiste inconnu')
                    if isinstance(artists, list):
                        artists = ', '.join([a['name'] for a in artists])
                    st.caption(f"🎤 {artists}")
                    
                    # Indicateur de preview disponible
                    if track.get('preview_url'):
                        st.caption("🔊 Extrait 30s disponible")
                
                with col2:
                    if track.get('preview_url'):
                        st.audio(track['preview_url'], format="audio/mp3")
                    else:
                        st.caption("❌ Pas d'extrait")
                
                with col3:
                    if st.button("🔍 Analyser", key=f"analyze_saved_{i}"):
                        analyze_spotify_track(track, i, "saved")
                
                with col4:
                    if st.button("➕ Ajouter", key=f"add_saved_{i}"):
                        track_data = {
                            'name': track['name'],
                            'artists': track.get('artists', 'Artiste inconnu'),
                            'source': 'spotify',
                            'spotify_id': track.get('id'),
                            'uri': track.get('uri'),
                            'preview_url': track.get('preview_url'),
                            'genre': 'Non analysé',
                            'confidence': 0.0
                        }
                        st.session_state.analyzed_tracks.append(track_data)
                        st.success(f"✅ {track['name'][:40]} ajouté!")
        
        # Sections Top Tracks et Récents - VERSION AMÉLIORÉE
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Top tracks avec extraits", key="load_top"):
                try:
                    top_tracks = spotify_client.current_user_top_tracks(limit=25, time_range='medium_term')
                    tracks_with_previews = get_tracks_with_previews(top_tracks['items'])
                    st.session_state.spotify_top_tracks = tracks_with_previews
                    
                    if tracks_with_previews:
                        st.success(f"✅ {len(tracks_with_previews)} top tracks avec extraits chargés")
                    else:
                        st.warning("⚠️ Aucun top track avec extrait audio disponible")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
        
        with col2:
            if st.button("🕒 Récents avec extraits", key="load_recent"):
                try:
                    recent = spotify_client.current_user_recently_played(limit=25)
                    recent_tracks = [item['track'] for item in recent['items']]
                    tracks_with_previews = get_tracks_with_previews(recent_tracks)
                    st.session_state.spotify_recent_tracks = tracks_with_previews
                    
                    if tracks_with_previews:
                        st.success(f"✅ {len(tracks_with_previews)} titres récents avec extraits chargés")
                    else:
                        st.warning("⚠️ Aucun titre récent avec extrait audio disponible")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
        
        # Afficher les Top Tracks
        if 'spotify_top_tracks' in st.session_state and st.session_state.spotify_top_tracks:
            st.subheader("🎯 Vos top tracks avec extraits")
            
            for i, track in enumerate(st.session_state.spotify_top_tracks[:6]):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{track['name']}**")
                    artists = ", ".join([a['name'] for a in track['artists']])
                    st.caption(f"🎤 {artists}")
                    if track.get('preview_url'):
                        st.caption("🔊 Extrait disponible")
                
                with col2:
                    if track.get('preview_url'):
                        st.audio(track['preview_url'])
                
                with col3:
                    if st.button("🔍", key=f"analyze_top_{i}"):
                        analyze_spotify_track(track, i, "top")
                
                with col4:
                    if st.button("➕", key=f"add_top_{i}"):
                        track_data = {
                            'name': track['name'],
                            'artists': ", ".join([a['name'] for a in track['artists']]),
                            'source': 'spotify', 
                            'spotify_id': track['id'],
                            'uri': track['uri'],
                            'genre': 'Non analysé',
                            'confidence': 0.0
                        }
                        st.session_state.analyzed_tracks.append(track_data)
                        st.success("✅ Ajouté!")
        
        # Afficher les Récents
        if 'spotify_recent_tracks' in st.session_state and st.session_state.spotify_recent_tracks:
            st.subheader("🕒 Vos titres récents avec extraits")
            
            for i, track in enumerate(st.session_state.spotify_recent_tracks[:6]):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{track['name']}**")
                    artists = ", ".join([a['name'] for a in track['artists']])
                    st.caption(f"🎤 {artists}")
                    if track.get('preview_url'):
                        st.caption("🔊 Extrait disponible")
                
                with col2:
                    if track.get('preview_url'):
                        st.audio(track['preview_url'])
                
                with col3:
                    if st.button("🔍", key=f"analyze_recent_{i}"):
                        analyze_spotify_track(track, i, "recent")
                
                with col4:
                    if st.button("➕", key=f"add_recent_{i}"):
                        track_data = {
                            'name': track['name'],
                            'artists': ", ".join([a['name'] for a in track['artists']]),
                            'source': 'spotify', 
                            'spotify_id': track['id'],
                            'uri': track['uri'],
                            'genre': 'Non analysé',
                            'confidence': 0.0
                        }
                        st.session_state.analyzed_tracks.append(track_data)
                        st.success("✅ Ajouté!")
                        
    else:
        st.warning("⚠️ Connectez-vous à Spotify pour importer vos musiques")

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
                
                # Export CSV
                if st.session_state.analyzed_tracks:
                    csv_data = export_tracks_to_csv(
                        st.session_state.analyzed_tracks, 
                        f"playlist_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    )
                    if csv_data:
                        st.download_button(
                            label="📥 Télécharger CSV complet",
                            data=csv_data,
                            file_name=f"playlist_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
    
    else:
        st.info("👆 Analysez au moins 2 tracks dans l'onglet Analyse pour générer une playlist")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🧠 **Modèle:** CNN avec CBAM")
with col2:
    st.markdown("📊 **Méthode:** PCA + Distance euclidienne")  
with col3:
    st.markdown("💻 **Tech:** PyTorch, Librosa, Plotly")