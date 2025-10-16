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

load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Music Playlist Generator",
    page_icon="🎵",
    layout="wide"
)

# --- Définition des classes du modèle CNN ---
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
        output = F.sigmoid(output) * x
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
        output = F.sigmoid(output) * x 
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

# --- Fonctions utilitaires ---
def convertSongToMatrice(audio_path, size=599):
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

def download_youtube_audio(url, output_path):
    """Télécharge l'audio d'une vidéo YouTube"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return output_path + '.wav', info.get('title', 'Unknown')
    except Exception as e:
        st.error(f"Erreur téléchargement YouTube: {str(e)}")
        return None, None

def download_spotify_preview(preview_url, output_path):
    """Télécharge l'aperçu audio d'un titre Spotify"""
    try:
        import requests
        response = requests.get(preview_url, timeout=10)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
        return None
    except Exception as e:
        st.error(f"Erreur téléchargement Spotify: {str(e)}")
        return None

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
        
        # Option 1: Charger seulement les poids (recommandé)
        if weights_only:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(state_dict)
        else:
            # Option 2: Charger le modèle complet
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle: {str(e)}")
        st.info("Vérifiez que le fichier .pth existe et contient le state_dict du modèle")
        return None

@st.cache_resource
def load_model_direct(model_path):
    """Charge directement le modèle complet si sauvegardé avec torch.save(model, ...)"""
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle direct: {str(e)}")
        return None

def extract_features(model, spectrogram_tensor):
    """Extrait les features avant la couche de classification"""
    with torch.no_grad():
        # Forward jusqu'à la dernière couche avant classification
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
        features = F.relu(x)  # Features de 2048 dimensions
        
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
        
        # Extraire les features
        features = extract_features(model, spectrogram_tensor)
        
        # Prédiction du genre
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

# --- Interface principale ---
st.title("🎵 Music Playlist Generator")
st.markdown("Créez des playlists personnalisées avec l'IA - Analyse de genres musicaux par CNN")

# Initialiser le session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'analyzed_tracks' not in st.session_state:
    st.session_state.analyzed_tracks = []

# Barre latérale
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Charger le modèle
    st.subheader("🧠 Modèle CNN")
    model_path = "templates/assets/music/best_model_original_loss.pth"
    
    load_method = st.radio(
        "Méthode de chargement:",
        ["State Dict (recommandé)", "Modèle complet"],
        help="State Dict: charge seulement les poids. Modèle complet: charge tout l'objet"
    )
    
    if st.button("Charger le modèle"):
        if os.path.exists(model_path):
            with st.spinner("Chargement..."):
                if load_method == "State Dict (recommandé)":
                    st.session_state.model = load_model(model_path, weights_only=True)
                else:
                    st.session_state.model = load_model_direct(model_path)
                
                if st.session_state.model:
                    st.success("✅ Modèle chargé!")
                    # Afficher les infos du modèle
                    total_params = sum(p.numel() for p in st.session_state.model.parameters())
                    st.info(f"📊 Paramètres: {total_params:,}")
        else:
            st.error(f"❌ Fichier introuvable: {model_path}")
            st.info("💡 Vérifiez le chemin du fichier .pth")
    
    # Spotify
    st.subheader("🔐 Spotify")
    with st.expander("ℹ️ Configuration"):
        st.markdown("""
        1. [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
        2. Créez une application
        3. Ajoutez `http://localhost:8501` comme Redirect URI
        """)
    
    CLIENT_ID_SPOTIFY = os.getenv('CLIENT_ID_SPOTIFY')
    CLIENT_SECRET_SPOTIFY = os.getenv('CLIENT_SECRET_SPOTIFY')
    REDIRECT_URI_SPOTIFY = os.getenv('REDIRECT_URI_SPOTIFY', 'http://localhost:8501')
    
    spotify_client = None
    if CLIENT_ID_SPOTIFY and CLIENT_SECRET_SPOTIFY:
        try:
            scope = "user-top-read playlist-modify-public playlist-modify-private user-read-recently-played"
            sp_oauth = SpotifyOAuth(
                client_id=CLIENT_ID_SPOTIFY,
                client_secret=CLIENT_SECRET_SPOTIFY,
                redirect_uri=REDIRECT_URI_SPOTIFY,
                scope=scope,
                cache_path=".spotify_cache"
            )
            
            token_info = sp_oauth.get_cached_token()
            
            if not token_info:
                auth_url = sp_oauth.get_authorize_url()
                st.markdown(f"[🔗 Connexion Spotify]({auth_url})")
                redirect_response = st.text_input("URL de redirection:")
                
                if redirect_response:
                    code = sp_oauth.parse_response_code(redirect_response)
                    token_info = sp_oauth.get_access_token(code)
                    st.success("✅ Connecté!")
                    st.rerun()
            else:
                spotify_client = spotipy.Spotify(auth_manager=sp_oauth)
                st.success("✅ Spotify connecté")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

# Tabs principaux
tab1, tab2, tab3, tab4 = st.tabs(["🎬 YouTube", "🎧 Spotify", "📊 Analyse", "🎨 Playlist"])

# Tab 1: YouTube
with tab1:
    st.header("Ajouter depuis YouTube")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        youtube_url = st.text_input("URL YouTube:", placeholder="https://www.youtube.com/watch?v=...")
    
    if youtube_url:
        try:
            st.video(youtube_url)
            
            if st.button("➕ Ajouter et analyser", type="primary"):
                if st.session_state.model is None:
                    st.warning("⚠️ Veuillez charger le modèle d'abord")
                else:
                    with st.spinner("Téléchargement et analyse..."):
                        temp_dir = tempfile.mkdtemp()
                        output_path = os.path.join(temp_dir, "youtube_audio")
                        
                        audio_file, title = download_youtube_audio(youtube_url, output_path)
                        
                        if audio_file:
                            genre, confidence, features, probs = analyze_audio_genre(
                                audio_file, st.session_state.model
                            )
                            
                            if genre:
                                track_data = {
                                    'name': title or youtube_url,
                                    'source': 'youtube',
                                    'url': youtube_url,
                                    'genre': genre,
                                    'confidence': confidence,
                                    'features': features,
                                    'probabilities': probs
                                }
                                st.session_state.analyzed_tracks.append(track_data)
                                st.success(f"✅ Ajouté: {genre} ({confidence:.1%})")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
    
    # Afficher les tracks YouTube
    if st.session_state.analyzed_tracks:
        yt_tracks = [t for t in st.session_state.analyzed_tracks if t['source'] == 'youtube']
        if yt_tracks:
            st.subheader(f"Tracks YouTube ({len(yt_tracks)})")
            for i, track in enumerate(yt_tracks):
                with st.expander(f"{track['name'][:50]}..."):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Genre", track['genre'])
                    col2.metric("Confiance", f"{track['confidence']:.1%}")
                    col3.button("🗑️ Supprimer", key=f"del_yt_{i}")

# Tab 2: Spotify
with tab2:
    st.header("Importer depuis Spotify")
    
    if spotify_client:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Top tracks"):
                try:
                    top_tracks = spotify_client.current_user_top_tracks(limit=20, time_range='medium_term')
                    st.session_state.spotify_top_tracks = top_tracks['items']
                    st.success(f"✅ {len(top_tracks['items'])} tracks")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
        
        with col2:
            if st.button("🕒 Récents"):
                try:
                    recent = spotify_client.current_user_recently_played(limit=20)
                    st.session_state.spotify_recent_tracks = [item['track'] for item in recent['items']]
                    st.success(f"✅ {len(st.session_state.spotify_recent_tracks)} tracks")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
        
        # Afficher et analyser les tracks
        if 'spotify_top_tracks' in st.session_state:
            st.subheader("Top tracks")
            
            for i, track in enumerate(st.session_state.spotify_top_tracks[:10]):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{track['name']}**")
                    artists = ", ".join([a['name'] for a in track['artists']])
                    st.caption(artists)
                
                with col2:
                    if track.get('preview_url'):
                        st.audio(track['preview_url'])
                
                with col3:
                    if st.button("🔍", key=f"analyze_top_{i}"):
                        if st.session_state.model and track.get('preview_url'):
                            with st.spinner("Analyse..."):
                                temp_dir = tempfile.mkdtemp()
                                audio_path = os.path.join(temp_dir, f"spotify_{i}.mp3")
                                
                                if download_spotify_preview(track['preview_url'], audio_path):
                                    genre, confidence, features, probs = analyze_audio_genre(
                                        audio_path, st.session_state.model
                                    )
                                    
                                    if genre:
                                        track_data = {
                                            'name': track['name'],
                                            'artists': artists,
                                            'source': 'spotify',
                                            'spotify_id': track['id'],
                                            'uri': track['uri'],
                                            'genre': genre,
                                            'confidence': confidence,
                                            'features': features,
                                            'probabilities': probs
                                        }
                                        st.session_state.analyzed_tracks.append(track_data)
                                        st.success(f"{genre} ({confidence:.1%})")
                        else:
                            st.warning("Modèle non chargé ou pas d'aperçu")
    else:
        st.warning("⚠️ Connectez-vous à Spotify")

# Tab 3: Analyse
with tab3:
    st.header("📊 Analyse des genres")
    
    if st.session_state.analyzed_tracks:
        df = pd.DataFrame(st.session_state.analyzed_tracks)
        
        # Statistiques
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total tracks", len(df))
        col2.metric("Genres uniques", df['genre'].nunique())
        col3.metric("Confiance moy.", f"{df['confidence'].mean():.1%}")
        col4.metric("Source", f"YT: {len(df[df['source']=='youtube'])} / SP: {len(df[df['source']=='spotify'])}")
        
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
        if len(st.session_state.analyzed_tracks) >= 2:
            st.subheader("Visualisation PCA")
            
            features_list = [t['features'] for t in st.session_state.analyzed_tracks if t.get('features') is not None]
            
            if len(features_list) >= 2:
                pca_result, pca_model, scaler = perform_pca(features_list)
                
                if pca_result is not None:
                    pca_df = pd.DataFrame({
                        'PC1': pca_result[:, 0],
                        'PC2': pca_result[:, 1],
                        'name': [t['name'][:30] for t in st.session_state.analyzed_tracks if t.get('features') is not None],
                        'genre': [t['genre'] for t in st.session_state.analyzed_tracks if t.get('features') is not None]
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
                st.subheader("💾 Sauvegarder sur Spotify")
                
                col1, col2 = st.columns(2)
                with col1:
                    playlist_name = st.text_input("Nom de la playlist:", value="Ma Playlist IA")
                with col2:
                    playlist_public = st.checkbox("Playlist publique", value=False)
                
                playlist_description = st.text_area(
                    "Description:",
                    value=f"Playlist générée par IA - De {st.session_state.pca_df.iloc[track1_idx]['name']} à {st.session_state.pca_df.iloc[track2_idx]['name']}"
                )
                
                if st.button("📤 Créer sur Spotify", type="primary"):
                    try:
                        # Récupérer les URIs Spotify
                        spotify_tracks = [
                            t for t in st.session_state.analyzed_tracks 
                            if t.get('source') == 'spotify' and t.get('uri')
                        ]
                        
                        if spotify_tracks:
                            # Créer la playlist
                            user_id = spotify_client.current_user()['id']
                            playlist = spotify_client.user_playlist_create(
                                user=user_id,
                                name=playlist_name,
                                public=playlist_public,
                                description=playlist_description
                            )
                            
                            # Ajouter les tracks
                            track_uris = [t['uri'] for t in spotify_tracks]
                            
                            # Spotify limite à 100 tracks par requête
                            for i in range(0, len(track_uris), 100):
                                spotify_client.playlist_add_items(
                                    playlist['id'], 
                                    track_uris[i:i+100]
                                )
                            
                            st.success(f"✅ Playlist '{playlist_name}' créée avec {len(track_uris)} titres!")
                            st.markdown(f"[🔗 Ouvrir dans Spotify]({playlist['external_urls']['spotify']})")
                        else:
                            st.warning("⚠️ Aucune track Spotify dans la sélection")
                    
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
    
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

# Bouton de reset
with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Réinitialiser tout", type="secondary"):
        for key in list(st.session_state.keys()):
            if key != 'model':
                del st.session_state[key]
        st.success("✅ Réinitialisé!")
        st.rerun()