"""
Multi-City Weather Prediction System - Streamlit Version
Predicts temperature for Paris and Silicon Valley
Using Meteostat, trained models, and comparing with OpenWeatherMap forecasts
"""

import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
from meteostat import Point, Daily
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

from deep_translator import GoogleTranslator


# Page configuration
st.set_page_config(
    page_title= "Multi-City Weather Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #2980b9, #6dd5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .city-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# City Configuration
@dataclass
class City:
    name: str
    lat: float
    lon: float
    timezone: str
    emoji: str

CITIES = {
    'paris': City('Paris', 48.8566, 2.3522, 'Europe/Paris', '🗼'),
    'silicon_valley': City('Silicon Valley', 37.3875, -122.0575, 'America/Los_Angeles', '🌉')
}

# API Configuration
OPENWEATHER_API_KEY = "9831a5fb2b1386a9a8c3eeac4f8200fa"

# Model Configuration
WINDOW_SIZE = 30
FORECAST_HORIZON = 7

# Session state initialization
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}


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

def gettext(text):
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

# Functions
@st.cache_data(ttl=3600)
def collect_historical_data(city, years_back=10):
    """Collect historical weather data from Meteostat"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back*365)

    location = Point(city.lat, city.lon)
    data = Daily(location, start_date, end_date)
    df = data.fetch()

    if df.empty:
        return None

    df['city'] = city.name
    return df

def create_time_features(df):
    """Create time-based features"""
    df = df.copy()
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def preprocess_data(df):
    """Clean and standardize dataset"""
    df = df.copy()

    column_mapping = {
        'tavg': 'temp_avg', 'tmin': 'temp_min', 'tmax': 'temp_max',
        'prcp': 'precipitation', 'snow': 'snowfall', 'wdir': 'wind_direction',
        'wspd': 'wind_speed', 'wpgt': 'wind_gust', 'pres': 'pressure', 'tsun': 'sunshine'
    }
    df = df.rename(columns=column_mapping)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    df = create_time_features(df)

    if not {'temp_avg', 'temp_min', 'temp_max'}.issubset(df.columns):
        return None

    df = df[(df['temp_avg'] != 0) | (df['temp_min'] != 0) | (df['temp_max'] != 0)]
    return df

def load_model_info(city_key):
    """Load model and its feature information"""
    model_path = f'templates/assets/température/models/{city_key}_model.keras'
    info_path = f'templates/assets/température/models/{city_key}_info.pkl'

    if not os.path.exists(model_path):
        return None, None

    model = tf.keras.models.load_model(model_path)

    # Load feature info if available
    if os.path.exists(info_path):
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
        return model, info
    else:
        # If no info file, return model only
        return model, None

def prepare_scalers(df, expected_features=None):
    """Prépare les scalers, en garantissant les bonnes features."""
    temp_cols = [c for c in df.columns if c.startswith('temp_')]

    if expected_features is not None:
        numeric_cols = expected_features
        # Ajout des colonnes manquantes avec des zéros
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_cols].values.astype(np.float32)
    y = df[temp_cols].values.astype(np.float32)

    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler()

    X_scaler.fit(X)
    y_scaler.fit(y)

    return {
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'feature_cols': numeric_cols,
        'target_cols': temp_cols
    }


def predict_7day_forecast(model, recent_data, scalers):
    """Predict 7-day temperature forecast"""
    X_scaler = scalers['X_scaler']
    y_scaler = scalers['y_scaler']

    recent_scaled = X_scaler.transform(recent_data[-WINDOW_SIZE:])
    X_input = recent_scaled.reshape(1, WINDOW_SIZE, recent_scaled.shape[1])

    y_pred_scaled = model.predict(X_input, verbose=0)[0]
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    return y_pred

@st.cache_data(ttl=3600)
def get_openweather_forecast(city, days=8):
    """Get forecast from OpenWeatherMap"""
    url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        'lat': city.lat,
        'lon': city.lon,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'
    }

    try:
        response = requests.get(url, params=params, timeout=10 )
        data = response.json()

        daily_data = {}
        for item in data['list']:
            date = datetime.fromtimestamp(item['dt']).date()
            if date not in daily_data:
                daily_data[date] = {'temps': [], 'temp_min': [], 'temp_max': []}
            daily_data[date]['temps'].append(item['main']['temp'])
            daily_data[date]['temp_min'].append(item['main']['temp_min'])
            daily_data[date]['temp_max'].append(item['main']['temp_max'])

        forecast = []
        for date in sorted(daily_data.keys())[:days]:
            forecast.append({
                'date': date,
                'temp_avg': np.mean(daily_data[date]['temps']),
                'temp_min': np.min(daily_data[date]['temp_min']),
                'temp_max': np.max(daily_data[date]['temp_max'])
            })

        return forecast
    except Exception as e:
        st.error(f"Error fetching OpenWeatherMap data: {e}")
        return None
# Bouton de redirection
st.markdown(
    f"""
    <a href="https://gabriel.mariebrisson.fr" target="_blank" style="text-decoration:none;">
    <div style="
    display: inline-block;
    background: linear-gradient(135deg, #6A11CB 0%, #2575FC 100% );
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
    {gettext("Retour")}
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


# Main App
st.markdown(gettext('<h1 class="main-header"> Multi-City Weather Prediction System</h1>'), unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header(gettext("⚙️ Configuration"))

    selected_cities = st.multiselect(
        gettext("Select Cities"),
        options=list(CITIES.keys()),
        default=list(CITIES.keys()),
        format_func=lambda x: f"{CITIES[x].emoji} {CITIES[x].name}"
    )

    st.divider()

    show_historical = st.checkbox(gettext("Show Historical Data"), value=False)
    show_comparison = st.checkbox(gettext("Compare with OpenWeather"), value=True)

    st.divider()

    st.info(gettext("""
    **About this app:**
    - Uses Conv1D + LSTM models
    - Trained on 10 years of data
    - Predicts 7 days ahead
    - Compares with OpenWeatherMap
    """))

    if st.button(gettext("🔄 Refresh Predictions"), type="primary"):
        st.cache_data.clear()
        st.rerun()

# Main content
if not selected_cities:
    st.warning(gettext("Please select at least one city from the sidebar."))
    st.stop()

# Create tabs for each city
tabs = st.tabs([f"{CITIES[city].emoji} {CITIES[city].name}" for city in selected_cities])

for tab, city_key in zip(tabs, selected_cities):
    with tab:
        city = CITIES[city_key]

        st.markdown(gettext(f"""
        <div class="city-card">
            <h2>{city.emoji} {city.name}</h2>
            <p>📍 Coordinates: {city.lat}°N, {city.lon}°E</p>
            <p>🕐 Timezone: {city.timezone}</p>
        </div>
        """), unsafe_allow_html=True)

        # Data loading
        with st.spinner(gettext(f"Loading data for {city.name}...")):
            try:
                # Collect and preprocess data
                df_raw = collect_historical_data(city, years_back=10)
                if df_raw is None:
                    st.error(gettext(f"No data available for {city.name}"))
                    continue

                df = preprocess_data(df_raw)
                if df is None:
                    st.error(gettext(f"Error preprocessing data for {city.name}"))
                    continue

                # Charger le modèle et les infos
                model, model_info = load_model_info(city_key)
                if model_info is None:
                    st.error(gettext(f"⚠️ Missing feature info for {city.name}."))
                    continue

                # S'assurer que le DataFrame contient les bonnes colonnes
                expected_features = model_info['feature_cols']
                for col in expected_features:
                    if col not in df.columns:
                        df[col] = 0  # colonne manquante = 0 par défaut

                # Préparer les scalers avec les features attendues
                scalers = prepare_scalers(df, expected_features=expected_features)
                recent_data = df[expected_features].tail(WINDOW_SIZE).values


                # Load model
                model_path = f'templates/assets/température/models/{city_key}_model.keras'
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path)

                    # Make prediction
                    model_pred = predict_7day_forecast(model, recent_data, scalers)

                    # Get OpenWeather forecast
                    ow_forecast = get_openweather_forecast(city) if show_comparison else None

                    # Display predictions
                    st.subheader(gettext("📊 7-Day Temperature Forecast"))

                    # Prepare data for visualization
                    today = datetime.now().date()
                    dates = [today + timedelta(days=i) for i in range(len(model_pred))]

                    # Create dataframe for display
                    pred_df = pd.DataFrame({
                        gettext('Date'): dates,
                        gettext('Avg Temp'): model_pred[:, 0],
                        gettext('Min Temp'): model_pred[:, 1],
                        gettext('Max Temp'): model_pred[:, 2]
                    })

                    if ow_forecast:
                        # Make sure lengths match
                        ow_forecast = ow_forecast[:len(dates)]
                        if len(ow_forecast) < len(dates):
                            # pad missing days with NaN
                            missing = len(dates) - len(ow_forecast)
                            for i in range(missing): # Changed _ to i to avoid conflict
                                ow_forecast.append({'temp_avg': np.nan, 'temp_min': np.nan, 'temp_max': np.nan})

                        pred_df['OW Avg'] = [f['temp_avg'] for f in ow_forecast]
                        pred_df['OW Min'] = [f['temp_min'] for f in ow_forecast]
                        pred_df['OW Max'] = [f['temp_max'] for f in ow_forecast]
                        pred_df['Δ Avg'] = pred_df[gettext('Avg Temp')] - pred_df['OW Avg']

                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(gettext("Today's Avg"), f"{model_pred[0, 0]:.1f}°C")
                    with col2:
                        st.metric(gettext("Today's Min"), f"{model_pred[0, 1]:.1f}°C")
                    with col3:
                        st.metric(gettext("Today's Max"), f"{model_pred[0, 2]:.1f}°C")
                    with col4:
                        if ow_forecast:
                            mae = np.mean(np.abs(pred_df['Δ Avg']))
                            st.metric(gettext("MAE vs OW"), f"{mae:.2f}°C")

                    # Interactive chart
                    fig = go.Figure()

                    # Model predictions
                    fig.add_trace(go.Scatter(
                        x=pred_df[gettext('Date')], y=pred_df[gettext('Avg Temp')],
                        name='Model Avg', mode='lines+markers',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=8)
                    ))

                    fig.add_trace(go.Scatter(
                        x=pred_df[gettext('Date')], y=pred_df[gettext('Max Temp')],
                        name='Model Max', mode='lines',
                        line=dict(color='#f093fb', width=2, dash='dash')
                    ))

                    fig.add_trace(go.Scatter(
                        x=pred_df[gettext('Date')], y=pred_df[gettext('Min Temp')],
                        name='Model Min', mode='lines',
                        line=dict(color='#4facfe', width=2, dash='dash')
                    ))

                    # OpenWeather comparison
                    if ow_forecast:
                        fig.add_trace(go.Scatter(
                            x=pred_df[gettext('Date')], y=pred_df['OW Avg'],
                            name='OpenWeather Avg', mode='lines+markers',
                            line=dict(color='#ff6b6b', width=2),
                            marker=dict(size=6, symbol='x')
                        ))

                    fig.update_layout(
                        title=f"Temperature Forecast - {city.name}",
                        xaxis_title="Date",
                        yaxis_title="Temperature (°C)",
                        hovermode='x unified',
                        height=500,
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Data table
                    st.subheader(gettext("📋 Detailed Forecast"))
                    st.dataframe(
                        pred_df.style.format({
                            gettext('Avg Temp'): '{:.1f}°C',
                            gettext('Min Temp'): '{:.1f}°C',
                            gettext('Max Temp'): '{:.1f}°C',
                            'OW Avg': '{:.1f}°C',
                            'OW Min': '{:.1f}°C',
                            'OW Max': '{:.1f}°C',
                            'Δ Avg': '{:.1f}°C'
                        }),
                        use_container_width=True
                    )

                    # Historical data
                    if show_historical:
                        st.subheader(gettext("📈 Historical Temperature Trends"))

                        hist_df = df[['temp_avg', 'temp_min', 'temp_max']].tail(365)

                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Scatter(
                            x=hist_df.index, y=hist_df['temp_avg'],
                            name='Average', mode='lines',
                            line=dict(color='#667eea', width=2)
                        ))

                        fig_hist.update_layout(
                            title=gettext("Last 365 Days - Temperature History"),
                            xaxis_title=gettext("Date"),
                            yaxis_title=gettext("Temperature (°C)"),
                            height=400,
                            template='plotly_white'
                        )

                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(gettext("**📊 Historical Statistics (Last Year)**"))
                            stats = hist_df['temp_avg'].describe()
                            st.write(gettext(f"- Mean: {stats['mean']:.1f}°C"))
                            st.write(gettext(f"- Std: {stats['std']:.1f}°C"))
                            st.write(gettext(f"- Min: {stats['min']:.1f}°C"))
                            st.write(gettext(f"- Max: {stats['max']:.1f}°C"))

                        with col2:
                            st.markdown(gettext("**📅 Monthly Averages**"))
                            monthly = df[['temp_avg']].groupby(df.index.month).mean()
                            for month, temp in monthly.iterrows():
                                month_name = datetime(2000, month, 1).strftime('%B')
                                st.write(f"- {month_name}: {temp['temp_avg']:.1f}°C")

                else:
                    st.error(gettext(f"Model not found for {city.name}. Please train the model first."))
                    st.info(gettext(f"Expected path: {model_path}"))

            except Exception as e:
                st.error(gettext(f"Error processing {city.name}: {str(e)}"))
                st.exception(e)

# Section Présentation
st.header(gettext("Présentation"))
st.markdown(gettext(
    """
    Ce projet vise à prédire la méteo sur différente ville en fonction de l'historique et des paramètres tel que la pression, le taux de pluis etc ... Les séries temporelles sont un domaine complexe, que l'on retrouve dans la météo, nénamoins le cas d'application est diverse et varié tel que :

    **Applications potentielles :**
    - **Finance :** Prévision des prix des actions, analyse des tendances du marché.
    - **Santé :** Surveillance des signes vitaux, prédiction des épidémies.
    - **Agriculture :** Prévision des températures, prévision des plantations.
    - **Énergie :** Prévision de la demande énergétique, gestion des réseaux électriques.
    - **Transport :** Prévision du trafic, gestion des flottes de véhicules.
    - **Commerce de détail :** Prévision des ventes, gestion des stocks.
    Pour cela, nous avons utilisé l'api Meteostat qui nous permet de récupérer les données météorologiques historiques de différentes villes. Nous avons choisi Paris et la Silicon Valley pour leur contraste climatique. Nous avons fait le dernier entrainement sur les données de 8 octobre 2025, plus les données sont récentes plus la précision est bonne.
    Les données collectées incluent des paramètres tels que la température moyenne, minimale et maximale, les précipitations, la vitesse du vent, la pression atmosphérique, etc. Ces données sont ensuite nettoyées et standardisées pour garantir une qualité optimale avant l'entraînement du modèle.
    """
))

# Section Architecture du Modèle
st.header(gettext("Architecture du Modèle"))
st.markdown(gettext(
    """
    Pour prédire des séries temporelles multivariées, il est crucial de capturer à la fois les motifs locaux et les dépendances séquentielles dans les données. Notre modèle combine des couches convolutionnelles et LSTM pour atteindre cet objectif.

    Le modèle se compose de plusieurs blocs :

    - **Couche Conv1D :** Cette couche extrait les motifs locaux dans les séquences temporelles grâce à des filtres convolutifs. Elle permet au modèle de détecter des tendances ou des motifs répétitifs sur de petites fenêtres de temps.

    - **Couches LSTM :** Trois couches LSTM sont empilées pour capturer les dépendances temporelles à long terme. Les deux premières couches renvoient des séquences complètes (`return_sequences=True`) afin que les informations temporelles puissent être transmises aux couches suivantes. La troisième couche renvoie uniquement le dernier état caché (`return_sequences=False`), qui résume l'information séquentielle.

    - **Couches Dense :** Après l'extraction des motifs et des dépendances séquentielles, les couches denses transforment la représentation en sorties prédictives. Des couches `Dropout` sont intercalées pour réduire le surapprentissage et améliorer la généralisation.

    Le modèle produit une sortie structurée pour plusieurs pas de temps (`forecast_horizon`) et pour plusieurs variables cibles (`n_targets`). La compilation utilise la **loss Huber** adaptée aux valeurs aberrantes, l'optimiseur **Adam** avec un taux d'apprentissage de 0.001, et suit les métriques **MAE** et **MSE** pour évaluer les performances.

    Les hyperparamètres clés incluent : la taille de la fenêtre (`window_size`), le nombre de filtres et de neurones dans les couches LSTM et Dense, le taux de dropout et le nombre de pas de temps prédits (`forecast_horizon`).
    """
))
st.image("./templates/assets/température/model_architecture.png",
         caption="Structure du modèle de prédiction de météo",
         width=800)  # largeur en pixels


# Section Résultats
st.header(gettext("Résultats"))
st.markdown(gettext(
    """
    Les tests montrent que le modèle prédit correctement les températures journalières pour différentes localisations.

    📍 **Paris**
    MAE : 2.84°C
    Le modèle suit globalement les valeurs observées, avec de petits écarts pour certaines journées.

    📍 **Silicon Valley**
    MAE : 0.98°C
    Les prédictions sont proches des valeurs réelles, montrant une bonne précision du modèle.

    Ces résultats confirment que le modèle généralise bien et que les techniques de régularisation, comme le dropout, permettent de limiter le surajustement.
    """
))


# Section Coût et Maintenance
st.header(gettext("Coût de Développement"))
st.markdown(gettext(
    """
    Le modèle a été entraîné sur une machine Linux avec les caractéristiques suivantes :

    - **Processeur :** AMD Ryzen 5 3500X 6 cœurs, fréquence max 4,12 GHz
    - **RAM :** 15 Go

    L’entraînement a été extrêmement rapide, prenant moins de **2 secondes** pour traiter **10 ans de données** d’une seule ville.

    Ces performances montrent que le modèle est très léger et efficace, capable de générer des prédictions rapides tout en restant précis.

    **Analyse des coûts :** L’usage de ressources limitées rend ce modèle économique et facilement déployable sur des machines standards.

    **Perspectives d'amélioration :** Il serait possible d’étendre le modèle à plusieurs villes sur le même modéle ou d’intégrer des données supplémentaires sans augmenter significativement le temps de calcul.
    """
))


# Footer
st.markdown(gettext(
    """
    ---
    Développé par [Gabriel Marie-Brisson](https://gabriel.mariebrisson.fr )
    """
))
