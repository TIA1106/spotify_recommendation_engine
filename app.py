# app.py - Premium Pastel Spotify Dashboard with Gradient Cards & Animated Charts

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import numpy as np

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="🎧 Spotify Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------
# Gradient Background, Fonts & Sliders Fix
# ---------------------------
st.markdown("""
<style>
/* Body gradient & fonts */
body {
    background: linear-gradient(135deg, #FFE4E1, #FFD1DC);
    font-family: 'Segoe UI', sans-serif;
}

/* Headers */
h1 { color: #FF69B4; text-align:center; font-size:40px; }
h3 { color: #FFB6C1; text-align:center; }

/* Buttons */
.stButton>button {
    background-color: #FFB6C1;
    color: black;
    border-radius: 10px;
    height: 40px;
    font-weight:bold;
}

/* Metric cards labels */
.stMetricLabel { color: #FF69B4; font-weight:bold; }

/* Slider customization: remove number background & color text */
div[data-baseweb="slider"] span {
    background: none !important;
    color: #FF69B4 !important;
    font-weight: bold;
}

/* Slider track gradient and handle color */
div[data-baseweb="slider"] .base-slider-inner .base-slider-track {
    background: linear-gradient(90deg, #FFA07A, #FFB6C1) !important;  /* orange to pink */
}
div[data-baseweb="slider"] .base-slider-inner .base-slider-handle {
    background: #FF69B4 !important;  /* bright pink handle */
}

div[data-baseweb="slider"] .base-slider-inner .base-slider-handle {
    background: #FFB6C1 !important;
}

/* Rounded table rows + hover effect */
.stDataFrame table {
    border-collapse: separate;
    border-spacing: 0 10px;
}
.stDataFrame td, .stDataFrame th {
    border: none !important;
    padding: 10px;
}
.stDataFrame tr {
    border-radius: 10px;
    background-color: #FFE4E1;
}
.stDataFrame tr:hover {
    background-color: #FFB6C1 !important;
    color: white;
}

/* Gradient metric cards */
.metric-card {
    border-radius: 15px;
    padding: 20px;
    color: white;
    text-align: center;
    font-weight: bold;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown("<h1>🎧 Spotify Song Popularity Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3>Analyze, Predict, and Discover Similar Songs!</h3>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv("spotify.csv")
    df = df.dropna()
    df = df[df['popularity'] > 0]
    if 'album_cover_url' not in df.columns:
        df['album_cover_url'] = 'https://via.placeholder.com/50'
    return df

df = load_data()
features = ['danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']

st.sidebar.markdown("## 🎶 Filter & Explore")
genres = df['track_genre'].unique()
genre = st.sidebar.selectbox("Select Genre", genres)
filtered = df[df['track_genre'] == genre]

st.sidebar.markdown("## 🎯 Find Similar Songs")
@st.cache_data
def fit_knn(df, features):
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(X_scaled)
    return knn, scaler

knn_model, scaler = fit_knn(df, features)

with st.sidebar.expander("🔍 Select Song"):
    song_choice = st.selectbox("Select a Song", df['track_name'])
    if st.button("Show Similar"):
        idx = df[df['track_name'] == song_choice].index[0]
        song_features = scaler.transform(df.loc[[idx], features])
        distances, indices = knn_model.kneighbors(song_features)
        similar_songs = df.iloc[indices[0][1:]]
        st.markdown("### 🎵 Similar Songs")
        for i, row in similar_songs.iterrows():
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin-bottom:10px; padding:5px; background:#FFE4E1; border-radius:10px;'>
                <img src="{row['album_cover_url']}" width="50" height="50" style='border-radius:5px; margin-right:10px;'/>
                <div>
                    <b>{row['track_name']}</b><br>
                    {row['artists']} | {row['track_genre']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------
# Gradient Metric Cards with Icons
# ---------------------------
st.subheader("🏆 Genre Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<div class='metric-card' style='background:linear-gradient(135deg,#FF69B4,#FFB6C1)'>🎵 Top Energy<br><h2>{round(filtered['energy'].max(),2)}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card' style='background:linear-gradient(135deg,#FFC0CB,#FFD1DC)'>⚡ Top Danceability<br><h2>{round(filtered['danceability'].max(),2)}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card' style='background:linear-gradient(135deg,#FFB6C1,#FF69B4)'>💖 Avg Popularity<br><h2>{round(filtered['popularity'].mean(),1)}</h2></div>", unsafe_allow_html=True)

# ---------------------------
# Visualizations with subtle animation
# ---------------------------
st.subheader(f"📊 Visualizations: {genre}")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(filtered, x='danceability', y='energy',
                      color='popularity', size='valence',
                      hover_data=['track_name', 'artists'],
                      color_continuous_scale=px.colors.sequential.RdPu,
                      title="Danceability vs Energy vs Popularity")
    fig1.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='white')))
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255, 228, 241,0.3)')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(filtered, x='popularity', nbins=30,
                        title="Popularity Distribution",
                        color_discrete_sequence=['#FFB6C1'])
    fig2.update_traces(opacity=0.8)
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255, 228, 241,0.3)')
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Predict Song Popularity
# ---------------------------
st.subheader("🎯 Predict Song Popularity")
X = df[features]
y = df['popularity']
scaler_pred = StandardScaler()
X_scaled = scaler_pred.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

st.markdown("Adjust sliders to define a song's audio features:")
vals = []
for feature in features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    vals.append(st.slider(feature, min_val, max_val, mean_val))

if st.button("Predict Popularity"):
    pred = model.predict([vals])[0]
    st.markdown(f"<h2 style='color:#FF69B4; text-align:center'>Predicted Popularity: {pred:.2f}/100</h2>", unsafe_allow_html=True)
