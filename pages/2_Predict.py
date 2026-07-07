import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Predict Song Popularity",
    page_icon="🎵",
    layout="wide"
)

BASE_DIR = Path(__file__).parent.parent

# Load model and encoder
model = joblib.load(BASE_DIR / "music_popularity_model.pkl")
encoder = joblib.load(BASE_DIR / "genre_encoder.pkl")

st.title("🎵 Predict Song Popularity")
st.write("Enter the song details below and click **Predict Popularity**.")

st.divider()

# -------------------------------
# Input Layout
# -------------------------------
left, right = st.columns(2)

with left:
    duration_ms = st.number_input("Duration (ms)", 10000, 600000, 180000)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.50)
    energy = st.slider("Energy", 0.0, 1.0, 0.50)
    loudness = st.slider("Loudness", -60.0, 5.0, -10.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.30)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.00)

with right:
    liveness = st.slider("Liveness", 0.0, 1.0, 0.20)
    valence = st.slider("Valence", 0.0, 1.0, 0.50)
    tempo = st.number_input("Tempo", 40.0, 250.0, 120.0)
    time_signature = st.selectbox("Time Signature", [3, 4, 5])
    explicit = st.selectbox("Explicit", [0, 1])

    genres = encoder.classes_
    genre = st.selectbox("Genre", genres)

genre_encoded = encoder.transform([genre])[0]

st.divider()

if st.button("🎯 Predict Popularity", use_container_width=True):

    features = pd.DataFrame([{
        "duration_ms": duration_ms,
        "explicit": explicit,
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "time_signature": time_signature,
        "track_genre": genre_encoded
    }])

    prediction = model.predict(features)[0]

    prediction = max(0, min(100, prediction))

    st.success(f"🎵 Predicted Popularity Score: {prediction:.2f}/100")

    st.progress(int(prediction))

    if prediction >= 75:
        st.success("🔥 This song has High Popularity Potential!")

    elif prediction >= 50:
        st.warning("⭐ This song has Medium Popularity Potential!")

    else:
        st.error("🎧 This song has Low Popularity Potential.")