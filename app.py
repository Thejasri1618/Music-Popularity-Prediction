import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="AI Music Popularity Prediction",
    page_icon="🎵",
    layout="wide"
)

# ---------------- CSS ----------------

try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    df = pd.read_csv(
        r"C:\Users\ADMIN\Downloads\archive (1)\spotify-tracks-dataset.csv"
    )

    df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore")

    return df


@st.cache_resource
def load_model():
    model = joblib.load("music_popularity_model.pkl")
    encoder = joblib.load("genre_encoder.pkl")
    return model, encoder


try:
    df = load_data()
    model, encoder = load_model()

except Exception as e:
    st.error(f"Error loading files:\n{e}")
    st.stop()
# ---------------- SIDEBAR ----------------

with st.sidebar:

    selected = option_menu(
        "🎵 Spotify AI",
        ["Home","Predict","Dashboard","Model","About"],
        icons=["house","music-note","bar-chart","cpu","info-circle"],
        default_index=0
    )

# ============================================================
# HOME
# ============================================================

if selected=="Home":

    st.title("🎵 AI Music Popularity Prediction")

    st.markdown("""
Predict Spotify song popularity using Machine Learning.

This project uses **Random Forest Regression** trained on Spotify tracks.
""")

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Songs",len(df))
    c2.metric("Genres",df["track_genre"].nunique())
    c3.metric("Artists",df["artists"].nunique())
    c4.metric("Average Popularity",round(df["popularity"].mean(),2))

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10),use_container_width=True)

    st.divider()

    st.subheader("Popularity Distribution")

    fig=px.histogram(
        df,
        x="popularity",
        nbins=30,
        color_discrete_sequence=["green"]
    )

    st.plotly_chart(fig,use_container_width=True)

# ============================================================
# PREDICT
# ============================================================

elif selected=="Predict":

    st.title("🎯 Predict Song Popularity")

    col1,col2=st.columns(2)

    with col1:

        duration=st.number_input(
            "Duration (ms)",
            10000,
            600000,
            180000
        )

        explicit=st.selectbox(
            "Explicit",
            [0,1]
        )

        danceability=st.slider(
            "Danceability",
            0.0,
            1.0,
            0.60
        )

        energy=st.slider(
            "Energy",
            0.0,
            1.0,
            0.70
        )

        loudness=st.slider(
            "Loudness",
            -60.0,
            5.0,
            -7.0
        )

        speechiness=st.slider(
            "Speechiness",
            0.0,
            1.0,
            0.05
        )

    with col2:

        acousticness=st.slider(
            "Acousticness",
            0.0,
            1.0,
            0.20
        )

        instrumentalness=st.slider(
            "Instrumentalness",
            0.0,
            1.0,
            0.00
        )

        liveness=st.slider(
            "Liveness",
            0.0,
            1.0,
            0.15
        )

        valence=st.slider(
            "Valence",
            0.0,
            1.0,
            0.55
        )

        tempo=st.number_input(
            "Tempo",
            40.0,
            250.0,
            120.0
        )

        genre=st.selectbox(
            "Genre",
            encoder.classes_
        )

    if st.button("Predict Popularity",use_container_width=True):

        genre_encoded=encoder.transform([genre])[0]

        sample=pd.DataFrame([{

            "duration_ms":duration,
            "explicit":explicit,
            "danceability":danceability,
            "energy":energy,
            "key":5,
            "loudness":loudness,
            "mode":1,
            "speechiness":speechiness,
            "acousticness":acousticness,
            "instrumentalness":instrumentalness,
            "liveness":liveness,
            "valence":valence,
            "tempo":tempo,
            "time_signature":4,
            "track_genre":genre_encoded

        }])

        prediction=model.predict(sample)[0]

        prediction=max(0,min(100,prediction))

        st.success(f"Predicted Popularity : {prediction:.2f}")

        fig=go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            gauge={
                "axis":{"range":[0,100]}
            }
        ))

        fig.update_layout(height=400)

        st.plotly_chart(fig,use_container_width=True)
        # ============================================================
# DASHBOARD
# ============================================================

elif selected == "Dashboard":

    st.title("📊 Music Analytics Dashboard")

    tab1, tab2, tab3 = st.tabs([
        "Popularity",
        "Genres",
        "Correlation"
    ])

    with tab1:

        st.subheader("Popularity Distribution")

        fig = px.histogram(
            df,
            x="popularity",
            nbins=25,
            color_discrete_sequence=["#1DB954"]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 10 Popular Songs")

        top = df.sort_values(
            "popularity",
            ascending=False
        ).head(10)

        fig = px.bar(
            top,
            x="track_name",
            y="popularity",
            color="popularity"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:

        st.subheader("Top Genres")

        genre_df = (
            df["track_genre"]
            .value_counts()
            .head(15)
            .reset_index()
        )

        genre_df.columns = [
            "Genre",
            "Songs"
        ]

        fig = px.bar(
            genre_df,
            x="Genre",
            y="Songs",
            color="Songs"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:

        st.subheader("Danceability vs Popularity")

        fig = px.scatter(
            df.sample(2000),
            x="danceability",
            y="popularity",
            color="energy",
            hover_data=["track_name"]
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODEL
# ============================================================

elif selected == "Model":

    st.title("🤖 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Algorithm",
        "Random Forest"
    )

    col2.metric(
        "Prediction",
        "Regression"
    )

    col3.metric(
        "Dataset",
        f"{len(df):,} Songs"
    )

    st.divider()

    st.subheader("Features Used")

    features = [
        "duration_ms",
        "explicit",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "track_genre"
    ]

    st.dataframe(
        pd.DataFrame({
            "Model Features": features
        }),
        use_container_width=True
    )

    st.success("""
✔ Random Forest Regressor

✔ Trained using Spotify Tracks Dataset

✔ Predicts song popularity between 0–100.
""")


# ============================================================
# ABOUT
# ============================================================

elif selected == "About":

    st.title("ℹ About Project")

    st.markdown("""
# 🎵 AI Music Popularity Prediction

### Problem Statement

Predict the popularity score of a Spotify song using Machine Learning.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Streamlit

---

## Machine Learning

- Random Forest Regressor

---

## Dataset

Spotify Tracks Dataset

Contains more than **114,000 songs** and their audio features.

---

## Features

- Danceability
- Energy
- Loudness
- Tempo
- Acousticness
- Instrumentalness
- Speechiness
- Valence
- Genre

---

## Developed By

Theja Sri
""")
