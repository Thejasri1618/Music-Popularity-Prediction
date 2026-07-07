import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

df = pd.read_csv("spotify-tracks-dataset.csv")

st.title("🎵 AI Music Popularity Prediction System")

st.markdown
-

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎵 Total Songs", len(df))

with col2:
    st.metric("🎼 Genres", df["track_genre"].nunique())

with col3:
    st.metric("⭐ Avg Popularity", round(df["popularity"].mean(),2))

with col4:
    st.metric("👨‍🎤 Artists", df["artists"].nunique())

st.divider()


st.subheader("📄 Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

st.divider()

st.subheader("⚙ Project Workflow")

st.markdown("""
