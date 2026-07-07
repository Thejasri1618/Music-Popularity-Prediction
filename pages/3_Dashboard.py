import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv("spotify-tracks-dataset.csv")

st.title("📊 Music Analytics Dashboard")
st.markdown("Explore Spotify song trends using interactive charts.")

st.divider()

# ---------------------------------
# Top Statistics
# ---------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("🎵 Songs", len(df))
col2.metric("🎤 Artists", df["artists"].nunique())
col3.metric("🎼 Genres", df["track_genre"].nunique())
col4.metric("⭐ Avg Popularity", round(df["popularity"].mean(),2))

st.divider()

# ---------------------------------
# Popularity Distribution
# ---------------------------------

st.subheader("⭐ Popularity Distribution")

fig = px.histogram(
    df,
    x="popularity",
    nbins=30,
    color_discrete_sequence=["#1DB954"]
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Top Genres
# ---------------------------------

st.subheader("🎼 Top 10 Genres")

top_genres = df["track_genre"].value_counts().head(10)

fig = px.bar(
    x=top_genres.index,
    y=top_genres.values,
    color=top_genres.values,
    color_continuous_scale="greens"
)

fig.update_layout(
    xaxis_title="Genre",
    yaxis_title="Number of Songs"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Energy vs Popularity
# ---------------------------------

st.subheader("⚡ Energy vs Popularity")

fig = px.scatter(
    df.sample(3000),
    x="energy",
    y="popularity",
    color="danceability",
    hover_data=["track_name"],
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Explicit Songs
# ---------------------------------

st.subheader("🎧 Explicit vs Non Explicit Songs")

explicit = df["explicit"].value_counts()

fig = px.pie(
    values=explicit.values,
    names=["Non Explicit","Explicit"]
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Dataset Preview
# ---------------------------------

st.subheader("📄 Dataset")

st.dataframe(df.head(20), use_container_width=True)