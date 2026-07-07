import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="Model Performance",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Model Performance")

st.write("""
This page evaluates the **Random Forest Regression Model**
used for predicting Spotify song popularity.
""")

st.divider()

# ---------------------------------------
# Load Dataset
# ---------------------------------------

df = pd.read_csv("spotify-tracks-dataset.csv")

# Remove unnecessary columns
drop_cols = ["track_id", "track_name", "artists", "album_name"]
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

# Encode Genre
encoder = joblib.load("genre_encoder.pkl")
df["track_genre"] = encoder.transform(df["track_genre"])

# Features & Target
X = df.drop("popularity", axis=1)
y = df["popularity"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Load Model
model = joblib.load("music_popularity_model.pkl")

# Predictions
pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

# ---------------------------------------
# Metrics Cards
# ---------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("📉 MAE", round(mae,2))
col2.metric("📊 RMSE", round(rmse,2))
col3.metric("🏆 R² Score", round(r2,3))

st.divider()

# ---------------------------------------
# Feature Importance
# ---------------------------------------

st.subheader("🌲 Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature")
)

st.divider()

# ---------------------------------------
# Model Comparison
# ---------------------------------------

st.subheader("📋 Model Comparison")

comparison = pd.DataFrame({
    "Model":[
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Performance":[
        "Medium",
        "Good",
        "Excellent"
    ],
    "Reason":[
        "Cannot capture complex relationships",
        "Can overfit easily",
        "Best accuracy and generalization"
    ]
})

st.dataframe(comparison, use_container_width=True)

st.divider()

# ---------------------------------------
# Why Random Forest?
# ---------------------------------------

st.subheader("💡 Why Random Forest?")

st.success("""
✔ Handles non-linear relationships

✔ High prediction accuracy

✔ Reduces overfitting

✔ Robust for large datasets

✔ Works well with mixed numerical features
""")