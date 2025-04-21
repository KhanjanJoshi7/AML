import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import re

# Load model and encoder
model = joblib.load("movie_like_predictor.pkl")
mlb = joblib.load("genre_encoder.pkl")

# Sample genre list from the encoder
all_genres = mlb.classes_.tolist()

st.title("🎬 Movie Like Predictor")

movies = pd.read_csv("movies.csv")  # <-- This was missing

def clean_title(title):
    title = str(title).strip()
    # Remove year in parentheses, e.g. "Toy Story (1995)" → "Toy Story"
    return re.sub(r"\s+\(\d{4}\)", "", title)

movies['clean_title'] = movies['title'].apply(clean_title)

# Remove duplicates and sort
movie_list = sorted(movies['clean_title'].drop_duplicates())

# Dropdown for movie selection

# --- User Inputs ---
movie_title = st.selectbox("Choose a movie", movie_list)
year = st.number_input("Release Year", min_value=1900, max_value=datetime.now().year, value=2008)
user_activity = st.slider("User Activity (No. of Ratings Given)", 0, 1000, 100)
user_avg_rating = st.slider("User's Avg Rating", 0.0, 5.0, 3.5)
selected_genres = st.multiselect("Select Genres", all_genres, default=["Action", "Drama"])

# --- Feature Preparation ---
genre_vector = [1 if genre in selected_genres else 0 for genre in all_genres]
features = genre_vector + [year, user_activity, user_avg_rating]
features_df = pd.DataFrame([features], columns=all_genres + ["year", "user_activity", "user_avg_rating"])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(features_df)[0]
    st.success("✅ You will likely **like** this movie!" if prediction else "❌ You may **not** like this movie.")
