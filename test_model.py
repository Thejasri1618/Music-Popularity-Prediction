import os
import joblib

print("Current Folder:", os.getcwd())

print("Files in this folder:")
print(os.listdir())

print()

print("Loading model...")

model = joblib.load("music_popularity_model.pkl")

print("Model Loaded Successfully!")

print(type(model))

