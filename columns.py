import pandas as pd

file_path = r"C:\Users\ADMIN\Downloads\archive (1)\spotify-tracks-dataset.csv"

df = pd.read_csv(file_path)

print(df.columns.tolist())