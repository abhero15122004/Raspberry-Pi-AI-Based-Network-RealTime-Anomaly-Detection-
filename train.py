import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
df = pd.read_csv("network_data.csv", header=None)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Train Isolation Forest
model_if = IsolationForest(contamination=0.05, random_state=42)
model_if.fit(data_scaled)
joblib.dump(model_if, "model_isolation_forest.pkl")

# Train PCA
pca = PCA(n_components=0.95)
pca.fit(data_scaled)
joblib.dump(pca, "model_pca.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Models trained and saved.")
