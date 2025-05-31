import pandas as pd
import numpy as np
import joblib

# Load models
model_if = joblib.load("model_isolation_forest.pkl")
pca = joblib.load("model_pca.pkl")
scaler = joblib.load("scaler.pkl")

# Load and preprocess data
df = pd.read_csv("network_data.csv", header=None)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
data_scaled = scaler.transform(df)

# Predict using Isolation Forest
if_preds = model_if.predict(data_scaled)
if_anomalies = np.where(if_preds == -1)[0]

# Predict using PCA reconstruction error
X_pca = pca.transform(data_scaled)
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.mean(np.abs(data_scaled - X_reconstructed), axis=1)
threshold = np.percentile(reconstruction_error, 95)
pca_anomalies = np.where(reconstruction_error > threshold)[0]

# Hybrid anomalies
hybrid_anomalies = np.union1d(if_anomalies, pca_anomalies)
anomalies_df = df.iloc[hybrid_anomalies]
anomalies_df.to_csv("anomalies_detected.csv", index=False)

print("✅ Anomalies saved to anomalies_detected.csv")
