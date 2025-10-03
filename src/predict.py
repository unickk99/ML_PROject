# src/predict.py
import pandas as pd
import pickle
from preprocess import preprocess

# Load new/unlabeled data
df_new = pd.read_csv("data")

# Remove Class if exists
if "Class" in df_new.columns:
    df_new = df_new.drop("Class", axis=1)

# Load scaler
with open("D:/MY_ML_Project/Modal/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Preprocess
df_new, _ = preprocess(df_new, scaler=scaler, fit_scaler=False)

# Load model
with open("D:/MY_ML_Project/Modal/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict
predictions = model.predict(df_new)
probabilities = model.predict_proba(df_new)[:,1]

# Add predictions to dataframe
df_new["Predicted_Class"] = predictions
df_new["Fraud_Probability"] = probabilities


# Output results
print(df_new.head())

# Save to CSV
df_new.to_csv("D:/data/predictions.csv", index=False)
print("✅ Predictions saved to predictions.csv")
