import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

def preprocess(df, scaler=None, fit_scaler=True):
    if fit_scaler:
        scaler = StandardScaler()
        df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    else:
        df["Amount_scaled"] = scaler.transform(df[["Amount"]])

   # Save scaler if fitting for evalution
    if fit_scaler:
        os.makedirs("D:/MY_ML_Project/Modal", exist_ok=True)
        with open("D:/MY_ML_Project/Modal/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        

    if "Amount" in df.columns:
        df = df.drop("Amount", axis=1)

    return df, scaler