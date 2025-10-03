import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
from preprocess import preprocess

# Load data
df = pd.read_csv("D:\data\creditcard.csv")
                 
#preprocess 
df , scaler = preprocess(df , fit_scaler = True)

# split data and target 

X = df.drop("Class" , axis =1)
y = df["Class"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100 , random_state=42)
model.fit(X_train, y_train)

# Save trained model 
import os
import pickle

# ✅ Always make sure the folder exists
os.makedirs("D:/MY_ML_Project/Modal", exist_ok=True)

# ✅ Use forward slashes (best for cross-platform)
with open("D:/MY_ML_Project/Modal/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)


print("Model trained and saved successfully!")

