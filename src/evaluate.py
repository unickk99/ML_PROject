import pandas as pd
import pickle
from preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


test = pd.read_csv(r"D:\data\test.csv")

with open("D:\MY_ML_Project\Modal\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

test , _ = preprocess(test , scaler= scaler ,fit_scaler= False)

#load the modal
with open("D:/MY_ML_Project/Modal/scaler.pkl", "rb") as f:
    model = pickle.load(f)

# data and target

X_test = test.drop("Class", axis=1)
y_test = test["Class"]

#load modal
with open("D:/MY_ML_Project/Modal/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions
y_pred = model.predict(X_test)


# Evaluate
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
