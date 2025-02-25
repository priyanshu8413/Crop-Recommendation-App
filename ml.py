import os
import pickle
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")  # Ensure this file exists

# Define features (X) and target (y)
X = df.drop(columns=["label"])  # Ensure "label" is correct
y = df["label"]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AdaBoost Classifier
model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Save model in the correct directory
os.makedirs("../Streamlit/Crop_Production_App/Crop_Production_App", exist_ok=True)  # Ensure folder exists
model_path = "crop_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved successfully at: {os.path.abspath(model_path)}")
