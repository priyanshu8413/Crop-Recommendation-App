import pickle
import os

# Ensure the directory exists
os.makedirs("../Streamlit/Crop_Production_App/Crop_Production_App/Crop_Production_App", exist_ok=True)

# Dummy model (replace with your trained model)
model = {"message": "This is a dummy model"}

# Save model
model_path = "../Streamlit/Crop_Production_App/Crop_Production_App/Crop_Production_App/crop_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved successfully at: {os.path.abspath(model_path)}")
