import os

encoder_path = "../Streamlit/Crop_Production_App/Crop_Production_App/Crop_Production_App/Crop_Production_App/label_encoder.pkl"

if os.path.exists(encoder_path):
    print(f"✅ Label Encoder found at: {os.path.abspath(encoder_path)}")
else:
    print("❌ Label Encoder NOT found! Check the directory.")
