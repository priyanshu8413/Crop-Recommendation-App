import streamlit as st
#import requests
import pickle
import os
import base64
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import requests
# Load Model
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Label Encoder
with open(
        "../Streamlit/Crop_Production_App/Crop_Production_App/Crop_Production_App/Crop_Production_App/label_encoder.pkl",
        "rb") as f:
    le = pickle.load(f)

# Define the path to crop images folder
IMAGE_FOLDER = "project_Photo"

# Load the dataset to train the label encoder correctly
file_path = "Crop_recommendation.csv"
df = pd.read_csv(file_path)

# Train the LabelEncoder properly
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    predicted_label = model.predict(input_data)[0]
    predicted_crop = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_crop
# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "predicted_crop" not in st.session_state:
    st.session_state.predicted_crop = None


def main():
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Prediction":
        prediction_page()
    elif st.session_state.page == "Result":
        result_page()
    elif st.session_state.page == "About":
        about_page()

def  home_page():
    st.sidebar.title("Navigation")
    if st.sidebar.button("Go to Prediction"):
        st.session_state.page = "Prediction"
        st.rerun()
    page_bg_img = '''
        <style>
            .stApp {
                background: url("https://i.pinimg.com/originals/a0/8f/de/a08fde1a5b62749ada7453ec27af12cd.jpg") no-repeat center center fixed;
                background-size: cover;
            }
            .center-text {
                text-align: center;
                font-size: 30px !important;
                font-weight: bold;
                color: black;
                padding: 100px;
            }
            .title-text {
                font-size: 50px !important;
                font-weight: bold !important;
                color: #ffcc00;
                text-align: center;
                line-height: 1.1; 
            }
        </style>
        '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown('<p class="title-text">🏡 Home Page</p>', unsafe_allow_html=True)
    st.markdown('<p class="title-text">🌾 Crop Recommendation App</p>', unsafe_allow_html=True)
    welcome_placeholder = st.empty()

        # Text to Display Word by Word
    welcome_text = "Welcome to Crop Recommendation App".split()  # Splitting into words

    while True:
         for i in range(1, len(welcome_text) + 1):
            current_text = " ".join(welcome_text[:i])  # Show words progressively
            welcome_placeholder.markdown(f'<p class="center-text">{current_text}</p>', unsafe_allow_html=True)
            time.sleep(0.1)  # Adjust speed as needed

            time.sleep(1)  # Hold full text for 1 second before restarting

def  prediction_page():
        st.sidebar.title("Navigation")
        if st.sidebar.button("Go to About"):
            st.session_state.page = "About"
            st.rerun()
        st.markdown("""<style>
            .stApp {
                background: url("https://i.imgur.com/npfXN0d.jpeg") no-repeat center center fixed;
                background-size: cover;
            }
        </style>""", unsafe_allow_html=True)

        st.markdown("# **Enter Soil and Climate Parameters to Predict the Best Crop:**")
        st.markdown(
            """
            <style>
                /* Make input labels bold and red */
                label {
                    font-size: 30px !important;
                    font-weight: bold !important;
                    color: black !important;
                    
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        def get_current_location():
            """Fetch user's approximate location using a free IP geolocation API."""
            try:
                response = requests.get("https://ipinfo.io/json")
                data = response.json()
                if "loc" in data:
                    lat, lon = map(float, data["loc"].split(","))
                    return lat, lon
                else:
                    st.error("❌ Could not detect location.")
                    return None, None
            except requests.exceptions.RequestException:
                st.error("❌ Error fetching location.")
                return None, None

        def get_nasa_data(lat,lon):
            try:
                # Get current and past dates (7 days ago)
                end_date = (datetime.today()-timedelta(days=4)).strftime('%Y%m%d')
                start_date = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')

                # NASA POWER API URL
                url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,RH2M,PRECTOTCORR&community=SB&&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"

                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                # ✅ Fix: Ensure response has expected keys
                if "properties" not in data or "parameter" not in data["properties"]:
                    st.error("❌ NASA API response is missing expected data. Please try again.")
                    return None

                # Extract weather parameters
                parameters = data["properties"]["parameter"]
                temperature_data = parameters.get("T2M", {})
                humidity_data = parameters.get("RH2M", {})
                rainfall_data = parameters.get("PRECTOTCORR", {})

                # Get latest available values
                temperature = list(temperature_data.values())[-1] if temperature_data else None
                humidity = list(humidity_data.values())[-1] if humidity_data else None
                rainfall = list(rainfall_data.values())[-1] if rainfall_data else None

                # Ignore invalid values (-999)
               # temperature = temperature if temperature != -999 else None
                #humidity = humidity if humidity != -999 else None
                #rainfall = rainfall if rainfall != -999 else None

                # Ensure valid response
                if temperature is None or humidity is None or rainfall is None:
                    st.warning("⚠ NASA API did not return valid values. Please enter manually.")
                    return None

                return {
                    "Temperature (°C)": temperature,
                    "Humidity (%)": humidity,
                    "Rainfall (mm)": rainfall,
                }

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error fetching NASA data: {e}")
                return None
        # Streamlit UI
        #st.title("Crop Prediction with NASA Data")

        if st.button("Put The Temperature Rainfall & Humidity"):
            lat, lon = get_current_location() # Example coordinates
            nasa_data = get_nasa_data(lat,lon)

            if nasa_data:
                st.session_state["temperature"] = str(nasa_data.get('Temperature (°C)', "25"))
                st.session_state["humidity"] = str(nasa_data.get('Humidity (%)', "50"))
                st.session_state["rainfall"] = str(nasa_data.get('Rainfall (mm)', "5"))
                st.toast("✅ Autofilled values from NASA API!", icon="✔")
            else:
                st.warning("⚠ NASA API did not return valid data. Please enter manually.")
                st.session_state.setdefault("temperature", "25")
                st.session_state.setdefault("humidity", "50")
                st.session_state.setdefault("rainfall", "5")

            st.rerun()

        # Input Fields
        st.text_input("Temperature (°C)", key="temperature", placeholder="Enter Temperature Value")
        st.text_input("Humidity (%)", key="humidity", placeholder="Enter Humidity Value")
        st.text_input("Rainfall (mm)", key="rainfall", placeholder="Enter Rainfall Value")


        def validate_input(field_name, min_val, max_val, label):
               value = st.session_state.get(field_name, "")

            # Check if the value is empty
               if value == "":
                  return

            # Check if input is a valid number
               try:
                 num_value = float(value)
                 if not (min_val <= num_value <= max_val):
                    st.toast(f"❌ {label} must be between {min_val} and {max_val}.", icon="⚠️")
                    st.session_state[f"{field_name}_error"] = True
                 else:
                     st.session_state[f"{field_name}_error"] = False  # Clear error if valid
               except ValueError:
                    st.toast(f"❌ Please enter a numeric value for {label}.", icon="⚠️")
                    st.session_state[f"{field_name}_error"] = True

        # Input fields with individual validation
        st.text_input("Nitrogen (N) Content", placeholder="Enter The Nitrogen Content", key="N",
                            on_change=validate_input, args=("N", 0, 200, "Nitrogen (N)"))

        st.text_input("Phosphorous (P) Content", placeholder="Enter Phosphorous Content", key="P",
                          on_change=validate_input, args=("P", 0, 200, "Phosphorous (P)"))

        st.text_input("Potassium (K) Content", placeholder="Enter Potassium Content", key="K",
                       on_change=validate_input, args=("K", 0, 200, "Potassium (K)"))



        st.text_input("pH Value", placeholder="Enter pH Value", key="ph",
                      on_change=validate_input, args=("ph", 0, 14, "pH Value"))


        # Ensure session state has a default value before calling text_input
        # Ensure 'temperature' key exists in session state before using it in text_input

        # Ensure session state has a default value before calling text_input
        #if "temperature" not in st.session_state:
         #   st.session_state["temperature"] = ""

        # Ensure NASA data exists and is valid
        # Convert to string



        # Form for submission only
        with st.form("prediction_form"):
            submit = st.form_submit_button("Find")

        if submit:
            # Check if any error exists
            if any(st.session_state.get(f"{key}_error", False) for key in
                   ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]):
                st.error("❌ Fix the errors before submitting!")
            else:
                try:
                    # Convert values to float
                    N = float(st.session_state.N)
                    P = float(st.session_state.P)
                    K = float(st.session_state.K)
                    temperature = float(st.session_state.temperature)
                    humidity = float(st.session_state.humidity)
                    ph = float(st.session_state.ph)
                    rainfall = float(st.session_state.rainfall)

                    # Make prediction only if no errors
                    st.session_state.predicted_crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
                    st.session_state.page = "Result"
                    st.rerun()

                except ValueError:
                    st.toast("❌ Please enter valid numeric values before submitting.", icon="⚠️")


def result_page():
        st.sidebar.title("Navigation")
        if st.sidebar.button("Predict Again"):
            st.session_state.page = "Prediction"
            st.rerun()
        predicted_crop = st.session_state.predicted_crop
        page_bg_img = '''
               <style>
                   .stApp {
                       background: url("https://i.imgur.com/UjFDgWG.jpeg") no-repeat center center fixed;
                       background-size: cover;
                   }
               </style>
               '''
        st.markdown(page_bg_img, unsafe_allow_html=True)


        if predicted_crop:
            predicted_crop = predicted_crop.capitalize()
            st.subheader(f"Predicted Crop: {predicted_crop}")
            image_path = os.path.join(IMAGE_FOLDER, f"{predicted_crop}.jpg")
            if os.path.exists(image_path):
                st.image(image_path, caption=f"{predicted_crop} Field",use_container_width=True,width=10000 )
            else:
                st.warning(f"Image not found for {predicted_crop}. Check the folder.")
        else:
            st.warning("No prediction available. Please go back to the Prediction page.")

def about_page():
        st.sidebar.title("Navigation")
        if st.sidebar.button("Go to Home"):
            st.session_state.page = "Home"
            st.rerun()
        page_bg_img = '''
        <style>
            .stApp {
                background: url("https://www.shutterstock.com/image-photo/golden-sunlight-bathes-vibrant-soybean-600nw-2464146967.jpg") no-repeat center center fixed;
                background-size: cover;
            }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

        def get_base64(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()

        image_path = "aboutsection.jpg"
        image_base64 = get_base64(image_path)

        about_section_css = f'''
        <style>
            .about-section {{
                background-image: url("data:image/jpeg;base64,{image_base64}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                background-repeat: no-repeat;
                padding: 40px;
                border-radius: 15px;
                color: white;
                font-size: 20px;
                text-align: center;
            }}
        </style>
        '''

        st.markdown(about_section_css, unsafe_allow_html=True)
        st.markdown(
            '<div class="about-section"><h2>About This App</h2><p>This is a  Crop Recommendation App   that helps farmers choose the best crops based on soil conditions.</p><p>Where there is a farm, There is a Charm</p><p>Crown with Passion, Nurtured with Care</p><p>Thank You!</p></div>',
            unsafe_allow_html=True
        )


if __name__ == '__main__':
    main()
