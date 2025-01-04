import streamlit as st
import pandas as pd
import pickle
import requests
import os

st.set_page_config(layout="wide")
st.title('Economic Distance Level (EDL) Prediction')

def load_model_from_url(model_url):
    """Downloads and loads a model from the given URL."""
    try:
        response = requests.get(model_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        model = pickle.loads(response.content)
        return model
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading model from {model_url}: {e}")
        return None
    except pickle.UnpicklingError as e:
        st.error(f"Error unpickling model from {model_url}: {e}")
        return None

# **مهم:** آدرس‌های زیر را با آدرس‌های خام واقعی فایل‌های مدل خود در GitHub جایگزین کنید.
# ساختار زیر به شما امکان دسترسی به مدل ها و scaler ها را به صورت جداگانه می دهد
model_urls = {
    "Residential": {  # نام نوع مصرف
        "ANN_Residential": "https://raw.githubusercontent.com/msk1364/edl1-dr.-razini-/main/models/Residential/ANN_Residential.pkl",
        "scaler": "https://raw.githubusercontent.com/msk1364/edl1-dr.-razini-/main/models/Residential/scaler.pkl",
    },
    "Commercial": {  # نام نوع مصرف
        "ANN_Commercial": "https://raw.githubusercontent.com/msk1364/edl1-dr.-razini-/main/models/Commercial/ANN_Commercial.pkl",
        "scaler": "https://raw.githubusercontent.com/msk1364/edl1-dr.-razini-/main/models/Commercial/scaler.pkl",
    },
    "Industrial": {  # نام نوع مصرف
        "ANN_Industrial": "https://raw.githubusercontent.com/msk1364/edl1-dr.-razini-/main/models/Industrial/ANN_Industrial.pkl",
        "scaler": "https://raw.githubusercontent.com/msk1364/edl1-dr.-razini-/main/models/Industrial/scaler.pkl",
    },
    "Agricultural": {  # نام نوع مصرف
        "ANN_Agricultural": "https://raw.githubusercontent.com/msk1364/edl1-dr.-razini-/main/models/Agricultural/ANN_Agricultural.pkl",
        "scaler": "https://raw.githubusercontent.com/msk1364/edl1-dr.-razini-/main/models/Agricultural/scaler.pkl",
    },
    # ... سایر انواع مصرف و مدل‌ها
}

# Load models
models = {}
for consumption_type, model_data in model_urls.items():
    models[consumption_type] = {}
    for model_name, url in model_data.items():
        model = load_model_from_url(url)
        if model:
            models[consumption_type][model_name] = model
        else:
            st.error(f"Failed to load {model_name} for {consumption_type}")

if models:
    consumption_type = st.selectbox('Select Load Type:', list(models.keys()))

    if consumption_type:
        model_names = list(models[consumption_type].keys())
        # حذف scaler از لیست مدل ها برای نمایش
        model_names_without_scaler = [name for name in model_names if name != 'scaler']
        model_names_with_all = ["All Models"] + model_names_without_scaler
        selected_model_option = st.selectbox('Select Model:', model_names_with_all)

        features = {}
        features['Sensitivity/ExpectedInflationRate (%)'] = st.slider('Inflation Rate (%); Representing macro-economic factors ', 2, 20, 10, 1)
        features['Sensitivity/Solar Scaled Average (kWh/m²/day)'] = st.slider('Solar Radiation (Kwh/m^2); Representing climate factors ', 3, 6, 4, 1)
        features['Sensitivity/Grid Capital Cost ($/km)'] = st.slider('Grid Capital Cost ($/km); Representing grid extension costs factors', 300000, 1000000, 600000, 100000)
        features['Sensitivity/Grid Power Price ($/kWh)'] = st.slider('Grid Power Price ($/kWh); Representing Energy tariffs factors', 0.01, 0.30, 0.08, 0.01)
        features['Sensitivity/CO2 Penalty cost'] = st.slider('CO2 Penalty cost ($/ton CO2 eq.)', 0, 20, 10, 1)
        features['cost solar(kw/$)'] = st.slider('cost solar(kw/$); Representing equipment costs factors', 600, 1000, 750, 50)

        if st.button('Predict'):
            input_data = pd.DataFrame([list(features.values())], columns=list(features.keys()))

            if 'scaler' in models[consumption_type]:
                scaler = models[consumption_type]['scaler']
                input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
            else:
                input_data_scaled = input_data

            if selected_model_option == "All Models":
                for model_name, model in models[consumption_type].items():
                    if model_name != 'scaler':
                        try:
                            prediction = model.predict(input_data_scaled)
                            st.write(f'**Prediction with {model_name}:** {prediction[0]}')
                        except Exception as e:
                            st.error(f"Error during prediction with {model_name}: {e}")
            else:
                selected_model = models[consumption_type][selected_model_option]
                try:
                    prediction = selected_model.predict(input_data_scaled)
                    st.write(f'**Prediction with {selected_model_option}:** {prediction[0]}')
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    else:
        st.warning("Select a consumption type.")

else:
    st.warning("No models available. Please check your model URLs.")
