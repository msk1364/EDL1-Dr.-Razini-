import os
#import sklearn
import streamlit as st
import pandas as pd
import pickle
import requests
#from sklearn.preprocessing import StandardScaler

st.title('Economic Distance Level (EDL) Prediction')

model_source = st.selectbox("Select Model Source:", ["GitHub", "Local Disk"])

# URL های خام مدل‌ها برای هر نوع مصرف
github_model_urls = {
    "Industrial": [
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Industrial/DecisionTree.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Industrial/RandomForest.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Industrial/GaussianProcess.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Industrial/SVR.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Industrial/XGBoost.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Industrial/scaler.pkl"
    ],
    "Residential": [
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Residential/DecisionTree.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Residential/RandomForest.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Residential/GaussianProcess.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Residential/SVR.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Residential/XGBoost.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Residential/scaler.pkl"
    ],
    "Semi-Industrial": [
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Semi-Industrial/DecisionTree.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Semi-Industrial/RandomForest.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Semi-Industrial/GaussianProcess.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Semi-Industrial/SVR.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Semi-Industrial/XGBoost.pkl",
        "https://github.com/msk1364/EDL1-Dr.-Razini-/raw/refs/heads/main/models/Semi-Industrial/scaler.pkl"
    ]
}

models = {}

if model_source == "GitHub":
    @st.cache_resource
    def load_models_from_github(model_urls):
        models = {}
        for consumption_type, urls in model_urls.items():
            models[consumption_type] = {}
            for url in urls:
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    model_name = url.split("/")[-1][:-4]  # Extract the model name from the URL
                    model = pickle.loads(response.content)
                    models[consumption_type][model_name] = model
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching model from GitHub: {e}")
                except Exception as e:
                    st.error(f"Error loading model from {url}: {e}")
        return models

    models = load_models_from_github(github_model_urls)

elif model_source == "Local Disk":
    MODELS_DIR = "models"
    for consumption_type in os.listdir(MODELS_DIR):
        consumption_path = os.path.join(MODELS_DIR, consumption_type)
        if os.path.isdir(consumption_path):
            models[consumption_type] = {}
            for filename in os.listdir(consumption_path):
                if filename.endswith(".pkl"):
                    model_name = filename[:-4]
                    filepath = os.path.join(consumption_path, filename)
                    try:
                        with open(filepath, 'rb') as file:
                            models[consumption_type][model_name] = pickle.load(file)
                    except Exception as e:
                        st.error(f"Error loading model {filename} from consumption type {consumption_type}: {e}")
                        
            # Load the scaler for the consumption type
            scaler_path = os.path.join(consumption_path, "scaler.pkl")
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as scaler_file:
                        scaler = pickle.load(scaler_file)
                        models[consumption_type]['scaler'] = scaler
                except Exception as e:
                    st.error(f"Error loading scaler for consumption type {consumption_type}: {e}")

if models:
    consumption_type = st.selectbox('Select Load Type:', list(models.keys()))
    if consumption_type:
        model_names = list(models[consumption_type].keys())
        model_names_with_all = ["All Models"] + model_names
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

            # Check if there's a scaler for the selected consumption type and apply it
            if 'scaler' in models[consumption_type]:
                scaler = models[consumption_type]['scaler']
                input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
            else:
                input_data_scaled = input_data

            if selected_model_option == "All Models":
                for model_name, model in models[consumption_type].items():
                    if model_name != 'scaler':  # Skip scaler
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
    st.warning("No models available to load. Please select model source correctly and/or place the model files in the correct directory.")
