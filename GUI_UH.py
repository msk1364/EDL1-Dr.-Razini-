# # -*- coding: utf-8 -*-
# """
# Created on Thu Jan  2 12:23:35 2025

# @author: Ms. Ketabi
# """

# import streamlit as st
# import pandas as pd
# import pickle
# import requests
# import io
# import zipfile
# from sklearn.preprocessing import StandardScaler

# st.title('Economic Distance Level (EDL) Prediction')

# # Only allow loading models from GitHub
# @st.cache_resource
# def load_models_from_github(repo_url):
#     models = {}
#     try:
#         zip_url = repo_url.replace("github.com", "api.github.com/repos").replace("/tree/main", "/zipball/main")
#         response = requests.get(zip_url, stream=True)
#         response.raise_for_status()

#         with zipfile.ZipFile(io.BytesIO(response.content)) as z:
#             for file_info in z.infolist():
#                 if file_info.filename.endswith(".pkl"):
#                     with z.open(file_info) as f:
#                         try:
#                             model = pickle.load(f)
#                             parts = file_info.filename.split('/')
#                             if len(parts) >= 3:
#                                 consumption_type = parts[-2]
#                                 model_name = parts[-1][:-4]
#                                 if consumption_type not in models:
#                                     models[consumption_type] = {}
#                                 models[consumption_type][model_name] = model
#                         except Exception as e:
#                             st.error(f"Error loading model {file_info.filename}: {e}")
#         return models
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error fetching file from GitHub: {e}")
#         return None
#     except zipfile.BadZipFile as e:
#         st.error(f"Corrupted ZIP file: {e}")
#         return None
#     except Exception as e:
#         st.error(f"Unknown Error: {e}")
#         return None

# repo_url = st.text_input("Enter GitHub Repository URL:", "https://github.com/username/your_repo/tree/main")
# if repo_url:
#     models = load_models_from_github(repo_url)
# else:
#     models = {}


import streamlit as st
import pandas as pd
import pickle
import requests
import io
import zipfile
import os

# ... (سایر importها)

@st.cache_resource
def load_models_from_github(repo_url):
    models = {}
    try:
        zip_url = repo_url.replace("github.com", "api.github.com/repos").replace("/tree/main", "/zipball/main")
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for file_info in z.infolist():
                parts = file_info.filename.split('/')
                if len(parts) >= 3 and parts[0] == "models":
                    consumption_type = parts[-2]
                    model_name = parts[-1][:-4]
                    try:
                        with z.open(file_info) as f:
                            model = pickle.load(f)
                            if consumption_type not in models:
                                models[consumption_type] = {}
                            models[consumption_type][model_name] = model
                    except Exception as e:
                        st.error(f"Error loading model {file_info.filename}: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching file from GitHub: {e}")
    except zipfile.BadZipFile as e:
        st.error(f"Corrupted ZIP file: {e}")
    except Exception as e:
        st.error(f"Unknown Error: {e}")
    return models

# repo_url = "https://github.com/your_username/your_repository/tree/main" # آدرس مخزن خود را اینجا وارد کنید
repo_url = "https://github.com/msk1364/edl1-dr.-razini-/tree/main"
models = load_models_from_github(repo_url)

# ... (بقیه کد Streamlit شما)


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

            if selected_model_option == "All Models":
                for model_name, model in models[consumption_type].items():
                    try:
                        prediction = model.predict(input_data)
                        st.write(f'**Prediction with {model_name}:** {prediction[0]}')
                    except Exception as e:
                        st.error(f"Error during prediction with {model_name}: {e}")
            else:
                selected_model = models[consumption_type][selected_model_option]
                try:
                    prediction = selected_model.predict(input_data)
                    st.write(f'**Prediction with {selected_model_option}:** {prediction[0]}')
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    else:
        st.warning("Select a consumption type.")

else:
    st.warning("No models available to load. Please check the GitHub repository URL.")
