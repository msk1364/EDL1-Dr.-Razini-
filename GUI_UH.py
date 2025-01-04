# import streamlit as st
# import pandas as pd
# import pickle
# import os
# import requests
# import io
# import zipfile
# from sklearn.preprocessing import StandardScaler

# st.title('Economic Distance Level (EDL) Prediction')

# model_source = st.selectbox("Select Model Source:", ["GitHub", "Local Disk"])

# models = {}

# if model_source == "GitHub":
#     @st.cache_resource
#     def load_models_from_github(repo_url):
#         try:
#             zip_url = repo_url.replace("github.com", "api.github.com/repos").replace("/tree/main", "/zipball/main")
#             response = requests.get(zip_url, stream=True)
#             response.raise_for_status()

#             with zipfile.ZipFile(io.BytesIO(response.content)) as z:
#                 for file_info in z.infolist():
#                     if file_info.filename.endswith(".pkl"):
#                         with z.open(file_info) as f:
#                             try:
#                                 model = pickle.load(f)
#                                 parts = file_info.filename.split('/')
#                                 if len(parts) >= 3:
#                                     consumption_type = parts[-2]
#                                     model_name = parts[-1][:-4]
#                                     if consumption_type not in models:
#                                         models[consumption_type] = {}
#                                     models[consumption_type][model_name] = model
#                             except Exception as e:
#                                 st.error(f"Error loading model {file_info.filename}: {e}")
#             return models
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error fetching file from GitHub: {e}")
#             return None
#         except zipfile.BadZipFile as e:
#             st.error(f"Corrupted ZIP file: {e}")
#             return None
#         except Exception as e:
#             st.error(f"Unknown Error: {e}")
#             return None

#     repo_url = st.text_input("Enter GitHub Repository URL:", "https://github.com/username/your_repo/tree/main")
#     if repo_url:
#         models = load_models_from_github(repo_url)

# elif model_source == "Local Disk":
#     MODELS_DIR = "models"
#     for consumption_type in os.listdir(MODELS_DIR):
#         consumption_path = os.path.join(MODELS_DIR, consumption_type)
#         if os.path.isdir(consumption_path):
#             models[consumption_type] = {}
#             for filename in os.listdir(consumption_path):
#                 if filename.endswith(".pkl"):
#                     model_name = filename[:-4]
#                     filepath = os.path.join(consumption_path, filename)
#                     try:
#                         with open(filepath, 'rb') as file:
#                             models[consumption_type][model_name] = pickle.load(file)
#                     except Exception as e:
#                         st.error(f"Error loading model {filename} from consumption type {consumption_type}: {e}")
                        
#             # Load the scaler for the consumption type
#             scaler_path = os.path.join(consumption_path, "scaler.pkl")
#             if os.path.exists(scaler_path):
#                 try:
#                     with open(scaler_path, 'rb') as scaler_file:
#                         scaler = pickle.load(scaler_file)
#                         models[consumption_type]['scaler'] = scaler
#                 except Exception as e:
#                     st.error(f"Error loading scaler for consumption type {consumption_type}: {e}")

# if models:
#     consumption_type = st.selectbox('Select Consumption Type:', list(models.keys()))
#     if consumption_type:
#         model_names = list(models[consumption_type].keys())
#         model_names_with_all = ["All Models"] + model_names
#         selected_model_option = st.selectbox('Select Model:', model_names_with_all)

#         features = {}
#         features['Sensitivity/ExpectedInflationRate (%)'] = st.slider('Sensitivity/ExpectedInflationRate (%)', 0.0, 100.0, 5.0)
#         features['Sensitivity/Solar Scaled Average (kWh/m²/day)'] = st.slider('Sensitivity/Solar Scaled Average (kWh/m²/day)', 0.0, 10.0, 5.0)
#         features['Sensitivity/Grid Capital Cost ($/km)'] = st.slider('Sensitivity/Grid Capital Cost ($/km)', 0, 100000, 50000)
#         features['Sensitivity/Grid Power Price ($/kWh)'] = st.slider('Sensitivity/Grid Power Price ($/kWh)', 0.0, 1.0, 0.5)
#         features['Sensitivity/CO2 Penalty cost'] = st.slider('Sensitivity/CO2 Penalty cost', 0.0, 1.0, 0.5)
#         features['cost solar(kw/$)'] = st.slider('cost solar(kw/$)', 0.0, 10000.0, 5000.0)
#         features['cost wind(kw/$)'] = st.slider('cost wind(kw/$)', 0.0, 10000.0, 5000.0)
#         features['cost batry-nikel(kw/$)'] = st.slider('cost batry-nikel(kw/$)', 0.0, 10000.0, 5000.0)
#         features['cost convertor(kw/$)'] = st.slider('cost convertor(kw/$)', 0.0, 10000.0, 5000.0)
#         features['cost disel(kw/$)'] = st.slider('cost disel(kw/$)', 0.0, 10000.0, 5000.0)

#         if st.button('Predict'):
#             input_data = pd.DataFrame([list(features.values())], columns=list(features.keys()))

#             # Check if there's a scaler for the selected consumption type and apply it
#             if 'scaler' in models[consumption_type]:
#                 scaler = models[consumption_type]['scaler']
#                 input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
#             else:
#                 input_data_scaled = input_data

#             if selected_model_option == "All Models":
#                 for model_name, model in models[consumption_type].items():
#                     if model_name != 'scaler':  # Skip scaler
#                         try:
#                             prediction = model.predict(input_data_scaled)
#                             st.write(f'**Prediction with {model_name}:** {prediction[0]}')
#                         except Exception as e:
#                             st.error(f"Error during prediction with {model_name}: {e}")
#             else:
#                 selected_model = models[consumption_type][selected_model_option]
#                 try:
#                     prediction = selected_model.predict(input_data_scaled)
#                     st.write(f'**Prediction with {selected_model_option}:** {prediction[0]}')
#                 except Exception as e:
#                     st.error(f"Error during prediction: {e}")
#     else:
#         st.warning("Select a consumption type.")

# else:
#     st.warning("No models available to load. Please select model source correctly and/or place the model files in the correct directory.")



import streamlit as st
import pandas as pd
import pickle
import os
import requests
import io
import zipfile
from sklearn.preprocessing import StandardScaler

st.title('Economic Distance Level (EDL) Prediction')

model_source = st.selectbox("Select Model Source:", ["GitHub", "Local Disk"])

models = {}

if model_source == "GitHub":
    @st.cache_resource
    def load_models_from_github(repo_url):
        try:
            zip_url = repo_url.replace("github.com", "api.github.com/repos").replace("/tree/main", "/zipball/main")
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for file_info in z.infolist():
                    if file_info.filename.endswith(".pkl"):
                        with z.open(file_info) as f:
                            try:
                                model = pickle.load(f)
                                parts = file_info.filename.split('/')
                                if len(parts) >= 3:
                                    consumption_type = parts[-2]
                                    model_name = parts[-1][:-4]
                                    if consumption_type not in models:
                                        models[consumption_type] = {}
                                    models[consumption_type][model_name] = model
                            except Exception as e:
                                st.error(f"Error loading model {file_info.filename}: {e}")
            return models
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching file from GitHub: {e}")
            return None
        except zipfile.BadZipFile as e:
            st.error(f"Corrupted ZIP file: {e}")
            return None
        except Exception as e:
            st.error(f"Unknown Error: {e}")
            return None

    repo_url = st.text_input("Enter GitHub Repository URL:", "https://github.com/username/your_repo/tree/main")
    if repo_url:
        models = load_models_from_github(repo_url)

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
        # features['cost wind(kw/$)'] = st.slider('cost wind(kw/$)', 0.0, 10000.0, 5000.0)
        # features['cost batry-nikel(kw/$)'] = st.slider('cost batry-nikel(kw/$)', 0.0, 10000.0, 5000.0)
        # features['cost convertor(kw/$)'] = st.slider('cost convertor(kw/$)', 0.0, 10000.0, 5000.0)
        # features['cost disel(kw/$)'] = st.slider('cost disel(kw/$)', 0.0, 10000.0, 5000.0)

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
