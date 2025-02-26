import streamlit as st
import joblib
import pandas as pd

# Load trained models and scaler
model_paths = {
    "ElasticNet": "ElasticNet_best_model.joblib",
    "ExtraTrees": "ExtraTrees_best_model.joblib",
    "Stacking": "stacking_regressor_model.joblib",
    "Voting": "voting_regressor_model.joblib"
}

models = {}

try:
    scaler = joblib.load('robust_scaler.joblib')
    for model_name, path in model_paths.items():
        models[model_name] = joblib.load(path)
except FileNotFoundError as e:
    st.error(f"Error: {e}. Please ensure all model files exist.")
    st.stop()

# Streamlit page configuration
st.set_page_config(page_title='Regression For Axial Capacity for Driven piles', layout='wide')

# Title
st.markdown(
    "<h1 style='text-align: center;'>⚡ Regression Model Prediction App ⚡</h1>",
    unsafe_allow_html=True
)

# Get the original feature names from the scaler
original_feature_names = scaler.feature_names_in_

# Input section
col1, col2, col3 = st.columns([1, 2, 1])  # Center the form in the middle column

with col2:
    st.subheader('🔍 Input Features')
    input_features = []
    
    # Default values for input fields
    default_values = [
        2, 54.2, 12.03060833, 10.48590249, 36.34611045, 1.642185888, 0.994194682, 19.60371,
        1.279663294, 0.99580021, 21.12871765, 0.900056959, 0.995517238, 17.5228458, 0.83901715, 
        0.92709901, 48.01406613, 1.209547571, 0.852111046, 29.8722698, 1.011081313, 0.77697519, 
        60.94776343, 1.740575176, 0.83116449, 12.66254689, 0.43055865, 0.890173194, 10.90767144, 
        0.382355788, 0.981534565, 12.15775307, 0.3276072, 0.95374465
    ]

    for i, col in enumerate(original_feature_names):
        input_features.append(st.number_input(col, min_value=0.0, max_value=100.0, value=float(default_values[i])))

    input_df = pd.DataFrame([input_features], columns=original_feature_names)

    # Scale the input features
    input_scaled = scaler.transform(input_df)

    # Make predictions
    if st.button('Predict'):
        predictions = {}
        for model_name, model in models.items():
            predictions[model_name] = model.predict(input_scaled)[0]

        # Display Results
        st.subheader('📊 Prediction Results')
        for model_name, pred in predictions.items():
            st.write(f"⚡ **{model_name} Model Prediction:** {pred:.2f}")
