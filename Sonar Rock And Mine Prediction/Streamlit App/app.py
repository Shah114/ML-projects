# Importing Modules
import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load('D:/Projects/SonarRockvsMinePrediction/model/sonar_model.pkl')

# Streamlit app
st.title("Sonar Rock/Mine Prediction")
st.write("Enter the feature values to predict if it's Rock or Mine.")

# Create input fields for each feature
features = []
for i in range(60):  # Assuming 60 features
    value = st.number_input(f"Feature {i+1}", min_value=0.0, max_value=1.0, step=0.01)
    features.append(value)

# Predict
if st.button("Predict"):
    prediction = model.predict([features])
    st.write(f"The model predicts: {'Rock' if prediction[0] == 0 else 'Mine'}")