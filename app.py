import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load and prepare data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Streamlit app
st.title("Breast Cancer Prediction App")
st.write("""
This app predicts whether a breast tumor is **malignant** or **benign** based on input features.
""")

# Input features from the user
st.header("Enter Tumor Features")
features = {}
for feature_name in data.feature_names:
    features[feature_name] = st.number_input(feature_name, value=float(df[feature_name].mean()))

# Prediction
if st.button("Predict"):
    # Convert input features to array
    input_features = np.array([list(features.values())]).reshape(1, -1)
    input_features_scaled = scaler.transform(input_features)
    prediction = model.predict(input_features_scaled)[0]
    prediction_label = "Benign" if prediction == 1 else "Malignant"

    st.subheader(f"The tumor is predicted to be: **{prediction_label}**")

# Add footer
st.write("---")
st.write("Developed by Umesh Kumar Rai")
