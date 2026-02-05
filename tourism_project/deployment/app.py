import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Common constants
HUGGINGFACE_USER_NAME = os.getenv('HUGGINGFACE_USER_NAME')
HUGGINGFACE_MODEL_NAME = os.getenv('HUGGINGFACE_MODEL_NAME')

# Download the model from the Model Hub
# Note: Ensure the filename matches what you uploaded in your training script
try:
    model_path = hf_hub_download(
        repo_id=f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_MODEL_NAME}",
        filename="model.joblib"
    )
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model from Hugging Face: {e}")
    st.stop()

# Streamlit UI Setup
st.set_page_config(page_title="Wellness Package Prediction", layout="centered")
st.title("Tourism Wellness Package Purchase Prediction")
st.write("""
This tool predicts whether a customer is likely to purchase a **Wellness Package** based on their demographic and interaction history.
""")

st.divider()

# Create two columns for a cleaner UI layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
    Designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
    MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=25000.0)
    CityTier = st.slider("City Tier", 1, 3, 1)

with col2:
    st.subheader("Travel Behavior")
    TypeofContact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
    ProductPitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=15)
    NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
    NumberOfTrips = st.number_input("Number of Trips", min_value=0, value=2)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    PreferredPropertyStar = st.slider("Preferred Property Star", 3, 5, 3)

st.subheader("Additional Info")
c3, c4, c5 = st.columns(3)
with c3:
    Passport = st.selectbox("Has Passport?", ["Yes", "No"])
with c4:
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
with c5:
    NumberOfPersonVisiting = st.number_input("Adults Visiting", min_value=1, value=2)
    NumberOfChildrenVisiting = st.number_input("Children Visiting", min_value=0, value=0)

# Prepare input data matching the exact training schema
input_dict = {
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}

input_data = pd.DataFrame([input_dict])

# Prediction Logic
classification_threshold = 0.45

st.divider()
if st.button("Generate Prediction", type="primary"):
    # Get probability from XGBoost
    prediction_proba = model.predict_proba(input_data)[0, 1]

    # Apply custom threshold
    prediction = 1 if prediction_proba >= classification_threshold else 0

    if prediction == 1:
        st.success(f"High Potential: Customer is likely to **PURCHASE** (Prob: {prediction_proba:.2f})")
    else:
        st.warning(f"Low Potential: Customer is likely to **NOT PURCHASE** (Prob: {prediction_proba:.2f})")
