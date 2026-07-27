import streamlit as st
import pickle
import numpy as np

st.title('Salary Prediction App')

# ---- Load the trained model and scaler (with clear error messages) ----
try:
    with open('hr.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except ModuleNotFoundError as e:
    st.error(
        f"Could not load hr.pkl — a required package is missing: {e}. "
        "Make sure the library the model was trained with (e.g. xgboost, "
        "scikit-learn) is listed in requirements.txt."
    )
    st.stop()
except FileNotFoundError:
    st.error("hr.pkl not found. Make sure it's committed to the repo root.")
    st.stop()

try:
    with open('schr.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except ModuleNotFoundError as e:
    st.error(f"Could not load schr.pkl — a required package is missing: {e}.")
    st.stop()
except FileNotFoundError:
    st.error("schr.pkl not found. Make sure it's committed to the repo root.")
    st.stop()

# ---- Input fields ----
age = st.number_input('Age', min_value=0, max_value=120, value=30)
education = st.selectbox(
    'Education Level',
    ['High School or less', 'Intermediate', 'Graduation', 'PG']
)
experience_months = st.number_input(
    'Months of Experience', min_value=0, max_value=600, value=60
)  # Assuming max experience is 50 years

# Convert education to numeric encoding
education_mapping = {'High School or less': 0, 'Intermediate': 1, 'Graduation': 2, 'PG': 3}
education_encoded = education_mapping[education]

# Prepare the feature vector
features = np.array([[age, education_encoded, experience_months]], dtype=np.float64)

# ---- Predict ----
if st.button('Predict Salary'):
    try:
        features_scaled = scaler.transform(features)
        predicted_salary = model.predict(features_scaled)
        st.write(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
