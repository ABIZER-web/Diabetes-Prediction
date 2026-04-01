import streamlit as st
import pickle
import numpy as np

# Load the saved model from Jupyter
model = pickle.load(open('diabetes_model.pkl', 'rb'))

st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Risk Checker")

# User Input Fields
st.subheader("Enter Patient Details")
preg = st.slider("Pregnancies", 0, 17, 1)
glucose = st.number_input("Glucose Level", 0, 200, 100)
bp = st.number_input("Blood Pressure (mm Hg)", 0, 122, 70)
bmi = st.number_input("BMI (Body Mass Index)", 0.0, 67.0, 25.0)
age = st.slider("Age", 21, 81, 30)

# We need all 8 features the model was trained on
# Filling others with average/default values for simplicity
if st.button("Check Result"):
    # Input format: [Preg, Gluc, BP, SkinThick, Insulin, BMI, DPF, Age]
    # We use 20 for SkinThickness, 80 for Insulin, and 0.5 for Pedigree as defaults
    features = np.array([[preg, glucose, bp, 20, 80, bmi, 0.5, age]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("Result: Potential Risk Detected. Please consult a doctor.")
    else:
        st.success("Result: Low Risk Detected.")