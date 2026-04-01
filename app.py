import streamlit as st
import pickle
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from math import radians, cos, sin, asin, sqrt

# --- 1. SETTINGS & MODEL ---
st.set_page_config(page_title="Mumbai Diabetes AI", page_icon="🩺", layout="wide")

@st.cache_resource
def load_model():
    # Ensure 'diabetes_model.pkl' is in your desktop/diabetes folder
    return pickle.load(open('diabetes_model.pkl', 'rb'))

model = load_model()

# --- 2. SESSION STATE INITIALIZATION ---
# This is the "memory" that stops the flickering
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
    st.session_state.result = None
    st.session_state.user_lat = 19.0760 # Default Mumbai
    st.session_state.user_lon = 72.8777

# --- 3. HELPER FUNCTIONS ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371 
    dLat, dLon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dLat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dLon/2)**2
    return R * 2 * asin(sqrt(a))

# --- 4. GEOLOCATION ---
# This runs once to catch your location
loc = get_geolocation()
if loc and 'coords' in loc:
    st.session_state.user_lat = loc['coords']['latitude']
    st.session_state.user_lon = loc['coords']['longitude']

# --- 5. CLINIC DATABASE ---
mumbai_clinics = [
    {"name": "Dr. Pranay Pawar Clinic", "lat": 19.0725, "lon": 72.8660},
    {"name": "Dr. N.I. Mirza (Kalina)", "lat": 19.0740, "lon": 72.8675},
    {"name": "Kripa Medical (Santacruz)", "lat": 19.0710, "lon": 72.8645},
    {"name": "SevenHills Hospital (Andheri)", "lat": 19.1245, "lon": 72.8755},
    {"name": "Lilavati Hospital (Bandra)", "lat": 19.0514, "lon": 72.8234},
    {"name": "Nanavati Max (Vile Parle)", "lat": 19.0968, "lon": 72.8431},
    {"name": "H. N. Reliance (Girgaon)", "lat": 18.9592, "lon": 72.8211},
    {"name": "Fortis Hospital (Mulund)", "lat": 19.1678, "lon": 72.9531},
    {"name": "Kokilaben (Versova)", "lat": 19.1311, "lon": 72.8255},
    {"name": "S. L. Raheja (Mahim)", "lat": 19.0396, "lon": 72.8438},
    {"name": "Mysha Clinic", "lat": 19.0745, "lon": 72.8640}
]

# --- 6. USER INTERFACE ---
st.title("🩺 Smart Diabetes Predictor & Mumbai Clinic Finder")
st.info(f"📍 Current GPS: {st.session_state.user_lat:.4f}, {st.session_state.user_lon:.4f}")

col1, col2 = st.columns(2)
with col1:
    preg = st.slider("Pregnancies", 0, 17, 1)
    glucose = st.number_input("Glucose Level", 0, 200, 100)
    bp = st.number_input("Blood Pressure", 0, 150, 70)
    skin = st.number_input("Skin Thickness", 0, 99, 20)
with col2:
    insulin = st.number_input("Insulin Level", 0, 846, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.4, 0.5)
    age = st.slider("Age", 21, 81, 30)

# Trigger Prediction
if st.button("Analyze Risk Now", use_container_width=True):
    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    st.session_state.result = model.predict(features)[0]
    st.session_state.prediction_done = True

# --- 7. STABLE DISPLAY SECTION ---
if st.session_state.prediction_done:
    st.divider()
    if st.session_state.result == 1:
        st.error("### ⚠️ HIGH RISK DETECTED")
        st.subheader("📍 Clinics within 5km of you")

        # Map logic
        m = folium.Map(location=[st.session_state.user_lat, st.session_state.user_lon], zoom_start=14)
        folium.Marker([st.session_state.user_lat, st.session_state.user_lon], 
                       popup="You", icon=folium.Icon(color="blue", icon="user")).add_to(m)

        found_nearby = False
        for clinic in mumbai_clinics:
            dist = haversine(st.session_state.user_lat, st.session_state.user_lon, clinic['lat'], clinic['lon'])
            if dist <= 5.0:
                found_nearby = True
                folium.Marker(
                    [clinic['lat'], clinic['lon']], 
                    popup=f"{clinic['name']} ({dist:.2f} km)",
                    icon=folium.Icon(color="red", icon="plus-sign")
                ).add_to(m)
        
        if not found_nearby:
            st.warning("No clinics found within 5km. Showing closest Mumbai hospitals.")
            for clinic in mumbai_clinics[:3]:
                folium.Marker([clinic['lat'], clinic['lon']], popup=clinic['name'], icon=folium.Icon(color="orange")).add_to(m)

        # IMPORTANT: Fixed key prevents flickering
        st_folium(m, width=1200, height=500, key="stable_mumbai_map")
    else:
        st.success("### ✅ LOW RISK DETECTED")
        st.balloons()