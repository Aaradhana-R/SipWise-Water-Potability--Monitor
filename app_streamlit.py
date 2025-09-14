import streamlit as st
import pickle
import numpy as np
import os

# -----------------------
# Load your trained SVM model safely
# -----------------------
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "svm.pkl")  # Updated filename

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ svm.pkl not found! Please make sure it is in the same folder as this script.")

# -----------------------
# Streamlit App
# -----------------------
st.set_page_config(page_title="SipWise Water Potability Monitor", page_icon="💧")

st.title("💧 SipWise Water Potability Monitor")
st.write("Enter the water parameters below to check if the water is safe or unsafe:")

# -----------------------
# Input Parameters
# -----------------------
ph = st.number_input("pH value", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
solids = st.number_input("Solids", min_value=0.0, value=200.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)
conductivity = st.number_input("Conductivity", min_value=0.0, value=500.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=80.0)
turbidity = st.number_input("Turbidity", min_value=0.0, value=3.0)

# -----------------------
# Prediction Button
# -----------------------
if st.button("Check Potability"):
    if 'model' in locals():
        input_features = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity,
                                    organic_carbon, trihalomethanes, turbidity]])
        
        prediction = model.predict(input_features)
        
        if prediction[0] == 1:
            st.success("✅ Water is Safe to Drink!")
        else:
            st.error("⚠️ Water is Unsafe!")
    else:
        st.warning("⚠️ Model not loaded. Cannot make predictions.")

# -----------------------
# Optional: Show raw inputs
# -----------------------
if st.checkbox("Show Input Parameters"):
    st.write({
        "pH": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic Carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    })
