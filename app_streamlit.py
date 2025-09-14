import streamlit as st
import joblib
import os
import numpy as np

# -----------------------------
# 1️⃣ Load model and scaler
def load_model_and_scaler():
    """
    Load trained SVM model and scaler from common locations.
    """
    base_dir = os.path.dirname(__file__)
    
    # Possible paths
    model_paths = [
        os.path.join(base_dir, "svm.pkl"),
        os.path.join(base_dir, "models", "svm.pkl")
    ]
    scaler_paths = [
        os.path.join(base_dir, "scaler.pkl"),
        os.path.join(base_dir, "models", "scaler.pkl")
    ]
    
    # Find existing files
    model_path = next((p for p in model_paths if os.path.exists(p)), None)
    scaler_path = next((p for p in scaler_paths if os.path.exists(p)), None)
    
    if not model_path:
        st.error(f"❌ Model file not found. Checked: {model_paths}")
        st.stop()
    if not scaler_path:
        st.error(f"❌ Scaler file not found. Checked: {scaler_paths}")
        st.stop()
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        st.success("✅ Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model/scaler: {e}")
        st.stop()

# -----------------------------
# 2️⃣ Map model classes
def get_class_mapping(model):
    try:
        return {model.classes_[0]: "UnSafe", model.classes_[1]: "Safe"}
    except Exception as e:
        st.error(f"❌ Error mapping classes: {e}")
        st.stop()

# -----------------------------
# 3️⃣ Predict water quality
def predict_water_quality(model, scaler, input_values, class_mapping):
    try:
        # Apply scaler to new input
        scaled_input = scaler.transform([input_values])
        prediction_val = model.predict(scaled_input)[0]
        prediction = class_mapping[prediction_val]
        return prediction, scaled_input.tolist(), prediction_val
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None, None, None

# -----------------------------
# 4️⃣ Streamlit UI
def main():
    st.set_page_config(page_title="💧 Water Quality Prediction", page_icon="💧")
    st.title("💧 Water Quality Prediction")
    st.write("Enter the water parameters below to check if it's **Safe or UnSafe**.")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    class_mapping = get_class_mapping(model)
    
    # Features
    input_features = [
        "ph", "hardness", "solids", "chloramines", "sulfate",
        "conductivity", "organicCarbon", "trihalomethanes", "turbidity"
    ]
    
    # Collect user input
    user_inputs = {}
    for feat in input_features:
        user_inputs[feat] = st.number_input(f"Enter {feat}", step=0.01)
    
    # Predict button
    if st.button("🔍 Predict"):
        input_values = [float(user_inputs[feat]) for feat in input_features]
        prediction, scaled_input, prediction_val = predict_water_quality(
            model, scaler, input_values, class_mapping
        )
        if prediction:
            # Highlight prediction
            if prediction == "Safe":
                st.success(f"### ✅ Prediction: Water is **{prediction}**")
            else:
                st.error(f"### ⚠️ Prediction: Water is **{prediction}**")
            
            # Debug info
            st.write("🔎 Raw input:", input_values)
            st.write("🔎 Scaled input:", scaled_input)
            st.write("🔎 Model output:", prediction_val)

# -----------------------------
# Run the app
if __name__ == "__main__":
    main()
