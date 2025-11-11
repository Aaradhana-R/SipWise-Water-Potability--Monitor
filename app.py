from flask import Flask, render_template, request
import joblib
import os

# Set template folder - USE CAPITAL T!
app = Flask(__name__, template_folder='Templates')

# Load model + scaler
try:
    model = joblib.load("svm.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Define class mapping
class_mapping = {
    model.classes_[1]: "Safe",
    model.classes_[0]: "UnSafe"
}

# HOME PAGE
@app.route('/')
def home():
    return render_template('index.html')

# PREDICTION PAGE
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    confidence = None
    
    if request.method == 'POST':
        try:
            ph = float(request.form.get("ph"))
            hardness = float(request.form.get("hardness"))
            solids = float(request.form.get("solids"))
            chloramines = float(request.form.get("chloramines"))
            sulfate = float(request.form.get("sulfate"))
            conductivity = float(request.form.get("conductivity"))
            organicCarbon = float(request.form.get("organicCarbon"))
            trihalomethanes = float(request.form.get("trihalomethanes"))
            turbidity = float(request.form.get("turbidity"))
            
            input_values = [ph, hardness, solids, chloramines, sulfate, 
                          conductivity, organicCarbon, trihalomethanes, turbidity]
            
            scaled_input = scaler.transform([input_values])
            prediction_val = model.predict(scaled_input)[0]
            prob = model.predict_proba(scaled_input)[0]
            confidence = max(prob) * 100
            
            prediction = class_mapping[prediction_val]
            
            print(f"✅ Prediction: {prediction} ({confidence:.2f}%)")
            
        except Exception as e:
            prediction = "Error"
            print(f"❌ Error: {e}")
    
    return render_template('result.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
