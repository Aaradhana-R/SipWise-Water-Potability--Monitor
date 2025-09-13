from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model + scaler
model = joblib.load("svm.pkl")       # change if you want svm_balanced.pkl
scaler = joblib.load("scaler.pkl")   # make sure you have this file

# Print the classes to check order
print("✅ Model classes:", model.classes_)

# Create mapping dictionary dynamically
class_mapping = {
    model.classes_[1]: "Safe",
    model.classes_[0]: "UnSafe"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Get input values from form
            ph = float(request.form["ph"])
            hardness = float(request.form["hardness"])
            solids = float(request.form["solids"])
            chloramines = float(request.form["chloramines"])
            sulfate = float(request.form["sulfate"])
            conductivity = float(request.form["conductivity"])
            organicCarbon = float(request.form["organicCarbon"])
            trihalomethanes = float(request.form["trihalomethanes"])
            turbidity = float(request.form["turbidity"])

            input_values = [
                ph, hardness, solids, chloramines, sulfate,
                conductivity, organicCarbon, trihalomethanes, turbidity
            ]

            # Scale + predict
            scaled_input = scaler.transform([input_values])
            prediction_val = model.predict(scaled_input)[0]

            # Debugging
            print("🔎 Raw input:", input_values)
            print("🔎 Scaled input:", scaled_input)
            print("🔎 Model raw output:", prediction_val)

            # Use dynamic mapping
            prediction = class_mapping[prediction_val]

        except Exception as e:
            prediction = "Error in input values"
            print("❌ Error:", e)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
