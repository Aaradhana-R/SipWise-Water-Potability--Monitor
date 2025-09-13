import joblib

# Load the trained model
model = joblib.load("svm.pkl")
print("✅ Model classes:", model.classes_)

# ------------------------
# Test examples
# ------------------------

test_cases = {
    "SAFE Example": [7.0, 120, 1000, 2, 200, 800, 1.5, 50, 2],
    "UNSAFE Example 1": [3.5, 500, 50000, 20, 10, 1000, 30, 150, 15],
    "UNSAFE Example 2": [8.5, 400, 20000, 15, 500, 1200, 25, 200, 10]
}

# Run predictions
for name, values in test_cases.items():
    prediction_val = model.predict([values])[0]
    # Map to Safe/Unsafe
    if prediction_val == 0:
        prediction_label = "Safe"
    elif prediction_val == 1:
        prediction_label = "Unsafe"
    else:
        prediction_label = str(prediction_val)
    
    print(f"🔹 {name}: {prediction_label} (raw value: {prediction_val})")
