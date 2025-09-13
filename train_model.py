# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("archive (3)/Watera.csv")
print("Dataset loaded successfully!")
print(df.head())
print("Shape:", df.shape)

# 1.1️⃣ Check class balance
print("Potability value counts (original dataset):")
print(df["potability"].value_counts())

# 2️⃣ Fill missing values
df.fillna(df.mean(), inplace=True)

# -----------------------------
# Function to train SVM
def train_svm(X, y, description="dataset"):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Train
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy on {description}: {acc:.2f}")
    return model, scaler

# -----------------------------
# 3️⃣ Train on original imbalanced dataset
X_orig = df.drop("potability", axis=1)
y_orig = df["potability"]
model_orig, scaler_orig = train_svm(X_orig, y_orig, description="original imbalanced dataset")

# 4️⃣ Upsample minority for balanced dataset
df_majority = df[df.potability == 0]
df_minority = df[df.potability == 1]
df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)
df_balanced = pd.concat([df_majority, df_minority_upsampled])
print("Potability value counts (after upsampling):")
print(df_balanced["potability"].value_counts())

# 5️⃣ Train on balanced dataset
X_bal = df_balanced.drop("potability", axis=1)
y_bal = df_balanced["potability"]
model_bal, scaler_bal = train_svm(X_bal, y_bal, description="balanced dataset")

# 6️⃣ Save both models and scalers
joblib.dump(model_orig, "svm_orig.pkl")
joblib.dump(scaler_orig, "scaler_orig.pkl")
joblib.dump(model_bal, "svm_balanced.pkl")
joblib.dump(scaler_bal, "scaler_balanced.pkl")
print("Models and scalers saved successfully!")
