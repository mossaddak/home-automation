import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("Smart_Home_Fire_Risk_Balanced.csv")

# Drop leakage columns (same as notebook)
leakage_cols = ["Fire_Risk", "Fire_Risk_Label", "Alert_Triggered", "Fire_Sensor", "Sprinkler_Status"]
X = df.drop(columns=[c for c in leakage_cols if c in df.columns])
y = df["Fire_Risk"]

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# -----------------------------
# 3. Preprocessing
# -----------------------------
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# -----------------------------
# 4. Model Pipeline
# -----------------------------
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=30,
        min_samples_leaf=15,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# -----------------------------
# 5. Prediction function
# -----------------------------
def predict_fire_risk(input_dict):
    """
    input_dict: dictionary of feature_name -> value
    """
    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).max()

    return {
        "Fire_Risk_Prediction": int(prediction),
        "Confidence": round(probability * 100, 2)
    }

# -----------------------------
# 6. Example input
# -----------------------------

risk_sample_input = {
    "Temperature_C": 27.922065,
    "Humidity_%": 40.607585,
    "CO2_ppm": 503.063651,
    "Sound_Level_dB": 36.664384,
    "Light_Intensity_lux": 308.783013,
    "Power_Consumption_W": 295.248463,

    "Motion_Detected": 1,
    "Day_Night": "Day",

    "AC_Status": 0,
    "Fan_Status": 0,
    "Smart_LED_On": 0,
    "Smart_Door_Status": "Closed",
    "Window_Status": "Closed",

    "Occupancy": 1,
    "Smoke_Level": 0.181935,
    "Room_Type": "Study",
}

safe_sample_input = {
    "Temperature_C": 30.243536,
    "Humidity_%": 41.595070,
    "CO2_ppm": 763.943867,
    "Sound_Level_dB": 27.506587,
    "Light_Intensity_lux": 345.170395,
    "Power_Consumption_W": 543.465613,

    "Motion_Detected": 0,
    "Day_Night": "Night",

    "AC_Status": 1,
    "Fan_Status": 1,
    "Smart_LED_On": 0,
    "Smart_Door_Status": "Closed",
    "Window_Status": "Open",

    "Occupancy": 1,
    "Smoke_Level": 0.134182,
    "Room_Type": "Store"
}



result = predict_fire_risk(risk_sample_input)
print(result)
