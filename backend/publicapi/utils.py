import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

def get_data():
    df = pd.read_csv("Smart_Home_Fire_Risk_Balanced.csv")
    leakage_cols = [
        "Fire_Risk",
        "Fire_Risk_Label",
        "Alert_Triggered",
        "Fire_Sensor",
        "Sprinkler_Status",
    ]
    X = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    y = df["Fire_Risk"]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # Recreate the KNN pipeline
    knn_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", KNeighborsClassifier(n_neighbors=25, weights="distance")),
        ]
    )

    # Fit the model
    knn_pipe.fit(X, y)
    return knn_pipe