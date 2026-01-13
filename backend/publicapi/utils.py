import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", KNeighborsClassifier(n_neighbors=50, weights="uniform")),
        ]
    )

    model.fit(X_train, y_train)
    training_columns = X.columns.tolist()
    numeric_means = X_train.mean(numeric_only=True).to_dict()
    categorical_modes = X_train[cat_cols].mode().iloc[0].to_dict()

    return model, training_columns, numeric_means, categorical_modes
