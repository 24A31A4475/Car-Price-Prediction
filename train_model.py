import json
import logging
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# =============================================================================
# 1. LOGGING & CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_PATH = "car data.csv"
MODEL_OUTPUT_PATH = "car_model.pkl"
METADATA_OUTPUT_PATH = "model_metadata.json"
CURRENT_YEAR = 2026
RANDOM_SEED = 42


# =============================================================================
# 2. DATA PIPELINE & TRAINING EXECUTION
# =============================================================================
def train_and_export():
    # Step 1: Data Ingestion
    if not os.path.exists(DATA_PATH):
        logger.error(
            f"Dataset not found at path '{DATA_PATH}'. Please verify location."
        )
        raise FileNotFoundError(f"File '{DATA_PATH}' does not exist.")

    logger.info(f"Loading raw dataset from '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Dataset successfully loaded. Shape: {df.shape}")

    # Step 2: Feature Engineering & Preprocessing
    if "Year" in df.columns:
        logger.info(
            f"Engineering 'Car_Age' feature using context year {CURRENT_YEAR}..."
        )
        df["Car_Age"] = CURRENT_YEAR - df["Year"]
        df.drop(columns=["Year"], inplace=True)

    # Define Target and Feature Set
    target_col = "Selling_Price"
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' missing from input dataset."
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop identifier columns not used in modeling
    if "Car_Name" in X.columns:
        X = X.drop(columns=["Car_Name"])

    logger.info(f"Features set configured: {list(X.columns)}")

    # Step 3: Pipeline Transformers Setup
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # Numerical Transformer Pipeline (Imputation + Passthrough)
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    # Categorical Transformer Pipeline (Imputation + One-Hot Encoding)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Step 4: Model Pipeline Definition
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=3,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Step 5: Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    logger.info(
        f"Data split completed. Training samples: {len(X_train)}, Testing samples: {len(X_test)}"
    )

    # Step 6: Cross-Validation & Model Training
    logger.info("Performing 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(
        model_pipeline, X_train, y_train, cv=5, scoring="r2"
    )
    logger.info(
        f"Mean Cross-Validation R² Score: {np.mean(cv_scores):.4f} (± {np.std(cv_scores):.4f})"
    )

    logger.info("Fitting complete model pipeline on training set...")
    model_pipeline.fit(X_train, y_train)

    # Step 7: Out-of-Sample Evaluation
    y_pred = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logger.info("================ MODEL EVALUATION SUMMARY ================")
    logger.info(f"Mean Absolute Error (MAE):    ₹{mae:.2f} Lakhs")
    logger.info(f"Root Mean Squared Error (RMSE): ₹{rmse:.2f} Lakhs")
    logger.info(f"R-squared (R²) Score:          {r2:.4f} ({r2*100:.2f}%)")
    logger.info("==========================================================")

    # Step 8: Export Model Artifact & Metadata
    logger.info(f"Exporting trained model pipeline to '{MODEL_OUTPUT_PATH}'...")
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model_pipeline, f)

    # Save metadata summary
    metadata = {
        "model_type": "RandomForestRegressor",
        "trained_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": list(X.columns),
        "target": target_col,
        "metrics": {
            "r2_score": round(float(r2), 4),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "cv_r2_mean": round(float(np.mean(cv_scores)), 4),
        },
    }

    with open(METADATA_OUTPUT_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved metadata record to '{METADATA_OUTPUT_PATH}'.")
    logger.info("Training pipeline finished successfully.")


if __name__ == "__main__":
    train_and_export()