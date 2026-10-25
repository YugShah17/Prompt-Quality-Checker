"""
Model Training Script for Intelligent Prompt Quality Assessment Framework.
Trains a Scikit-Learn Random Forest Regressor on prompt linguistic features.

Workflow:
  1. Load dataset (data/prompts.csv)
  2. Extract explainable features from prompt text
  3. Split into 80% training and 20% testing sets
  4. Train RandomForestRegressor
  5. Evaluate real MAE and R² metrics
  6. Serialize model to model/prompt_quality_model.pkl
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.feature_extraction import get_feature_vector, FEATURE_NAMES

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "prompts.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_FILE = os.path.join(MODEL_DIR, "prompt_quality_model.pkl")

def train():
    print("=" * 65)
    print(" Intelligent Prompt Quality Assessment Framework - ML Training")
    print("=" * 65)

    # 1. Load dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f" Loaded dataset: {DATASET_PATH}")
    print(f" Total samples: {len(df)}")

    # 2. Extract features
    print("\n Extracting linguistic & structural features...")
    domains = df["domain"].tolist() if "domain" in df.columns else ["General"] * len(df)
    X = np.array([get_feature_vector(str(p), domain=str(d)) for p, d in zip(df["prompt"], domains)])
    y = df[["clarity", "specificity", "completeness", "relevance", "quality_score"]].values
    print(f" Feature matrix shape: {X.shape} ({len(FEATURE_NAMES)} features)")
    print(f" Target matrix shape:  {y.shape} (4 dimensions + overall score)")

    # 3. Train / Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f" Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 4. Train Random Forest Regressor
    print("\n Training Random Forest Regressor (n_estimators=100)...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    print(" Training completed successfully.")

    # 5. Evaluate real MAE and R²
    y_pred = model.predict(X_test)
    dim_names = ["Clarity", "Specificity", "Completeness", "Relevance", "Quality Score"]
    print("\n Evaluation Metrics on Test Set (20% held-out):")
    print("-" * 65)
    print(f" {'Dimension':<18} | {'MAE':<12} | {'R² Score':<12}")
    print("-" * 65)

    metrics = {}
    for i, name in enumerate(dim_names):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        metrics[name] = {"MAE": round(float(mae), 3), "R2": round(float(r2), 4)}
        print(f" {name:<18} | {mae:<12.3f} | {r2:<12.4f}")

    print("-" * 65)
    overall_mae = metrics["Quality Score"]["MAE"]
    overall_r2 = metrics["Quality Score"]["R2"]
    print(f" Overall Quality Score Model Performance: MAE = {overall_mae:.3f}, R² = {overall_r2:.4f}")

    # 6. Save model artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
        "model_type": "RandomForestRegressor"
    }
    joblib.dump(model_payload, MODEL_FILE)
    print(f"\n Saved model to: {MODEL_FILE}")
    print("=" * 65)
    return model_payload

if __name__ == "__main__":
    train()
