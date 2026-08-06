"""
churn_model.py

Mobile Gaming 7-Day Churn Prediction Model.

This script loads the Cookie Cats dataset, engineers churn features, trains a Random Forest
Classifier to predict 7-day player churn (churn_d7), evaluates performance metrics,
prints feature importances, and saves the trained model artifact to /models/churn_model.pkl.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def main():
    print("=" * 60)
    print("MOBILE GAMING CHURN PREDICTION MODEL TRAINING")
    print("=" * 60)

    # 1. Load Data with Error Handling
    data_path = os.path.join("data", "cookie_cats.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}.")
        print("Please place 'cookie_cats.csv' in the '/data' directory.")
        return

    print(f"\n1. Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    if df.empty:
        print("Error: Loaded dataset is empty.")
        return

    print(f"Initial row count: {len(df):,}")

    # 2. Target Creation (churn_d7: 1 if retention_7 is False, else 0)
    df['churn_d7'] = (~df['retention_7']).astype(int)

    # 3. Feature Engineering
    df['gate_45_flag'] = (df['version'] != 'gate_30').astype(int)
    df['retention_1_flag'] = df['retention_1'].astype(int)

    print("\n2. Target & Feature Summary:")
    print(df['churn_d7'].value_counts(normalize=True).rename({1: 'Churned (1)', 0: 'Retained (0)'}))

    # 4. Outlier Filtering (remove sum_gamerounds >= 5000)
    initial_count = len(df)
    df = df[df['sum_gamerounds'] < 5000].reset_index(drop=True)
    filtered_count = len(df)
    print(f"\n3. Filtering Outliers (sum_gamerounds >= 5000):")
    print(f"Removed {initial_count - filtered_count} outlier/bot rows. Remaining rows: {filtered_count:,}")

    if df.empty:
        print("Error: No data remaining after filtering.")
        return

    # Define Feature matrix (X) and Target vector (y)
    feature_cols = ['sum_gamerounds', 'gate_45_flag', 'retention_1_flag']
    X = df[feature_cols]
    y = df['churn_d7']

    # 5. Train / Test Split (80/20 with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\n4. Train/Test Split:")
    print(f"Training samples: {len(X_train):,}, Testing samples: {len(X_test):,}")

    # 6. Train RandomForestClassifier
    print("\n5. Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    print("Model training complete.")

    # 7. Model Evaluation
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "=" * 60)
    print("6. MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Retained (0)', 'Churned (1)']))

    # 8. Feature Importances
    print("=" * 60)
    print("7. FEATURE IMPORTANCES")
    print("=" * 60)
    importances = rf_model.feature_importances_
    for col, imp in zip(feature_cols, importances):
        print(f"Feature: {col:<20} | Importance: {imp:.4f} ({imp*100:.2f}%)")

    # 9. Save Model & Metrics Artifacts
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_filepath = os.path.join(models_dir, "churn_model.pkl")
    joblib.dump(rf_model, model_filepath)
    print(f"\n8. Saved trained model artifact to: {model_filepath}")

    metrics_data = {
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "feature_importances": {
            col: round(float(imp), 4) for col, imp in zip(feature_cols, importances)
        },
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    }
    metrics_filepath = os.path.join(models_dir, "metrics.json")
    with open(metrics_filepath, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"9. Saved model metrics JSON to: {metrics_filepath}")
    print("=" * 60)

if __name__ == "__main__":
    main()
