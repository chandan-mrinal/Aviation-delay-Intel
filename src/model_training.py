import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from preprocessing import preprocess_data, chronological_split
import json
import os

def train_models():
    print("Loading and preprocessing data...")
    X, y, df_raw = preprocess_data("data/flights_2023_sample.csv")
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    
    # 1. Logistic Regression (Baseline)
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    
    # 2. XGBoost (Main Model)
    print("Training XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        "LogisticRegression": {
            "accuracy": accuracy_score(y_test, lr_preds),
            "precision": precision_score(y_test, lr_preds),
            "recall": recall_score(y_test, lr_preds),
            "f1": f1_score(y_test, lr_preds),
            "auc": roc_auc_score(y_test, lr_probs)
        },
        "XGBoost": {
            "accuracy": accuracy_score(y_test, xgb_preds),
            "precision": precision_score(y_test, xgb_preds),
            "recall": recall_score(y_test, xgb_preds),
            "f1": f1_score(y_test, xgb_preds),
            "auc": roc_auc_score(y_test, xgb_probs)
        }
    }
    
    print("\nModel Metrics:")
    print(json.dumps(metrics, indent=4))
    
    # Save results
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f)
        
    # Feature Importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance.to_csv("results/feature_importance.csv", index=False)
    print("\nTop 5 Features:")
    print(importance.head())
    
    # Save Model
    joblib.dump(xgb_model, "results/xgb_model.pkl")
    joblib.dump(X.columns.tolist(), "results/model_features.pkl")
    
    print("\nModels and metrics saved to results/")

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
    train_models()
