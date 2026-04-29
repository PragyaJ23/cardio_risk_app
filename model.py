import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

FEATURE_COLS = ["Age", "Gender", "BMI", "SBP", "DBP", "HeartRate", "MedHistory"]
TARGET_COL   = "CardiacRisk"


def preprocess_and_split(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_pipeline(model_type: str = "Random Forest") -> Pipeline:
    estimator = (
        RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        if model_type == "Random Forest"
        else LogisticRegression(max_iter=1000, random_state=42)
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def get_feature_importance(pipeline: Pipeline, model_type: str) -> pd.Series:
    model = pipeline.named_steps["model"]
    if model_type == "Random Forest":
        scores = model.feature_importances_
    else:
        scores = np.abs(model.coef_[0])
    return pd.Series(scores, index=FEATURE_COLS).sort_values(ascending=False)


def get_shap_values(pipeline: Pipeline, X_test: pd.DataFrame, model_type: str):
    """Return SHAP values for the test set (first 200 rows for speed)."""
    imputed = pipeline.named_steps["imputer"].transform(X_test)
    scaled  = pipeline.named_steps["scaler"].transform(imputed)
    X_proc  = pd.DataFrame(scaled, columns=FEATURE_COLS)
    model   = pipeline.named_steps["model"]

    sample = X_proc.iloc[:200]
    if model_type == "Random Forest":
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(sample)
        # For binary classifiers, shap_values returns list; take class-1
        return (
            explainer,
            shap_vals[1] if isinstance(shap_vals, list) else shap_vals,
            sample,
        )
    else:
        explainer = shap.LinearExplainer(model, sample)
        shap_vals = explainer.shap_values(sample)
        return explainer, shap_vals, sample


def predict_patient(pipeline: Pipeline, patient_dict: dict) -> dict:
    df = pd.DataFrame([patient_dict], columns=FEATURE_COLS)
    prob     = pipeline.predict_proba(df)[0][1]
    label    = pipeline.predict(df)[0]
    if prob < 0.35:
        risk_level = "🟢 Low"
    elif prob < 0.65:
        risk_level = "🟡 Moderate"
    else:
        risk_level = "🔴 High"
    return {"probability": prob, "label": label, "risk_level": risk_level}
