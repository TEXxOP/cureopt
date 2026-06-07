"""
CureOpt AI — Prediction & Explainability Module
Loads trained models, runs inference, computes SHAP explanations.
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    XGBOOST_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH, ENSEMBLE_WEIGHTS_PATH,
    SAFETY_FACTOR, FEATURE_RANGES
)

FEATURE_COLS = [
    "cement_pct", "fly_ash_pct", "water_cement_ratio",
    "curing_method", "admixture_type", "ambient_temp_C",
    "humidity_pct", "curing_duration_hr", "automation_level"
]

# Lazy-loaded globals
_xgb_models = None
_rf_models = None
_scaler = None
_ensemble_weights = None


def _load_models():
    """Lazy load all model artifacts."""
    global _xgb_models, _rf_models, _scaler, _ensemble_weights
    if _xgb_models is None:
        _xgb_models = joblib.load(XGBOOST_MODEL_PATH)
        _rf_models = joblib.load(RF_MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _ensemble_weights = joblib.load(ENSEMBLE_WEIGHTS_PATH)


def _features_to_array(features_dict):
    """Convert features dict to numpy array in correct column order."""
    return np.array([[features_dict[col] for col in FEATURE_COLS]])


def predict_strength(features_dict):
    """
    Predict concrete strength at 8h, 16h, 24h.

    Args:
        features_dict: dict with keys matching FEATURE_COLS

    Returns:
        dict with strength_8hr_MPa, strength_16hr_MPa, strength_24hr_MPa
    """
    _load_models()
    X = _features_to_array(features_dict)
    X_scaled = _scaler.transform(X)

    w_xgb = _ensemble_weights["xgb"]
    w_rf = _ensemble_weights["rf"]

    results = {}
    target_names = ["strength_8hr_MPa", "strength_16hr_MPa", "strength_24hr_MPa"]
    for t, name in enumerate(target_names):
        pred_xgb = _xgb_models[t].predict(X_scaled)[0]
        pred_rf = _rf_models[t].predict(X_scaled)[0]
        results[name] = float(w_xgb * pred_xgb + w_rf * pred_rf)

    return results


def predict_strength_curve(features_dict, hours=None):
    """
    Predict strength at multiple time points via interpolation.

    Returns list of (hour, strength_mpa) tuples.
    """
    if hours is None:
        hours = list(range(1, 37))

    # Get predictions at 8, 16, 24 hours
    base_features = features_dict.copy()
    predictions_at_key_hours = {}

    for dur, target in [(8, "strength_8hr_MPa"), (16, "strength_16hr_MPa"), (24, "strength_24hr_MPa")]:
        f = base_features.copy()
        f["curing_duration_hr"] = dur
        pred = predict_strength(f)
        predictions_at_key_hours[dur] = pred[target]

    # Interpolate using cubic spline-like approach
    # anchor points: (0, 0), (8, s8), (16, s16), (24, s24), extrapolate to 36
    s0 = 0.0
    s8 = predictions_at_key_hours[8]
    s16 = predictions_at_key_hours[16]
    s24 = predictions_at_key_hours[24]

    # Simple piecewise linear + logarithmic growth extrapolation
    curve = []
    for h in hours:
        if h <= 0:
            s = 0.0
        elif h <= 8:
            s = s0 + (s8 - s0) * (h / 8.0)
        elif h <= 16:
            s = s8 + (s16 - s8) * ((h - 8) / 8.0)
        elif h <= 24:
            s = s16 + (s24 - s16) * ((h - 16) / 8.0)
        else:
            # Extrapolate: logarithmic growth beyond 24h
            rate = (s24 - s16) / 8.0  # rate of gain in last segment
            decay = 0.7  # slowing growth
            s = s24 + rate * decay * (h - 24)

        curve.append((h, max(0.0, s)))

    return curve


def find_earliest_demould(features_dict, required_mpa, safety_factor=SAFETY_FACTOR):
    """
    Find the earliest hour where predicted strength >= required_mpa * safety_factor.

    Returns:
        dict with demould_hour, predicted_strength, target_strength
    """
    target = required_mpa * safety_factor
    curve = predict_strength_curve(features_dict, hours=[h * 0.5 for h in range(2, 73)])  # 1h to 36h in 0.5h steps

    for hour, strength in curve:
        if strength >= target:
            return {
                "demould_hour": hour,
                "predicted_strength": round(strength, 2),
                "target_strength": round(target, 2),
                "required_mpa": required_mpa,
                "safety_factor": safety_factor,
            }

    # If never reached, return 36h as max
    return {
        "demould_hour": 36.0,
        "predicted_strength": round(curve[-1][1], 2),
        "target_strength": round(target, 2),
        "required_mpa": required_mpa,
        "safety_factor": safety_factor,
    }


def get_shap_explanation(features_dict, target_idx=2):
    """
    Compute SHAP values for a prediction (default: 24h strength).

    Returns:
        list of (feature_name, shap_value) sorted by |shap_value| descending, top 5.
    """
    _load_models()

    try:
        import shap
        X = _features_to_array(features_dict)
        X_scaled = _scaler.transform(X)

        # Use XGBoost model for SHAP (cleaner tree-based SHAP)
        explainer = shap.TreeExplainer(_xgb_models[target_idx])
        shap_values = explainer.shap_values(X_scaled)

        feature_shap = list(zip(FEATURE_COLS, shap_values[0]))
        feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
        return feature_shap[:5]

    except Exception:
        # Fallback: use feature importances from XGBoost
        importances = _xgb_models[target_idx].feature_importances_
        feature_imp = list(zip(FEATURE_COLS, importances))
        feature_imp.sort(key=lambda x: abs(x[1]), reverse=True)
        return feature_imp[:5]


def get_feature_importance_all():
    """Get average feature importance across all 3 targets."""
    _load_models()

    avg_importance = np.zeros(len(FEATURE_COLS))
    for t in range(3):
        avg_importance += _xgb_models[t].feature_importances_
    avg_importance /= 3

    feature_imp = list(zip(FEATURE_COLS, avg_importance))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    return feature_imp
