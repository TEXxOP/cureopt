"""
CureOpt AI — ML Model Training Pipeline
Trains XGBoost + Random Forest ensemble with Optuna hyperparameter tuning.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import optuna

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_PATH, XGBOOST_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH,
    ENSEMBLE_WEIGHTS_PATH, ML_CONFIG, ASSETS_DIR, RANDOM_SEED
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    "cement_pct", "fly_ash_pct", "water_cement_ratio",
    "curing_method", "admixture_type", "ambient_temp_C",
    "humidity_pct", "curing_duration_hr", "automation_level"
]

TARGET_COLS = ["strength_8hr_MPa", "strength_16hr_MPa", "strength_24hr_MPa"]


def load_and_split_data():
    """Load dataset and split into train/val/test."""
    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COLS].values

    # First split: train+val vs test (15%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=ML_CONFIG["test_size"], random_state=RANDOM_SEED
    )
    # Second split: train vs val (15% of original ≈ 17.6% of trainval)
    val_frac = ML_CONFIG["val_size"] / (1 - ML_CONFIG["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=RANDOM_SEED
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train):
    """Fit and save MinMaxScaler."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved: {SCALER_PATH}")
    return scaler, X_scaled


def tune_xgboost(X_train, y_train, target_idx, scaler):
    """Tune XGBoost for a single target using Optuna."""
    X_scaled = scaler.transform(X_train)
    y_target = y_train[:, target_idx]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *ML_CONFIG["xgboost_param_space"]["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *ML_CONFIG["xgboost_param_space"]["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *ML_CONFIG["xgboost_param_space"]["learning_rate"], log=True),
            "subsample": trial.suggest_float("subsample", *ML_CONFIG["xgboost_param_space"]["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *ML_CONFIG["xgboost_param_space"]["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *ML_CONFIG["xgboost_param_space"]["min_child_weight"]),
            "random_state": RANDOM_SEED,
            "verbosity": 0,
        }
        model = XGBRegressor(**params)
        scores = cross_val_score(
            model, X_scaled, y_target,
            scoring="neg_root_mean_squared_error",
            cv=ML_CONFIG["cv_folds"]
        )
        return -scores.mean()

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=ML_CONFIG["optuna_trials"], show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = RANDOM_SEED
    best_params["verbosity"] = 0
    return best_params


def tune_rf(X_train, y_train, target_idx, scaler):
    """Tune Random Forest for a single target using Optuna."""
    X_scaled = scaler.transform(X_train)
    y_target = y_train[:, target_idx]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *ML_CONFIG["rf_param_space"]["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *ML_CONFIG["rf_param_space"]["max_depth"]),
            "min_samples_split": trial.suggest_int("min_samples_split", *ML_CONFIG["rf_param_space"]["min_samples_split"]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", *ML_CONFIG["rf_param_space"]["min_samples_leaf"]),
            "random_state": RANDOM_SEED,
        }
        model = RandomForestRegressor(**params)
        scores = cross_val_score(
            model, X_scaled, y_target,
            scoring="neg_root_mean_squared_error",
            cv=ML_CONFIG["cv_folds"]
        )
        return -scores.mean()

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=ML_CONFIG["optuna_trials"], show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = RANDOM_SEED
    return best_params


def find_best_ensemble_weights(xgb_models, rf_models, X_val, y_val, scaler):
    """Find optimal ensemble weights on validation set."""
    X_val_scaled = scaler.transform(X_val)
    best_w_xgb = 0.7
    best_rmse = float("inf")

    for w_xgb in np.arange(0.5, 0.95, 0.05):
        w_rf = 1.0 - w_xgb
        total_rmse = 0
        for t in range(3):
            pred_xgb = xgb_models[t].predict(X_val_scaled)
            pred_rf = rf_models[t].predict(X_val_scaled)
            pred_ens = w_xgb * pred_xgb + w_rf * pred_rf
            total_rmse += np.sqrt(mean_squared_error(y_val[:, t], pred_ens))
        if total_rmse < best_rmse:
            best_rmse = total_rmse
            best_w_xgb = w_xgb

    return best_w_xgb, 1.0 - best_w_xgb


def evaluate_on_test(xgb_models, rf_models, w_xgb, w_rf, X_test, y_test, scaler):
    """Print evaluation metrics on test set."""
    X_test_scaled = scaler.transform(X_test)
    print("\n" + "=" * 70)
    print("📊 TEST SET EVALUATION — Ensemble (XGBoost × {:.0%} + RF × {:.0%})".format(w_xgb, w_rf))
    print("=" * 70)

    all_pass = True
    for t, target_name in enumerate(TARGET_COLS):
        pred_xgb = xgb_models[t].predict(X_test_scaled)
        pred_rf = rf_models[t].predict(X_test_scaled)
        pred_ens = w_xgb * pred_xgb + w_rf * pred_rf

        rmse = np.sqrt(mean_squared_error(y_test[:, t], pred_ens))
        mae = mean_absolute_error(y_test[:, t], pred_ens)
        r2 = r2_score(y_test[:, t], pred_ens)

        status_rmse = "✅" if rmse < 2.0 else "❌"
        status_r2 = "✅" if r2 > 0.92 else "❌"
        if rmse >= 2.0 or r2 <= 0.92:
            all_pass = False

        print(f"\n  {target_name}:")
        print(f"    RMSE: {rmse:.4f} MPa  {status_rmse} (target < 2.0)")
        print(f"    MAE:  {mae:.4f} MPa")
        print(f"    R²:   {r2:.4f}        {status_r2} (target > 0.92)")

    print("\n" + "=" * 70)
    if all_pass:
        print("🎉 ALL ACCEPTANCE CRITERIA PASSED!")
    else:
        print("⚠️  Some criteria not met — review model or data.")
    print("=" * 70)

    return all_pass


def train_full_pipeline():
    """
    End-to-end training pipeline:
    1. Load & split data
    2. Fit scaler
    3. Tune & train XGBoost (3 targets)
    4. Tune & train RF (3 targets)
    5. Find optimal ensemble weights
    6. Evaluate on test set
    7. Save all artifacts
    """
    print("🚀 CureOpt AI — Model Training Pipeline")
    print("=" * 70)

    # Check if dataset exists, generate if not
    if not os.path.exists(DATASET_PATH):
        print("⚙️  Dataset not found. Generating...")
        from data.generate_dataset import save_dataset
        save_dataset()

    # 1. Load data
    print("\n📂 Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # 2. Fit scaler
    print("\n📏 Fitting scaler...")
    scaler, X_train_scaled = fit_scaler(X_train)

    # 3. Train XGBoost models (one per target)
    xgb_models = []
    print("\n🌲 Training XGBoost models...")
    for t, target_name in enumerate(TARGET_COLS):
        print(f"\n   [{t+1}/3] Tuning XGBoost for {target_name}...")
        best_params = tune_xgboost(X_train, y_train, t, scaler)
        print(f"   Best params: {best_params}")

        model = XGBRegressor(**best_params)
        model.fit(X_train_scaled, y_train[:, t])
        xgb_models.append(model)

    # Save XGBoost models
    os.makedirs(ASSETS_DIR, exist_ok=True)
    joblib.dump(xgb_models, XGBOOST_MODEL_PATH)
    print(f"\n✅ XGBoost models saved: {XGBOOST_MODEL_PATH}")

    # 4. Train RF models
    rf_models = []
    print("\n🌳 Training Random Forest models...")
    for t, target_name in enumerate(TARGET_COLS):
        print(f"\n   [{t+1}/3] Tuning RF for {target_name}...")
        best_params = tune_rf(X_train, y_train, t, scaler)
        print(f"   Best params: {best_params}")

        model = RandomForestRegressor(**best_params)
        X_train_s = scaler.transform(X_train)
        model.fit(X_train_s, y_train[:, t])
        rf_models.append(model)

    joblib.dump(rf_models, RF_MODEL_PATH)
    print(f"\n✅ RF models saved: {RF_MODEL_PATH}")

    # 5. Find optimal ensemble weights
    print("\n⚖️  Finding optimal ensemble weights...")
    w_xgb, w_rf = find_best_ensemble_weights(xgb_models, rf_models, X_val, y_val, scaler)
    print(f"   Optimal weights: XGBoost={w_xgb:.2f}, RF={w_rf:.2f}")
    joblib.dump({"xgb": w_xgb, "rf": w_rf}, ENSEMBLE_WEIGHTS_PATH)

    # 6. Evaluate
    evaluate_on_test(xgb_models, rf_models, w_xgb, w_rf, X_test, y_test, scaler)

    print("\n✅ Training pipeline complete! All artifacts saved to assets/")


if __name__ == "__main__":
    train_full_pipeline()
