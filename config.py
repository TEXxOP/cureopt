"""
CureOpt AI — Central Configuration
All constants, regional profiles, cost parameters, and hyperparameters.
"""

import os

# ─── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASET_PATH = os.path.join(ASSETS_DIR, "synthetic_dataset.csv")
XGBOOST_MODEL_PATH = os.path.join(ASSETS_DIR, "xgboost_model.joblib")
RF_MODEL_PATH = os.path.join(ASSETS_DIR, "rf_model.joblib")
SCALER_PATH = os.path.join(ASSETS_DIR, "scaler.joblib")
ENSEMBLE_WEIGHTS_PATH = os.path.join(ASSETS_DIR, "ensemble_weights.joblib")

# ─── Dataset Generation ────────────────────────────────────────────────
RANDOM_SEED = 42
DATASET_SIZE = 5000

# ─── Feature Ranges ────────────────────────────────────────────────────
FEATURE_RANGES = {
    "cement_pct": (35.0, 65.0),
    "fly_ash_pct": (0.0, 30.0),
    "water_cement_ratio": (0.35, 0.55),
    "curing_method": [0, 1, 2],          # 0=Normal, 1=Steam, 2=Heated Chamber
    "admixture_type": [0, 1, 2],         # 0=None, 1=Accelerator, 2=Plasticizer
    "ambient_temp_C": (5.0, 45.0),
    "humidity_pct": (30.0, 95.0),
    "curing_duration_hr": (6.0, 36.0),
    "automation_level": [1, 2, 3],       # 1=Manual, 2=Semi, 3=Fully automated
}

CURING_METHOD_LABELS = {0: "Normal", 1: "Steam", 2: "Heated Chamber"}
ADMIXTURE_LABELS = {0: "None", 1: "Accelerator", 2: "Plasticizer"}
AUTOMATION_LABELS = {1: "Manual", 2: "Semi-Automated", 3: "Fully Automated"}

# ─── Nurse-Saul Maturity Function ──────────────────────────────────────
DATUM_TEMPERATURE = 0.0   # °C — standard datum for Nurse-Saul
STRENGTH_NOISE_COV = 0.03  # 3% coefficient of variation

# ─── Climatic Zone Profiles ────────────────────────────────────────────
CLIMATIC_PROFILES = {
    "Chennai (Hot/Humid)": {
        "ambient_temp_C": 35.0,
        "humidity_pct": 80.0,
        "description": "Hot and humid coastal climate. Faster initial hydration, risk of moisture loss.",
    },
    "Delhi (Cold Winter)": {
        "ambient_temp_C": 12.0,
        "humidity_pct": 50.0,
        "description": "Cold winters significantly slow hydration. Steam curing often essential.",
    },
    "Mumbai (Humid Coastal)": {
        "ambient_temp_C": 30.0,
        "humidity_pct": 85.0,
        "description": "Warm and humid. Good curing conditions, moderate hydration rate.",
    },
}

# ─── Cost Model Parameters (₹) ─────────────────────────────────────────
COST_PARAMS = {
    # Material cost per cubic meter (base, varies by mix)
    "cement_cost_per_pct": 120.0,        # ₹ per 1% cement content per m³
    "fly_ash_cost_per_pct": 40.0,        # ₹ per 1% fly ash per m³
    "water_cost_base": 50.0,             # ₹ base water cost per m³
    "admixture_costs": {
        "None": 0.0,
        "Accelerator": 800.0,            # ₹ per m³
        "Plasticizer": 600.0,            # ₹ per m³
    },

    # Energy cost per hour of curing
    "energy_cost_per_hr": {
        "Normal": 50.0,                  # ₹/hr — ambient, minimal energy
        "Steam": 350.0,                  # ₹/hr — steam generation
        "Heated Chamber": 250.0,         # ₹/hr — electric heating
    },

    # Labor cost per hour (by automation level)
    "labor_cost_per_hr": {
        1: 200.0,                        # Manual
        2: 120.0,                        # Semi-automated
        3: 60.0,                         # Fully automated
    },

    # Mold occupancy cost per hour
    "mold_occupancy_cost_per_hr": 333.0,  # ₹8,000/day ÷ 24h ≈ ₹333/hr
}

# Regional cost multipliers
REGIONAL_COST_MULTIPLIERS = {
    "Chennai (Hot/Humid)": 1.00,
    "Delhi (Cold Winter)": 1.05,          # Slightly higher logistics
    "Mumbai (Humid Coastal)": 1.08,       # Higher city costs
    "Custom": 1.00,
}

# ─── ML Model Hyperparameters ──────────────────────────────────────────
ML_CONFIG = {
    "test_size": 0.15,
    "val_size": 0.15,
    "optuna_trials": 50,
    "cv_folds": 5,

    "xgboost_param_space": {
        "n_estimators": (200, 800),
        "max_depth": (3, 8),
        "learning_rate": (0.01, 0.3),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "min_child_weight": (1, 10),
    },

    "rf_param_space": {
        "n_estimators": (50, 150),
        "max_depth": (5, 10),
        "min_samples_split": (2, 10),
        "min_samples_leaf": (1, 5),
    },
}

# ─── GA Optimizer Configuration ─────────────────────────────────────────
GA_CONFIG = {
    "population_size": 100,
    "n_generations": 50,
    "crossover_prob": 0.7,
    "mutation_prob": 0.2,
    "tournament_size": 3,
    "early_stop_generations": 10,     # Stop if no improvement for N gens
    "early_stop_threshold": 0.001,    # Minimum relative improvement
    "constraint_penalty": 1000.0,
    "fitness_weights": {
        "cycle_time": 0.5,
        "cost": 0.5,
    },
}

# Safety factor for strength constraint (IS:456 compliance)
SAFETY_FACTOR = 1.10  # Target strength = Required × 1.10

# ─── Dashboard Configuration ───────────────────────────────────────────
DASHBOARD_CONFIG = {
    "page_title": "CureOpt AI — Cycle Time Optimization",
    "page_icon": "🏗️",
    "layout": "wide",
    "default_required_strength": 25.0,  # MPa
    "max_scenarios": 4,
}

# ─── Simulation Baseline ───────────────────────────────────────────────
BASELINE_CYCLE_TIME_HR = 24.0  # Standard industry practice: Normal curing at 24h
