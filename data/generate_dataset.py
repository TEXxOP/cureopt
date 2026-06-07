"""
CureOpt AI — Synthetic Dataset Generation
Generates a 5,000-row dataset using domain-realistic distributions
validated against IS:456-2000 and modified Nurse-Saul maturity function.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RANDOM_SEED, DATASET_SIZE, DATUM_TEMPERATURE, STRENGTH_NOISE_COV,
    ASSETS_DIR, DATASET_PATH
)


def nurse_saul_maturity(temp_C, time_hr, datum_temp=DATUM_TEMPERATURE):
    """
    Compute Nurse-Saul maturity index M = Σ(T - T₀) × Δt
    where T is curing temperature and T₀ is datum temperature.
    """
    effective_temp = max(temp_C - datum_temp, 0)
    return effective_temp * time_hr


def compute_strength(cement_pct, fly_ash_pct, wc_ratio, curing_method,
                     admixture_type, ambient_temp_C, humidity_pct,
                     curing_duration_hr, automation_level, target_hour):
    """
    Compute predicted strength at target_hour using modified Nurse-Saul
    maturity with corrections for mix design, curing method, and humidity.

    Returns strength in MPa.
    """
    # ----- Base 28-day strength estimation (Abrams' law approximation) -----
    # f28 ~ k / (w/c)^n  —  simplified
    k_cement = 0.6 + (cement_pct - 35) / 60.0  # normalized [0.6, 1.1]
    base_28day = 55.0 * k_cement / (wc_ratio ** 0.85)

    # Fly ash effect: reduces early strength, slight boost to ultimate
    fly_ash_factor_early = 1.0 - 0.008 * fly_ash_pct  # reduces early by ~0.8% per %
    fly_ash_factor_late = 1.0 + 0.003 * fly_ash_pct   # slight late boost

    # ----- Curing temperature -----
    # Steam/heated curing raises effective temperature
    curing_temps = {
        0: ambient_temp_C,           # Normal: ambient
        1: ambient_temp_C + 40.0,    # Steam: +40°C
        2: ambient_temp_C + 25.0,    # Heated Chamber: +25°C
    }
    effective_curing_temp = curing_temps[curing_method]

    # Maturity at the target hour
    maturity = nurse_saul_maturity(effective_curing_temp, target_hour)

    # ----- Strength development curve (hyperbolic model) -----
    # S(t) = S_ult × M / (A + M)  where A is a rate constant
    # A depends on w/c ratio and admixture
    rate_constant_A = 300.0 + 200.0 * wc_ratio  # higher w/c → slower gain

    # Admixture effects
    if admixture_type == 1:  # Accelerator
        rate_constant_A *= 0.65    # 35% faster
    elif admixture_type == 2:  # Plasticizer
        rate_constant_A *= 1.10    # slightly slower (but lower w/c achievable)

    # Humidity correction: below 80% RH, curing efficiency drops
    humidity_factor = min(1.0, humidity_pct / 80.0)
    effective_maturity = maturity * humidity_factor

    # Ultimate strength with fly ash correction
    if target_hour <= 12:
        fly_factor = fly_ash_factor_early
    else:
        fly_factor = fly_ash_factor_early + (fly_ash_factor_late - fly_ash_factor_early) * \
                     min(1.0, (target_hour - 12) / 24.0)

    S_ult = base_28day * fly_factor

    # Hyperbolic strength development
    strength = S_ult * effective_maturity / (rate_constant_A + effective_maturity)

    # Clamp to reasonable range
    strength = max(0.0, min(strength, S_ult * 0.95))

    return strength


def generate_dataset(n_samples=DATASET_SIZE, seed=RANDOM_SEED):
    """
    Generate the synthetic dataset with domain-realistic distributions.

    Returns a pandas DataFrame with features and 3 target columns.
    """
    rng = np.random.RandomState(seed)

    # ----- Feature Generation -----

    # Cement %: roughly uniform 35–65
    cement_pct = rng.uniform(35.0, 65.0, n_samples)

    # Fly ash %: skewed toward lower values (most mixes have 0–15%)
    fly_ash_pct = rng.beta(2, 5, n_samples) * 30.0  # beta-distributed 0–30

    # Water-cement ratio: truncated normal μ=0.45, σ=0.05
    wc_ratio = np.clip(rng.normal(0.45, 0.05, n_samples), 0.35, 0.55)

    # Curing method: weighted categorical (Normal 40%, Steam 40%, Heated 20%)
    curing_method = rng.choice([0, 1, 2], n_samples, p=[0.40, 0.40, 0.20])

    # Admixture type: (None 50%, Accelerator 30%, Plasticizer 20%)
    admixture_type = rng.choice([0, 1, 2], n_samples, p=[0.50, 0.30, 0.20])

    # Ambient temperature: bimodal (hot zone ~35°C, cold zone ~15°C)
    temp_mode = rng.choice([0, 1], n_samples, p=[0.5, 0.5])
    ambient_temp = np.where(
        temp_mode == 0,
        np.clip(rng.normal(35.0, 5.0, n_samples), 5.0, 45.0),  # hot zone
        np.clip(rng.normal(15.0, 5.0, n_samples), 5.0, 45.0),  # cold zone
    )

    # Humidity: roughly uniform 30–95%
    humidity_pct = rng.uniform(30.0, 95.0, n_samples)

    # Curing duration: uniform 6–36h
    curing_duration_hr = rng.uniform(6.0, 36.0, n_samples)

    # Automation level: weighted (Manual 40%, Semi 40%, Full 20%)
    automation_level = rng.choice([1, 2, 3], n_samples, p=[0.40, 0.40, 0.20])

    # ----- Target Generation -----
    strength_8hr = np.zeros(n_samples)
    strength_16hr = np.zeros(n_samples)
    strength_24hr = np.zeros(n_samples)

    for i in range(n_samples):
        for target_hr, arr in [(8, strength_8hr), (16, strength_16hr), (24, strength_24hr)]:
            base_strength = compute_strength(
                cement_pct[i], fly_ash_pct[i], wc_ratio[i],
                curing_method[i], admixture_type[i],
                ambient_temp[i], humidity_pct[i],
                curing_duration_hr[i], automation_level[i],
                target_hr
            )
            # Add realistic noise: COV = 8% of predicted strength
            noise_sigma = STRENGTH_NOISE_COV * max(base_strength, 5.0)
            noisy_strength = base_strength + rng.normal(0, noise_sigma)
            arr[i] = max(0.5, noisy_strength)  # floor at 0.5 MPa

    # ----- Assemble DataFrame -----
    df = pd.DataFrame({
        "cement_pct": np.round(cement_pct, 2),
        "fly_ash_pct": np.round(fly_ash_pct, 2),
        "water_cement_ratio": np.round(wc_ratio, 4),
        "curing_method": curing_method.astype(int),
        "admixture_type": admixture_type.astype(int),
        "ambient_temp_C": np.round(ambient_temp, 1),
        "humidity_pct": np.round(humidity_pct, 1),
        "curing_duration_hr": np.round(curing_duration_hr, 1),
        "automation_level": automation_level.astype(int),
        "strength_8hr_MPa": np.round(strength_8hr, 2),
        "strength_16hr_MPa": np.round(strength_16hr, 2),
        "strength_24hr_MPa": np.round(strength_24hr, 2),
    })

    return df


def save_dataset(df=None):
    """Generate and save the dataset to CSV."""
    if df is None:
        df = generate_dataset()

    os.makedirs(ASSETS_DIR, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"✅ Dataset saved: {DATASET_PATH}")
    print(f"   Shape: {df.shape}")
    print(f"   Strength 8h  — mean: {df['strength_8hr_MPa'].mean():.1f}, "
          f"std: {df['strength_8hr_MPa'].std():.1f}, "
          f"range: [{df['strength_8hr_MPa'].min():.1f}, {df['strength_8hr_MPa'].max():.1f}]")
    print(f"   Strength 16h — mean: {df['strength_16hr_MPa'].mean():.1f}, "
          f"std: {df['strength_16hr_MPa'].std():.1f}, "
          f"range: [{df['strength_16hr_MPa'].min():.1f}, {df['strength_16hr_MPa'].max():.1f}]")
    print(f"   Strength 24h — mean: {df['strength_24hr_MPa'].mean():.1f}, "
          f"std: {df['strength_24hr_MPa'].std():.1f}, "
          f"range: [{df['strength_24hr_MPa'].min():.1f}, {df['strength_24hr_MPa'].max():.1f}]")
    return df


if __name__ == "__main__":
    save_dataset()
