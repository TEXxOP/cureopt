"""
CureOpt AI — Parametric Cost Model
Calculates total cycle cost = material + energy + labor + mold occupancy.
Region-aware pricing with automation-level adjustments.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    COST_PARAMS, REGIONAL_COST_MULTIPLIERS,
    CURING_METHOD_LABELS, ADMIXTURE_LABELS
)


def calculate_cost(params, region="Chennai (Hot/Humid)", automation_level=2):
    """
    Calculate total cycle cost breakdown.

    Args:
        params: dict with keys:
            - cement_pct (float)
            - fly_ash_pct (float)
            - water_cement_ratio (float)
            - curing_method (str or int): "Normal"/"Steam"/"Heated Chamber" or 0/1/2
            - admixture_type (str or int): "None"/"Accelerator"/"Plasticizer" or 0/1/2
            - cycle_time_hr (float): total hours in mold
        region: str — key in REGIONAL_COST_MULTIPLIERS
        automation_level: int — 1, 2, or 3

    Returns:
        dict with 'material', 'energy', 'labor', 'mold_occupancy', 'total' (all in ₹)
    """
    # Resolve labels if integers were passed
    curing = params.get("curing_method", "Normal")
    if isinstance(curing, (int, float)):
        curing = CURING_METHOD_LABELS.get(int(curing), "Normal")

    admixture = params.get("admixture_type", "None")
    if isinstance(admixture, (int, float)):
        admixture = ADMIXTURE_LABELS.get(int(admixture), "None")

    cycle_time = params.get("cycle_time_hr", 24.0)
    cement_pct = params.get("cement_pct", 50.0)
    fly_ash_pct = params.get("fly_ash_pct", 10.0)
    wc_ratio = params.get("water_cement_ratio", 0.45)

    regional_multiplier = REGIONAL_COST_MULTIPLIERS.get(region, 1.0)

    # ----- Material Cost -----
    material = (
        COST_PARAMS["cement_cost_per_pct"] * cement_pct +
        COST_PARAMS["fly_ash_cost_per_pct"] * fly_ash_pct +
        COST_PARAMS["water_cost_base"] * (wc_ratio / 0.45) +  # normalized to baseline w/c
        COST_PARAMS["admixture_costs"].get(admixture, 0.0)
    )

    # ----- Energy Cost -----
    energy_rate = COST_PARAMS["energy_cost_per_hr"].get(curing, 50.0)
    energy = energy_rate * cycle_time

    # ----- Labor Cost -----
    labor_rate = COST_PARAMS["labor_cost_per_hr"].get(automation_level, 120.0)
    labor = labor_rate * cycle_time

    # ----- Mold Occupancy Cost -----
    mold_occupancy = COST_PARAMS["mold_occupancy_cost_per_hr"] * cycle_time

    # Apply regional multiplier
    total = (material + energy + labor + mold_occupancy) * regional_multiplier

    return {
        "material": round(material * regional_multiplier, 2),
        "energy": round(energy * regional_multiplier, 2),
        "labor": round(labor * regional_multiplier, 2),
        "mold_occupancy": round(mold_occupancy * regional_multiplier, 2),
        "total": round(total, 2),
        "currency": "₹",
        "region": region,
        "curing_method": curing,
        "cycle_time_hr": cycle_time,
    }


def calculate_cost_comparison(base_params, region="Chennai (Hot/Humid)", automation_level=2):
    """
    Compare costs across all curing methods for a given mix design.

    Returns:
        dict mapping curing_method -> cost_dict
    """
    results = {}
    for method_int, method_name in CURING_METHOD_LABELS.items():
        p = base_params.copy()
        p["curing_method"] = method_name
        results[method_name] = calculate_cost(p, region=region, automation_level=automation_level)
    return results


def calculate_baseline_cost(region="Chennai (Hot/Humid)", automation_level=2):
    """Calculate baseline cost (Normal curing, 24h, standard mix)."""
    baseline_params = {
        "cement_pct": 50.0,
        "fly_ash_pct": 10.0,
        "water_cement_ratio": 0.45,
        "curing_method": "Normal",
        "admixture_type": "None",
        "cycle_time_hr": 24.0,
    }
    return calculate_cost(baseline_params, region=region, automation_level=automation_level)
