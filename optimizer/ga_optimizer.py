"""
CureOpt AI — Genetic Algorithm Optimizer + Grid Search Fallback
Minimizes f(x) = w₁·CycleTime + w₂·Cost subject to Strength ≥ Required.
"""

import os
import sys
import random
import numpy as np
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GA_CONFIG, SAFETY_FACTOR, FEATURE_RANGES,
    CURING_METHOD_LABELS, ADMIXTURE_LABELS, BASELINE_CYCLE_TIME_HR,
    RANDOM_SEED
)

# Defer heavy imports to avoid import-time failures when models aren't trained yet
_deap_imported = False


def _import_deap():
    """Lazy import DEAP to avoid import errors when not needed."""
    global _deap_imported
    if not _deap_imported:
        from deap import base, creator, tools, algorithms
        _deap_imported = True
    from deap import base, creator, tools, algorithms
    return base, creator, tools, algorithms


def _get_prediction_and_demould(features_dict, required_mpa):
    """Get demould prediction using the ML models."""
    from models.predict import predict_strength, find_earliest_demould
    demould = find_earliest_demould(features_dict, required_mpa, safety_factor=SAFETY_FACTOR)
    return demould


def _get_cost(params, region, automation_level):
    """Get cost for given parameters."""
    from optimizer.cost_model import calculate_cost
    return calculate_cost(params, region=region, automation_level=automation_level)


# ─── Grid Search Fallback ───────────────────────────────────────────────

def grid_search_optimize(required_mpa, region, ambient_temp, humidity,
                         automation_level=2):
    """
    Pre-computed grid search over canonical scenarios.
    3 curing methods × 3 admixtures × 5 w/c ratio steps = 45 scenarios.

    Returns ranked list of scenarios.
    """
    wc_steps = [0.35, 0.40, 0.45, 0.50, 0.55]
    cement_steps = [40.0, 50.0, 60.0]
    fly_ash_steps = [5.0, 15.0]

    scenarios = []

    for wc, cement, fly_ash, curing, admixture in product(
        wc_steps, cement_steps, fly_ash_steps,
        CURING_METHOD_LABELS.keys(), ADMIXTURE_LABELS.keys()
    ):
        features = {
            "cement_pct": cement,
            "fly_ash_pct": fly_ash,
            "water_cement_ratio": wc,
            "curing_method": curing,
            "admixture_type": admixture,
            "ambient_temp_C": ambient_temp,
            "humidity_pct": humidity,
            "curing_duration_hr": 24.0,
            "automation_level": automation_level,
        }

        try:
            demould = _get_prediction_and_demould(features, required_mpa)
            cycle_time = demould["demould_hour"]

            cost_params = features.copy()
            cost_params["cycle_time_hr"] = cycle_time
            cost = _get_cost(cost_params, region, automation_level)

            # Fitness: weighted sum (normalized)
            norm_time = cycle_time / BASELINE_CYCLE_TIME_HR
            baseline_cost = _get_cost({
                "cement_pct": 50, "fly_ash_pct": 10, "water_cement_ratio": 0.45,
                "curing_method": "Normal", "admixture_type": "None", "cycle_time_hr": 24
            }, region, automation_level)["total"]
            norm_cost = cost["total"] / max(baseline_cost, 1)

            w_time = GA_CONFIG["fitness_weights"]["cycle_time"]
            w_cost = GA_CONFIG["fitness_weights"]["cost"]
            fitness = w_time * norm_time + w_cost * norm_cost

            # Penalty if strength not met
            if demould["predicted_strength"] < required_mpa:
                fitness += GA_CONFIG["constraint_penalty"]

            scenarios.append({
                "cement_pct": cement,
                "fly_ash_pct": fly_ash,
                "water_cement_ratio": wc,
                "curing_method": CURING_METHOD_LABELS[curing],
                "admixture_type": ADMIXTURE_LABELS[admixture],
                "cycle_time_hr": round(cycle_time, 1),
                "predicted_strength": demould["predicted_strength"],
                "cost": cost,
                "fitness": round(fitness, 4),
                "time_reduction_pct": round(
                    (1 - cycle_time / BASELINE_CYCLE_TIME_HR) * 100, 1
                ),
                "cost_index": round(cost["total"] / max(baseline_cost, 1), 3),
            })
        except Exception:
            continue

    # Sort by fitness (lower is better)
    scenarios.sort(key=lambda s: s["fitness"])
    return scenarios


# ─── Genetic Algorithm Optimizer ────────────────────────────────────────

def ga_optimize(required_mpa, region, ambient_temp, humidity,
                automation_level=2, verbose=False):
    """
    DEAP-based genetic algorithm optimizer.

    Chromosome: [cement_pct, fly_ash_pct, wc_ratio, curing_method, admixture]

    Returns:
        dict with optimal strategy
    """
    base, creator, tools_deap, algorithms = _import_deap()

    # Clean up any previous DEAP creator definitions
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Gene generators
    toolbox.register("cement", random.uniform, 35.0, 65.0)
    toolbox.register("fly_ash", random.uniform, 0.0, 30.0)
    toolbox.register("wc_ratio", random.uniform, 0.35, 0.55)
    toolbox.register("curing", random.randint, 0, 2)
    toolbox.register("admixture", random.randint, 0, 2)

    def create_individual():
        return creator.Individual([
            toolbox.cement(),
            toolbox.fly_ash(),
            toolbox.wc_ratio(),
            toolbox.curing(),
            toolbox.admixture(),
        ])

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools_deap.initRepeat, list, toolbox.individual)

    # Evaluation function
    baseline_cost_val = _get_cost({
        "cement_pct": 50, "fly_ash_pct": 10, "water_cement_ratio": 0.45,
        "curing_method": "Normal", "admixture_type": "None", "cycle_time_hr": 24
    }, region, automation_level)["total"]

    def evaluate(individual):
        cement, fly_ash, wc, curing, admixture = individual

        # Clamp values
        cement = np.clip(cement, 35.0, 65.0)
        fly_ash = np.clip(fly_ash, 0.0, 30.0)
        wc = np.clip(wc, 0.35, 0.55)
        curing = int(np.clip(round(curing), 0, 2))
        admixture = int(np.clip(round(admixture), 0, 2))

        features = {
            "cement_pct": cement,
            "fly_ash_pct": fly_ash,
            "water_cement_ratio": wc,
            "curing_method": curing,
            "admixture_type": admixture,
            "ambient_temp_C": ambient_temp,
            "humidity_pct": humidity,
            "curing_duration_hr": 24.0,
            "automation_level": automation_level,
        }

        try:
            demould = _get_prediction_and_demould(features, required_mpa)
            cycle_time = demould["demould_hour"]

            cost_params = features.copy()
            cost_params["cycle_time_hr"] = cycle_time
            cost = _get_cost(cost_params, region, automation_level)

            norm_time = cycle_time / BASELINE_CYCLE_TIME_HR
            norm_cost = cost["total"] / max(baseline_cost_val, 1)

            w_time = GA_CONFIG["fitness_weights"]["cycle_time"]
            w_cost = GA_CONFIG["fitness_weights"]["cost"]
            fitness = w_time * norm_time + w_cost * norm_cost

            if demould["predicted_strength"] < required_mpa:
                fitness += GA_CONFIG["constraint_penalty"]

            return (fitness,)
        except Exception:
            return (GA_CONFIG["constraint_penalty"] * 2,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools_deap.cxTwoPoint)
    toolbox.register("mutate", tools_deap.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools_deap.selTournament, tournsize=GA_CONFIG["tournament_size"])

    # Run GA
    random.seed(RANDOM_SEED)
    pop = toolbox.population(n=GA_CONFIG["population_size"])

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    best_fitness_history = []

    for gen in range(GA_CONFIG["n_generations"]):
        # Select + reproduce
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CONFIG["crossover_prob"]:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < GA_CONFIG["mutation_prob"]:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Track best
        best = tools_deap.selBest(pop, 1)[0]
        best_fitness_history.append(best.fitness.values[0])

        if verbose:
            print(f"  Gen {gen+1}: best fitness = {best.fitness.values[0]:.4f}")

        # Early stopping
        if len(best_fitness_history) >= GA_CONFIG["early_stop_generations"]:
            recent = best_fitness_history[-GA_CONFIG["early_stop_generations"]:]
            improvement = (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-6)
            if improvement < GA_CONFIG["early_stop_threshold"]:
                if verbose:
                    print(f"  ⏹️ Early stopping at generation {gen+1}")
                break

    # Extract best solution
    best = tools_deap.selBest(pop, 1)[0]
    cement, fly_ash, wc, curing, admixture = best

    cement = np.clip(cement, 35.0, 65.0)
    fly_ash = np.clip(fly_ash, 0.0, 30.0)
    wc = np.clip(wc, 0.35, 0.55)
    curing = int(np.clip(round(curing), 0, 2))
    admixture = int(np.clip(round(admixture), 0, 2))

    features = {
        "cement_pct": round(cement, 1),
        "fly_ash_pct": round(fly_ash, 1),
        "water_cement_ratio": round(wc, 3),
        "curing_method": curing,
        "admixture_type": admixture,
        "ambient_temp_C": ambient_temp,
        "humidity_pct": humidity,
        "curing_duration_hr": 24.0,
        "automation_level": automation_level,
    }

    demould = _get_prediction_and_demould(features, required_mpa)
    cost_params = features.copy()
    cost_params["cycle_time_hr"] = demould["demould_hour"]
    cost = _get_cost(cost_params, region, automation_level)

    return {
        "cement_pct": round(cement, 1),
        "fly_ash_pct": round(fly_ash, 1),
        "water_cement_ratio": round(wc, 3),
        "curing_method": CURING_METHOD_LABELS[curing],
        "admixture_type": ADMIXTURE_LABELS[admixture],
        "cycle_time_hr": round(demould["demould_hour"], 1),
        "predicted_strength": demould["predicted_strength"],
        "cost": cost,
        "fitness": round(best.fitness.values[0], 4),
        "time_reduction_pct": round(
            (1 - demould["demould_hour"] / BASELINE_CYCLE_TIME_HR) * 100, 1
        ),
        "cost_index": round(cost["total"] / max(baseline_cost_val, 1), 3),
        "generations_run": len(best_fitness_history),
    }


# ─── Main Optimize Function ────────────────────────────────────────────

def optimize(required_mpa=25.0, region="Chennai (Hot/Humid)", ambient_temp=35.0,
             humidity=70.0, automation_level=2, use_ga=True, verbose=False):
    """
    Run full optimization. Uses GA by default with grid-search fallback.

    Returns:
        dict with optimal strategy and metadata
    """
    if use_ga:
        try:
            result = ga_optimize(
                required_mpa=required_mpa,
                region=region,
                ambient_temp=ambient_temp,
                humidity=humidity,
                automation_level=automation_level,
                verbose=verbose
            )
            result["method"] = "Genetic Algorithm"
            return result
        except Exception as e:
            if verbose:
                print(f"  ⚠️ GA failed ({e}), falling back to grid search")

    # Grid search fallback
    scenarios = grid_search_optimize(
        required_mpa=required_mpa,
        region=region,
        ambient_temp=ambient_temp,
        humidity=humidity,
        automation_level=automation_level
    )
    if scenarios:
        best = scenarios[0]
        best["method"] = "Grid Search"
        return best
    else:
        return {"error": "No feasible solution found"}


def run_scenario_comparison(required_mpa=25.0, region="Chennai (Hot/Humid)",
                            ambient_temp=35.0, humidity=70.0,
                            automation_level=2, top_n=4):
    """
    Run grid search and return top N scenarios for comparison view.
    """
    scenarios = grid_search_optimize(
        required_mpa=required_mpa,
        region=region,
        ambient_temp=ambient_temp,
        humidity=humidity,
        automation_level=automation_level
    )
    return scenarios[:top_n]
