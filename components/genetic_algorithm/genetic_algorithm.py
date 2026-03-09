# ============================================================
# Genetic Algorithm Feature Selection Component
# Uses DEAP to evolve optimal feature subsets
# ============================================================

import argparse
import os
import json
import random
import time
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import mlflow

def parse_args():
    parser = argparse.ArgumentParser("genetic_algorithm")
    parser.add_argument("--train_filtered", type=str)
    parser.add_argument("--test_filtered", type=str)
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument("--n_generations", type=int, default=20)
    parser.add_argument("--crossover_prob", type=float, default=0.7)
    parser.add_argument("--mutation_prob", type=float, default=0.2)
    parser.add_argument("--tournament_size", type=int, default=3)
    parser.add_argument("--min_features", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_ga_selected", type=str)
    parser.add_argument("--test_ga_selected", type=str)
    parser.add_argument("--ga_metrics", type=str)
    return parser.parse_args()

def load_parquet_folder(path):
    """Load all parquet files from a folder"""
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(path, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)

def setup_deap(n_features, min_features, random_seed):
    """Set up DEAP genetic algorithm toolbox"""

    # Clean up any existing DEAP creator classes
    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual

    # Minimize RMSE (negative because DEAP maximizes by default)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Each gene is a binary value (0=feature excluded, 1=feature included)
    toolbox.register("attr_bool", random.randint, 0, 1)

    # Individual = binary vector of length n_features
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=n_features
    )

    # Population = list of individuals
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual
    )

    # Selection: tournament selection
    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=3
    )

    # Crossover: two-point crossover
    toolbox.register("mate", tools.cxTwoPoint)

    # Mutation: flip bit mutation
    toolbox.register(
        "mutate",
        tools.mutFlipBit,
        indpb=0.05
    )

    return toolbox

def evaluate_individual(individual, X, y, feature_names, min_features):
    """
    Fitness function:
    - Uses Random Forest cross-validation RMSE
    - Penalizes individuals with fewer than min_features selected
    """
    selected_idx = [i for i, bit in enumerate(individual) if bit == 1]

    # Penalize if too few features selected
    if len(selected_idx) < min_features:
        return (9999.0,)

    X_selected = X[:, selected_idx]

    # Use fast Random Forest with cross-validation
    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    # Negative MSE scores from sklearn
    scores = cross_val_score(
        model, X_selected, y,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    rmse = np.sqrt(-scores.mean())
    return (rmse,)

def run_genetic_algorithm(toolbox, X, y, feature_names,
                          population_size, n_generations,
                          crossover_prob, mutation_prob,
                          min_features, random_seed):
    """Run the genetic algorithm evolution"""

    random.seed(random_seed)
    np.random.seed(random_seed)

    # Register evaluation with data
    toolbox.register(
        "evaluate",
        evaluate_individual,
        X=X, y=y,
        feature_names=feature_names,
        min_features=min_features
    )

    # Initialize population
    population = toolbox.population(n=population_size)

    # Statistics to track
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # Hall of fame — keeps best individual
    hof = tools.HallOfFame(1)

    print(f"\n=== GENETIC ALGORITHM EVOLUTION ===")
    print(f"Population size : {population_size}")
    print(f"Generations     : {n_generations}")
    print(f"Crossover prob  : {crossover_prob}")
    print(f"Mutation prob   : {mutation_prob}")
    print(f"Min features    : {min_features}")
    print(f"Total features  : {len(feature_names)}")

    start_time = time.time()

    # Run evolution
    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=n_generations,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    elapsed = time.time() - start_time

    print(f"\nEvolution complete in {elapsed:.2f} seconds")

    # Get best individual
    best_individual = hof[0]
    best_features_idx = [i for i, bit in enumerate(best_individual) if bit == 1]
    best_features = [feature_names[i] for i in best_features_idx]
    best_rmse = best_individual.fitness.values[0]

    print(f"\n=== BEST INDIVIDUAL ===")
    print(f"Features selected : {len(best_features)}")
    print(f"Best RMSE         : {best_rmse:.4f}")
    print(f"Selected features : {best_features[:10]}...")

    return best_features, best_rmse, elapsed, logbook

def main():
    args = parse_args()

    mlflow.start_run()

    print("=" * 50)
    print("GENETIC ALGORITHM FEATURE SELECTION COMPONENT")
    print("=" * 50)

    # ── Load Data ────────────────────────────────────────────
    print("\nLoading filtered features...")
    train_df = load_parquet_folder(args.train_filtered)
    test_df = load_parquet_folder(args.test_filtered)

    print(f"Train loaded : {train_df.shape}")
    print(f"Test loaded  : {test_df.shape}")

    # Separate features from target
    target_col = "target_RUL"
    feature_cols = [c for c in train_df.columns if c != target_col]

    X = train_df[feature_cols].values
    y = train_df[target_col].values

    print(f"Features available : {len(feature_cols)}")

    # ── Setup DEAP ───────────────────────────────────────────
    toolbox = setup_deap(
        n_features=len(feature_cols),
        min_features=args.min_features,
        random_seed=args.random_seed
    )

    # ── Run Genetic Algorithm ────────────────────────────────
    best_features, best_rmse, elapsed, logbook = run_genetic_algorithm(
        toolbox=toolbox,
        X=X,
        y=y,
        feature_names=feature_cols,
        population_size=args.population_size,
        n_generations=args.n_generations,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        min_features=args.min_features,
        random_seed=args.random_seed
    )

    # ── Apply Selection ──────────────────────────────────────
    train_ga_selected = train_df[best_features + [target_col]]
    test_ga_selected = test_df[best_features]

    print(f"\n=== FINAL SHAPES ===")
    print(f"Train GA selected : {train_ga_selected.shape}")
    print(f"Test GA selected  : {test_ga_selected.shape}")

    # ── Log Metrics ──────────────────────────────────────────
    mlflow.log_metric("ga_best_rmse", best_rmse)
    mlflow.log_metric("ga_selected_features", len(best_features))
    mlflow.log_metric("ga_runtime_seconds", elapsed)
    mlflow.log_param("population_size", args.population_size)
    mlflow.log_param("n_generations", args.n_generations)
    mlflow.log_param("crossover_prob", args.crossover_prob)
    mlflow.log_param("mutation_prob", args.mutation_prob)

    # ── Save Outputs ─────────────────────────────────────────
    os.makedirs(args.train_ga_selected, exist_ok=True)
    os.makedirs(args.test_ga_selected, exist_ok=True)
    os.makedirs(args.ga_metrics, exist_ok=True)

    train_ga_selected.to_parquet(
        os.path.join(args.train_ga_selected, "train_ga_selected.parquet")
    )
    test_ga_selected.to_parquet(
        os.path.join(args.test_ga_selected, "test_ga_selected.parquet")
    )

    # Save GA metrics
    metrics = {
        "ga_best_rmse": round(best_rmse, 4),
        "ga_selected_features": len(best_features),
        "ga_runtime_seconds": round(elapsed, 2),
        "population_size": args.population_size,
        "n_generations": args.n_generations,
        "crossover_prob": args.crossover_prob,
        "mutation_prob": args.mutation_prob,
        "selected_feature_names": best_features
    }

    with open(os.path.join(args.ga_metrics, "ga_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nGenetic algorithm component complete!")
    print(f"Train GA selected : {args.train_ga_selected}")
    print(f"Test GA selected  : {args.test_ga_selected}")
    print(f"GA metrics        : {args.ga_metrics}")

    mlflow.end_run()

if __name__ == "__main__":
    main()