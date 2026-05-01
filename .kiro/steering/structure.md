# Project Structure

```
├── src/
│   ├── main.py                  # CLI entry point — argument parsing, data loading, model export
│   ├── app.py                   # Streamlit dashboard — live GA visualization and model download
│   ├── engine/
│   │   ├── ga_rf_optimizer.py   # PRIMARY: custom GA implementation from scratch
│   │   ├── ga_deap.py           # ALTERNATIVE: same GA using the DEAP framework
│   │   ├── ga_logger.py         # Logging, per-phase profiling, summary report generation
│   │   └── __init__.py
│   ├── demo/                    # Standalone scripts to visualize each GA operator in isolation
│   │   ├── demo_selection.py
│   │   ├── demo_crossover.py
│   │   ├── demo_mutate.py
│   │   └── demo_ga_operators.py
│   └── test/
│       └── test_ga_rf_optimizer.py  # pytest unit tests for ga_rf_optimizer.py
├── data/
│   ├── processed/
│   │   └── diabetes_treated.csv     # Pre-processed dataset (Fase 1 output) — source of truth
│   └── raw/
│       └── diabetes.csv             # Original Pima Indians dataset
├── models/                          # Serialized .pkl models (git-tracked originals, generated outputs)
│   ├── model_diabetes_rf_original.pkl        # Fase 1 baseline model (required for comparison)
│   └── model_diabetes_rf_optimized_*.pkl     # GA-optimized models (timestamped)
├── logs/                            # Runtime outputs — one log + one summary per execution
│   ├── ga_rf_optimizer_*.log
│   └── summary_ga_rf_optimizer_*.txt
├── images/                          # Result summary figures (auto-generated when goal is reached)
├── .env                             # GA_LOG_LEVEL=INFO|DEBUG (not committed)
├── .env.example                     # Template for .env
├── requirements.txt                 # Pinned dependencies
└── system_diabetes/                 # Separate sub-project (Fase 1 diagnostic system — standalone)
```

## Architecture

```
main.py / app.py
    └── engine/ga_rf_optimizer.py   (or ga_deap.py)
            ├── Individual           class — list subclass with .fitness attribute
            ├── create_seeded_pop()  population init with GridSearch seed at index 0
            ├── evaluate()           fitness = mean_cv - 0.1 × std_cv (5-fold CV)
            ├── crossover()          two-point with divergence guarantee
            ├── mutate()             uniform integer per gene
            ├── selection()          tournament (tournsize=3)
            ├── _gen_loop()          one full generation cycle
            └── run_ga()             main loop (CLI path)
        └── engine/ga_logger.py
                └── GALogger         per-run file logger + profiling + summary export
```

## Key Conventions

- **`Individual` extends `list`**: genes are accessed directly via `ind[i]`, not `ind.genes[i]`
- **`fitness = None`** signals that an individual needs re-evaluation (set after crossover/mutation)
- **`random_state=42`** is fixed on `RandomForestClassifier` for reproducible CV scores; the GA's own `random` module has no global seed (intentional stochasticity)
- **`n_jobs=-1`** is always passed to `RandomForestClassifier` to use all available CPU cores
- **`sys.path.insert(0, ...)`** is used in entry points (`main.py`, `app.py`, test files) to resolve `engine.*` imports regardless of working directory
- **Log sections** are prefixed with bracketed tags: `[RUN_START]`, `[GEN_STATS]`, `[CROSSOVER]`, `[MUTATION]`, etc.
- **Timestamps** follow `YYYYMMDD_HHMMSS` format for log/image filenames and `YYMMDDHHMM` for model filenames
- **Models are only exported** when `best_fitness > target_cv` (goal reached)
- **`system_diabetes/`** is an independent sub-project from Fase 1; do not modify it when working on the GA optimizer
