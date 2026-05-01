# Product Overview

This project is a **Genetic Algorithm (GA) hyperparameter optimizer** for a `RandomForestClassifier`, developed as part of FIAP AI para DEVs Tech Challenge — Fase 2 (Turma 8IADT, Grupo 61).

## Goal

Automatically find the optimal combination of 5 `RandomForestClassifier` hyperparameters using a custom-built Genetic Algorithm, surpassing the CV 5-fold accuracy baseline of **0.7867** achieved by GridSearch in Fase 1.

## Dataset

Pima Indians Diabetes dataset (`data/processed/diabetes_treated.csv`), pre-processed in Fase 1. Binary classification target: `Outcome` (0 = non-diabetic, 1 = diabetic). Split: 80% train / 20% test, stratified, `random_state=42`.

## Chromosome Encoding

Each individual is a list of 5 integer genes:

| Index | Hyperparameter       | Type    | Range / Options        |
|-------|----------------------|---------|------------------------|
| [0]   | `n_estimators`       | int     | 20–60                  |
| [1]   | `max_depth`          | int     | 5, 10, 15, 20, 25      |
| [2]   | `min_samples_leaf`   | int     | 1–10                   |
| [3]   | `min_samples_split`  | int     | 2–20                   |
| [4]   | `max_features`       | binary  | 0=`sqrt`, 1=`log2`     |

## Key Design Decisions

- **Fitness**: `mean_cv - 0.1 × std_cv` (penalizes unstable models)
- **Seed elitism**: Population[0] is always the GridSearch best from Fase 1 (`[30, 15, 1, 5, 1]`)
- **Lazy evaluation**: Only re-evaluate individuals whose fitness was invalidated by crossover/mutation
- **Implicit elitism**: `best_ind` is tracked globally outside the population
- **Stopping criterion**: `cv_acc > PHASE1_CV_ACCURACY × (1 + target_improvement)`

## Interfaces

- **CLI**: `python src/main.py ga_rf_optimizer [options]`
- **Dashboard**: `streamlit run src/app.py` — live fitness evolution, hyperparameter exploration, model export
- **Alternative**: `ga_deap` engine using the DEAP framework (reference implementation)
