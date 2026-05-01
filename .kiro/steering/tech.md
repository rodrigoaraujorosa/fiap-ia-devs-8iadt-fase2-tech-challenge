# Tech Stack

## Language & Runtime

- **Python 3.10+** (required for `float | None` union type syntax)
- Virtual environment: `.venv/` (standard `venv`)

## Core Libraries

| Library | Role |
|---|---|
| `scikit-learn` | `RandomForestClassifier`, `cross_val_score`, `train_test_split` |
| `pandas` / `numpy` | Data loading and manipulation |
| `streamlit` | Interactive dashboard (`src/app.py`) |
| `altair` | Declarative charts in the Streamlit dashboard |
| `matplotlib` | Summary result images saved to `images/` |
| `deap` | Alternative GA engine (`src/engine/ga_deap.py`) |
| `python-dotenv` | Load `GA_LOG_LEVEL` from `.env` |
| `pytest` | Unit tests |

## Environment Configuration

Create a `.env` file at the project root to control log verbosity:

```dotenv
GA_LOG_LEVEL=INFO   # or DEBUG for detailed per-individual logs
```

## Common Commands

```bash
# Install dependencies (activate venv first)
pip install -r requirements.txt

# Run the Streamlit dashboard (recommended)
streamlit run src/app.py

# Run the GA via CLI (custom-built engine)
python src/main.py ga_rf_optimizer
python src/main.py ga_rf_optimizer --n_pop 20 --target_improvement 0.01 --mut_pb 0.5 --mut_indpb 0.4

# Run the GA via CLI (DEAP engine)
python src/main.py ga_deap --n_pop 10

# Run unit tests
pytest src/test/test_ga_rf_optimizer.py -v -s

# Run demo scripts (no dataset required)
python src/demo/demo_selection.py
python src/demo/demo_crossover.py
python src/demo/demo_mutate.py
python src/demo/demo_ga_operators.py
```

## CLI Parameters

| Parameter | Default | Description |
|---|---|---|
| `algorithm` | — | `ga_rf_optimizer` or `ga_deap` |
| `--n_pop` | 10 | Population size |
| `--target_improvement` | 0.0 | % improvement over baseline (e.g. `0.01` = +1%) |
| `--mut_pb` | 0.6 | Probability an individual undergoes mutation |
| `--mut_indpb` | 0.5 | Probability each gene is mutated individually |

## Output Artifacts

- **Models**: `models/model_diabetes_rf_optimized_{timestamp}.pkl` (only saved when goal is reached)
- **Logs**: `logs/ga_rf_optimizer_{timestamp}.log`
- **Summaries**: `logs/summary_ga_rf_optimizer_{timestamp}.txt`
- **Images**: `images/resultado_ga_rf_optimizer_{timestamp}.png`
