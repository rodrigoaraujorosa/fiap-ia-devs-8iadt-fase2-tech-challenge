import argparse
import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Garante que o módulo engine seja encontrado ao rodar direto de src/ ou da raiz
sys.path.insert(0, os.path.dirname(__file__))

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "diabetes_treated.csv"
)

ALGORITHMS = {
    "ga_handmade": ("engine.ga_handmade", "Handmade"),
    "ga_deap":     ("engine.ga_deap",     "DEAP"),
}

# Acurácia CV 5-fold obtida pelo GridSearch na Fase 1 (valor de referência)
PHASE1_CV_ACCURACY = 0.7867


def parse_args():
    parser = argparse.ArgumentParser(
        description="Otimização de hiperparâmetros de RandomForest via Algoritmo Genético."
    )
    parser.add_argument(
        "algorithm",
        choices=ALGORITHMS.keys(),
        help="Implementação do AG a executar: 'ga_handmade' ou 'ga_deap'.",
    )
    parser.add_argument(
        "--n_pop",
        type=int,
        default=10,
        metavar="N",
        help="Tamanho da população (padrão: 10).",
    )
    parser.add_argument(
        "--target_improvement",
        type=float,
        default=0.0,
        metavar="P",
        help="Melhoria percentual desejada sobre a meta base (padrão: 0.0 = sem melhoria, apenas superar PHASE1_CV_ACCURACY).",
    )
    parser.add_argument(
        "--cx_pb",
        type=float,
        default=0.7,
        metavar="F",
        help="Probabilidade de crossover entre dois indivíduos (padrão: 0.7).",
    )
    parser.add_argument(
        "--mut_pb",
        type=float,
        default=0.6,
        metavar="F",
        help="Probabilidade de um indivíduo sofrer mutação (padrão: 0.6).",
    )
    parser.add_argument(
        "--mut_indpb",
        type=float,
        default=0.5,
        metavar="F",
        help="Probabilidade de mutar cada gene individualmente (padrão: 0.5).",
    )
    return parser.parse_args()


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def decode_individual(ind):
    return {
        "n_estimators": int(ind[0]),
        "max_depth": int(ind[1]),
        "min_samples_leaf": int(ind[2]),
        "min_samples_split": int(ind[3]),
        "max_features": "sqrt" if ind[4] == 0 else "log2",
    }


def get_fitness(best, algorithm: str) -> float:
    """Normaliza o acesso ao fitness entre as duas implementações.
    - ga_handmade: best.fitness            (float simples)
    - ga_deap:     best.fitness.values[0]  (objeto Fitness do DEAP)
    """
    if algorithm == "ga_deap":
        return best.fitness.values[0]
    return float(best.fitness)


def main():
    args = parse_args()
    module_path, label = ALGORITHMS[args.algorithm]

    # Importação dinâmica do módulo escolhido
    import importlib
    ga_module = importlib.import_module(module_path)
    run_ga = ga_module.run_ga

    print("=" * 60)
    print(f"  Algoritmo Genético — Otimização de RandomForest ({label})")
    print("=" * 60)

    print("\n[1/4] Carregando dados...")
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    print(f"      Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    print(f"\n[2/4] Executando AG (população={args.n_pop}, melhoria alvo={args.target_improvement*100:.0f}% sobre {PHASE1_CV_ACCURACY:.4f} → meta:>{PHASE1_CV_ACCURACY*(1+args.target_improvement):.4f})...")
    t0 = time.time()
    try:
        best = run_ga(
            X_train, y_train,
            n_pop=args.n_pop,
            target_improvement=args.target_improvement,
            cx_pb=args.cx_pb,
            mut_pb=args.mut_pb,
            mut_indpb=args.mut_indpb,
        )
    except KeyboardInterrupt:
        print("\n      Execução cancelada pelo usuário.")
        return
    elapsed = time.time() - t0
    print(f"      Concluído em {elapsed:.1f}s")

    print("\n[3/4] Melhor indivíduo encontrado:")
    params = decode_individual(best)
    for k, v in params.items():
        print(f"      {k}: {v}")
    print(f"      Fitness (CV acc): {get_fitness(best, args.algorithm):.4f}")

    print("\n[4/4] Avaliando no conjunto de teste...")
    clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"      Acurácia no teste: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Não diabético", "Diabético"]))
    print("=" * 60)


if __name__ == "__main__":
    main()
