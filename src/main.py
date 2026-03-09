import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Garante que o módulo engine seja encontrado ao rodar direto de src/ ou da raiz
sys.path.insert(0, os.path.dirname(__file__))

from engine.ga_deap import run_ga

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "diabetes_treated.csv"
)


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def decode_individual(ind):
    max_depth_options = [None, 5, 10, 15, 20, 25]
    depth_idx = max(0, min(int(ind[1]), len(max_depth_options) - 1))
    return {
        "n_estimators": int(ind[0]),
        "max_depth": max_depth_options[depth_idx],
        "min_samples_leaf": int(ind[2]),
        "min_samples_split": int(ind[3]),
        "max_features": "sqrt" if ind[4] == 0 else "log2",
    }


def main():
    print("=" * 60)
    print("  Algoritmo Genético — Otimização de RandomForest")
    print("=" * 60)

    print("\n[1/4] Carregando dados...")
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    print(f"      Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    print("\n[2/4] Executando AG (população=20, gerações=10)...")
    t0 = time.time()
    best = run_ga(X_train, y_train, n_pop=20, ngen=10)
    elapsed = time.time() - t0
    print(f"      Concluído em {elapsed:.1f}s")

    print("\n[3/4] Melhor indivíduo encontrado:")
    params = decode_individual(best)
    for k, v in params.items():
        print(f"      {k}: {v}")
    print(f"      Fitness (CV acc): {best.fitness.values[0]:.4f}")

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
