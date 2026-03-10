
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "diabetes_treated.csv")


def evaluate_hyperparameters(n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features, X, y):
    """Função de aptidão: treina um RandomForest com os hiperparâmetros fornecidos
    e retorna a acurácia média em validação cruzada de 3 folds.
    O DEAP exige que o retorno seja uma tupla, mesmo com um único valor.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )
    scores = cross_val_score(clf, X, y, cv=5)
    return (scores.mean(),)


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Ajuste os hiperparâmetros abaixo para validar
    # {'max_depth': 15, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 30}
    n_estimators     = 30
    max_depth        = 15
    min_samples_leaf = 1
    min_samples_split = 5
    max_features     = "log2"

    acc, = evaluate_hyperparameters(
        n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features,
        X_train, y_train,
    )
    print(f"Acurácia média (CV 3-fold): {acc:.4f}")


if __name__ == "__main__":
    main()
