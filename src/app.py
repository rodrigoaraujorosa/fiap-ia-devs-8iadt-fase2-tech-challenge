"""Visualização em tempo real do Algoritmo Genético (ga_handmade) com Streamlit.

Execute com:
    streamlit run src/app.py
"""
import copy
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from engine.ga_handmade import (
    PHASE1_CV_ACCURACY,
    Individual,
    _gen_loop,
    create_seeded_pop,
    evaluate,
)

# --------------------------------------------------------------------------------
# Configuração da página
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="AG · Otimização de RandomForest",
    page_icon="🧬",
    layout="wide",
)

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "diabetes_treated.csv"
)

# --------------------------------------------------------------------------------
# Utilitários para formatação, decodificação e exibição de indivíduos e populações
# --------------------------------------------------------------------------------
@st.cache_data
def load_data() -> list:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def decode(ind: Individual) -> dict:
    return {
        "n_estimators": int(ind[0]),
        "max_depth": int(ind[1]),
        "min_samples_leaf": int(ind[2]),
        "min_samples_split": int(ind[3]),
        "max_features": "sqrt" if ind[4] == 0 else "log2",
    }


def pop_to_df(population: list[Individual]) -> pd.DataFrame:
    rows = []
    for i, ind in enumerate(population):
        row: dict = {"#": i + 1}
        row.update(decode(ind))
        row["fitness"] = round(float(ind.fitness), 4) if ind.fitness is not None else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("#").sort_values("fitness", ascending=False)


def run_ga_streaming(X_train, y_train, n_pop: int, max_gen: int):
    """Gerador: produz estatísticas da população após cada geração."""
    population = create_seeded_pop(n_pop)
    for ind in population:
        ind.fitness = evaluate(ind, X_train, y_train)

    best_ind = copy.copy(max(population, key=lambda ind: ind.fitness))  # type: ignore[arg-type]

    yield {
        "gen": 0,
        "population": list(population),
        "best_ind": copy.copy(best_ind),
        "best_fitness": float(best_ind.fitness),  # type: ignore[arg-type]
        "mean_fitness": sum(float(i.fitness) for i in population) / len(population),  # type: ignore[arg-type]
        "done": False,
    }

    for gen in range(1, max_gen + 1):
        population = _gen_loop(population, X_train, y_train)

        current_best = max(population, key=lambda ind: ind.fitness)  # type: ignore[arg-type]
        if current_best.fitness > best_ind.fitness:  # type: ignore[operator]
            best_ind = copy.copy(current_best)

        done = round(float(best_ind.fitness), 4) > PHASE1_CV_ACCURACY  # type: ignore[arg-type]

        yield {
            "gen": gen,
            "population": list(population),
            "best_ind": copy.copy(best_ind),
            "best_fitness": float(best_ind.fitness),  # type: ignore[arg-type]
            "mean_fitness": sum(float(i.fitness) for i in population) / len(population),  # type: ignore[arg-type]
            "done": done,
        }

        if done:
            break

# --------------------------------------------------------------------------------
# Layout estático: título, descrição e sidebar para controle de parâmetros do AG
# --------------------------------------------------------------------------------
st.title("🧬 Algoritmo Genético — Otimização de RandomForest")
st.markdown(
    "Acompanhe em **tempo real** a evolução dos hiperparâmetros do "
    "**RandomForestClassifier** para o dataset de diabetes (Pima Indians).  \n"
    f"**Meta:** superar o CV 5-fold de **{PHASE1_CV_ACCURACY:.4f}** "
    "obtido pelo GridSearch na Fase 1."
)

# --------------------------------------------------------------------------------
# Sidebar para controle de parâmetros do AG
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parâmetros do AG")
    n_pop = st.slider(
        "Tamanho da população", min_value=10, max_value=50, value=20, step=5
    )
    max_gen = st.slider(
        "Máx. gerações", min_value=20, max_value=300, value=100, step=20
    )
    st.divider()
    st.info(
        f"**Meta:** CV acc > **{PHASE1_CV_ACCURACY:.4f}**  \n"
        "*(GridSearch — Fase 1)*"
    )
    start = st.button("▶ Iniciar AG", type="primary", width="stretch")

# --------------------------------------------------------------------------------
# Métricas: geração atual, melhor fitness, fitness médio e status (meta atingida ou evoluindo)
# --------------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
ph_gen    = c1.empty()
ph_best   = c2.empty()
ph_mean   = c3.empty()
ph_status = c4.empty()

ph_gen.metric("Geração", "—")
ph_best.metric("Melhor CV acc", "—", help=f"Meta: > {PHASE1_CV_ACCURACY:.4f}")
ph_mean.metric("CV acc médio", "—")
ph_status.metric("Status", "Aguardando")

# --------------------------------------------------------------------------------
# Gráfico + melhor indivíduo
# --------------------------------------------------------------------------------
chart_col, params_col = st.columns([3, 1])
ph_chart = chart_col.empty()
with params_col:
    st.markdown("**🏅 Melhor indivíduo**")
    ph_params = st.empty()

st.divider()

# --------------------------------------------------------------------------------
# Tabela da população
# --------------------------------------------------------------------------------
with st.expander("📋 População atual", expanded=True):
    ph_table = st.empty()

# --------------------------------------------------------------------------------
# Executa o AG e atualiza as métricas, gráfico, melhor indivíduo e tabela da população a cada geração
# --------------------------------------------------------------------------------
if start:
    X_train, X_test, y_train, y_test = load_data()

    hist_best: list[float] = []
    hist_mean: list[float] = []
    gen_idx:   list[int]   = []
    last_stats: dict | None = None

    for stats in run_ga_streaming(X_train, y_train, n_pop, max_gen):
        last_stats = stats
        gen      = stats["gen"]
        best_fit = stats["best_fitness"]
        mean_fit = stats["mean_fitness"]
        best_ind = stats["best_ind"]

        gen_idx.append(gen)
        hist_best.append(round(best_fit, 4))
        hist_mean.append(round(mean_fit, 4))

        # Métricas ao vivo
        delta = round(best_fit - PHASE1_CV_ACCURACY, 4)
        ph_gen.metric("Geração", gen)
        ph_best.metric("Melhor CV acc", f"{best_fit:.4f}", delta=f"{delta:+.4f}")
        ph_mean.metric("CV acc médio", f"{mean_fit:.4f}")
        ph_status.metric(
            "Status",
            "✅ Meta atingida!" if stats["done"] else "🔄 Evoluindo…",
        )

        # Gráfico de convergência
        chart_df = (
            pd.DataFrame(
                {"Geração": gen_idx, "Melhor fitness": hist_best, "Fitness médio": hist_mean}
            )
            .melt("Geração", var_name="Métrica", value_name="CV acc")
        )
        chart = (
            alt.Chart(chart_df)
            .mark_line()
            .encode(
                x=alt.X("Geração:Q", axis=alt.Axis(tickMinStep=1, format="d")),
                y=alt.Y("CV acc:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format=".4f")),
                color="Métrica:N",
            )
            .properties(height=280)
        )
        ph_chart.altair_chart(chart, width="stretch")

        # Parâmetros do melhor indivíduo
        params = decode(best_ind)
        params_df = pd.DataFrame.from_dict(
            {k: [str(v)] for k, v in params.items()}, orient="columns"
        ).T
        params_df.columns = ["Valor"]
        ph_params.dataframe(params_df, width="stretch")

        # Tabela da população com barra de progresso no fitness
        ph_table.dataframe(
            pop_to_df(stats["population"]),
            width="stretch",
            column_config={
                "fitness": st.column_config.ProgressColumn(
                    "Fitness (CV acc)",
                    min_value=0.60,
                    max_value=0.92,
                    format="%.4f",
                )
            },
        )

    # --------------------------------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------------------------------
    if last_stats:
        best_ind = last_stats["best_ind"]
        best_fit = last_stats["best_fitness"]
        gen      = last_stats["gen"]

        if last_stats["done"]:
            st.success(
                f"🏆 Meta superada na geração **{gen}**! "
                f"CV acc: **{best_fit:.4f}** > {PHASE1_CV_ACCURACY:.4f}"
            )
        else:
            st.warning(
                f"⚠️ Limite de gerações atingido ({max_gen}). "
                f"Melhor CV acc: **{best_fit:.4f}**"
            )

        st.subheader("📊 Avaliação no conjunto de teste")
        params = decode(best_ind)
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred   = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        r1, r2 = st.columns(2)
        r1.metric("Acurácia no teste", f"{test_acc:.4f}")
        r2.metric("CV acc (treino)", f"{best_fit:.4f}")

        report = classification_report(
            y_test,
            y_pred,
            target_names=["Não diabético", "Diabético"],
            output_dict=True,
        )
        report_df = (
            pd.DataFrame(report)
            .T.drop(columns=["support"], errors="ignore")
            .astype(float)
        )
        st.dataframe(
            report_df.style.format("{:.4f}", na_rep="—"),
            width="stretch",
        )
