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
    CX_PB,
    MUT_INDPB,
    MUT_PB,
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


def ind_to_bar_chart(ind: Individual) -> alt.Chart:
    """Barra horizontal com 5 células coloridas, uma por gene do indivíduo."""
    genes = [
        {"gene": "n_estimators",    "abrev": "n_est",  "norm": (ind[0] - 20) / (60 - 20),  "detalhe": str(ind[0])},
        {"gene": "max_depth",       "abrev": "depth",  "norm": (ind[1] - 5)  / (25 - 5),   "detalhe": str(ind[1])},
        {"gene": "min_s_leaf",      "abrev": "leaf",   "norm": (ind[2] - 1)  / (10 - 1),   "detalhe": str(ind[2])},
        {"gene": "min_s_split",     "abrev": "split",  "norm": (ind[3] - 2)  / (20 - 2),   "detalhe": str(ind[3])},
        {"gene": "max_features",    "abrev": "feat",   "norm": float(ind[4]),               "detalhe": "log2" if ind[4] else "sqrt"},
    ]
    df = pd.DataFrame(genes)
    df["linha"] = " "
    order = [g["abrev"] for g in genes]
    return (
        alt.Chart(df)
        .mark_rect(stroke="white", strokeWidth=3)
        .encode(
            x=alt.X("abrev:N", sort=order, axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("linha:N", axis=None),
            color=alt.Color("norm:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), legend=None),
            tooltip=[
                alt.Tooltip("gene:N", title="Hiperparâmetro"),
                alt.Tooltip("detalhe:N", title="Valor"),
            ],
        )
        .properties(height=55)
    )


def make_explore_chart(df: pd.DataFrame) -> alt.VConcatChart:
    """Dispersão dos hiperparâmetros explorados ao longo das gerações, colorido por fitness.

    Usa vconcat em vez de facet para que use_container_width se aplique corretamente
    e a legenda não vaze para fora do expander.
    """
    genes = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split"]
    color = alt.Color(
        "fitness:Q",
        scale=alt.Scale(scheme="viridis"),
        legend=alt.Legend(title="Fitness CV"),
    )
    tooltip = [
        alt.Tooltip("Geração:Q"),
        alt.Tooltip("hiperparâmetro:N", title="Hiperparâmetro"),
        alt.Tooltip("valor:Q", title="Valor"),
        alt.Tooltip("fitness:Q", title="Fitness", format=".4f"),
    ]

    panels = []
    for i, gene in enumerate(genes):
        sub = df[["Geração", "fitness", gene]].rename(columns={gene: "valor"})
        sub = sub.assign(hiperparâmetro=gene)
        show_x = i == len(genes) - 1   # eixo X só no último painel
        panel = (
            alt.Chart(sub)
            .mark_circle(size=50, opacity=0.65)
            .encode(
                x=alt.X(
                    "Geração:Q",
                    axis=alt.Axis(tickMinStep=1, format="d", title="Geração" if show_x else None, labels=show_x),
                ),
                y=alt.Y("valor:Q", title=gene),
                color=color,
                tooltip=tooltip,
            )
            .properties(height=110)
        )
        panels.append(panel)

    return alt.vconcat(*panels, spacing=6).resolve_scale(color="shared")


def run_ga_streaming(
    X_train, y_train,
    n_pop: int,
    max_gen: int,
    target_improvement: float = 0.0,
    cx_pb: float = CX_PB,
    mut_pb: float = MUT_PB,
    mut_indpb: float = MUT_INDPB,
):
    """Gerador: produz estatísticas da população após cada geração."""
    # Meta dinâmica calculada a partir do percentual de melhoria desejado
    target_cv = round(PHASE1_CV_ACCURACY * (1 + target_improvement), 4)

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
        "target_cv": target_cv,
        "done": False,
    }

    for gen in range(1, max_gen + 1):
        population = _gen_loop(population, X_train, y_train, cx_pb=cx_pb, mut_pb=mut_pb, mut_indpb=mut_indpb)

        current_best = max(population, key=lambda ind: ind.fitness)  # type: ignore[arg-type]
        if current_best.fitness > best_ind.fitness:  # type: ignore[operator]
            best_ind = copy.copy(current_best)

        done = round(float(best_ind.fitness), 4) > target_cv  # type: ignore[arg-type]

        yield {
            "gen": gen,
            "population": list(population),
            "best_ind": copy.copy(best_ind),
            "best_fitness": float(best_ind.fitness),  # type: ignore[arg-type]
            "mean_fitness": sum(float(i.fitness) for i in population) / len(population),  # type: ignore[arg-type]
            "target_cv": target_cv,
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
    f"**Meta base (GridSearch Fase 1):** CV 5-fold = **{PHASE1_CV_ACCURACY:.4f}**. "
    "Defina no painel lateral o percentual de melhoria desejado para uma solução sub ótima."
)

# --------------------------------------------------------------------------------
# Sidebar para controle de parâmetros do AG
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parâmetros do AG")
    n_pop = st.slider(
        "Tamanho da população", min_value=10, max_value=50, value=10, step=5
    )
    max_gen = st.slider(
        "Máx. gerações", min_value=20, max_value=300, value=100, step=20
    )
    target_improvement = st.slider(
        "Melhoria alvo (%)", min_value=0.0, max_value=30.0, value=0.0, step=0.5,
        help=f"Percentual acima de {PHASE1_CV_ACCURACY:.4f} (GridSearch Fase 1) que o AG deve atingir.",
    ) / 100.0
    target_cv = round(PHASE1_CV_ACCURACY * (1 + target_improvement), 4)
    st.subheader("🔬 Operadores genéticos")
    cx_pb = st.slider(
        "CX_PB — Prob. cruzamento", min_value=0.0, max_value=1.0, value=float(CX_PB), step=0.05,
        help="Probabilidade de dois indivíduos realizarem cruzamento.",
    )
    mut_pb = st.slider(
        "MUT_PB — Prob. mutação", min_value=0.0, max_value=1.0, value=float(MUT_PB), step=0.05,
        help="Probabilidade de um indivíduo sofrer mutação.",
    )
    mut_indpb = st.slider(
        "MUT_INDPB — Prob. por gene", min_value=0.0, max_value=1.0, value=float(MUT_INDPB), step=0.05,
        help="Probabilidade de cada gene individual ser mutado.",
    )
    st.divider()
    st.info(
        f"**Meta base:** {PHASE1_CV_ACCURACY:.4f} *(GridSearch — Fase 1)*  \n"
        f"**Meta atual:** CV accuracy > **{target_cv:.4f}** (+{target_improvement*100:.1f}%)"
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
ph_best.metric("Melhor CV accuracy", "—", help=f"Meta (Solução sub ótima): > {PHASE1_CV_ACCURACY:.4f}")
ph_mean.metric("CV accuracy médio", "—")
ph_status.metric("Status", "Aguardando")

# --------------------------------------------------------------------------------
# Gráfico + melhor indivíduo
# --------------------------------------------------------------------------------
chart_col, params_col = st.columns([3, 1])
ph_chart = chart_col.empty()
with params_col:
    st.markdown("**🏅 Melhor indivíduo**")
    ph_bar    = st.empty()
    ph_params = st.empty()

st.divider()

# --------------------------------------------------------------------------------
# Tabela da população
# --------------------------------------------------------------------------------
with st.expander("📋 População atual", expanded=True):
    ph_table = st.empty()

with st.expander("🗺️ Exploração do espaço de busca", expanded=False):
    st.caption("Cada ponto é um indivíduo avaliado. Cor = fitness CV (roxo=baixo, amarelo=alto).")
    ph_explore = st.empty()

# --------------------------------------------------------------------------------
# Executa o AG e atualiza as métricas, gráfico, melhor indivíduo e tabela da população a cada geração
# --------------------------------------------------------------------------------
if start:
    X_train, X_test, y_train, y_test = load_data()

    hist_best: list[float] = []
    hist_mean: list[float] = []
    gen_idx:   list[int]   = []
    all_explore_df: list[pd.DataFrame] = []
    last_stats: dict | None = None

    for stats in run_ga_streaming(X_train, y_train, n_pop, max_gen, target_improvement,
                                   cx_pb=cx_pb, mut_pb=mut_pb, mut_indpb=mut_indpb):
        last_stats = stats
        gen      = stats["gen"]
        best_fit = stats["best_fitness"]
        mean_fit = stats["mean_fitness"]
        best_ind = stats["best_ind"]

        gen_idx.append(gen)
        hist_best.append(round(best_fit, 4))
        hist_mean.append(round(mean_fit, 4))

        # Métricas ao vivo
        target_cv = stats["target_cv"]
        delta = round(best_fit - target_cv, 4)
        ph_gen.metric("Geração", gen)
        ph_best.metric("Melhor CV accuracy", f"{best_fit:.4f}", delta=f"{delta:+.4f}",
                       help=f"Meta: > {target_cv:.4f}")
        ph_mean.metric("CV accuracy médio", f"{mean_fit:.4f}")
        with ph_status.container():
            if stats["done"]:
                st.metric("Status", "✅ Meta atingida!")
                st.caption(f"Superou CV accuracy > {target_cv:.4f} (+{target_improvement*100:.0f}%)")
            else:
                st.metric("Status", "🔄 Evoluindo…")

        # Linha de meta no gráfico: regra horizontal em target_cv
        chart_df = (
            pd.DataFrame(
                {"Geração": gen_idx, "Melhor fitness": hist_best, "Fitness médio": hist_mean}
            )
            .melt("Geração", var_name="Métrica", value_name="CV accuracy")
        )
        rule = alt.Chart(pd.DataFrame({"meta": [target_cv]})).mark_rule(
            color="orange", strokeDash=[6, 3], strokeWidth=1.5
        ).encode(y=alt.Y("meta:Q"))
        chart = (
            alt.Chart(chart_df)
            .mark_line()
            .encode(
                x=alt.X("Geração:Q", axis=alt.Axis(tickMinStep=1, format="d")),
                y=alt.Y("CV accuracy:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format=".4f")),
                color="Métrica:N",
            )
        )
        ph_chart.altair_chart((chart + rule).properties(height=280), width="stretch")

        # Parâmetros do melhor indivíduo
        params = decode(best_ind)
        params_df = pd.DataFrame.from_dict(
            {k: [str(v)] for k, v in params.items()}, orient="columns"
        ).T
        params_df.columns = ["Valor"]
        ph_params.dataframe(params_df, width="stretch")
        ph_bar.altair_chart(ind_to_bar_chart(best_ind), width="stretch")

        # Tabela da população com barra de progresso no fitness
        ph_table.dataframe(
            pop_to_df(stats["population"]),
            width="stretch",
            column_config={
                "fitness": st.column_config.ProgressColumn(
                    "Fitness (CV accuracy)",
                    min_value=0.60,
                    max_value=0.92,
                    format="%.4f",
                )
            },
        )

        # Exploração do espaço de busca: acumula todos os indivíduos desta geração
        all_explore_df.append(pd.DataFrame([{
            "Geração": gen,
            "n_estimators": int(p[0]),
            "max_depth": int(p[1]),
            "min_samples_leaf": int(p[2]),
            "min_samples_split": int(p[3]),
            "fitness": round(float(p.fitness), 4) if p.fitness is not None else 0.0,
        } for p in stats["population"]]))
        ph_explore.altair_chart(
            make_explore_chart(pd.concat(all_explore_df, ignore_index=True)),
            width="stretch",
        )

    # --------------------------------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------------------------------
    if last_stats:
        best_ind = last_stats["best_ind"]
        best_fit = last_stats["best_fitness"]
        gen      = last_stats["gen"]

        # Atualiza o status final: considera "meta atingida" se superou ao menos PHASE1_CV_ACCURACY
        if last_stats["done"]:
            with ph_status.container():
                st.metric("Status", "✅ Meta atingida!")
                st.caption(f"Superou CV accuracy > {last_stats['target_cv']:.4f} (+{target_improvement*100:.0f}%)")
        elif best_fit > PHASE1_CV_ACCURACY:
            with ph_status.container():
                st.metric("Status", "📈 Meta alvo não atingida!")
                st.caption(f"Mas superou a meta base ({PHASE1_CV_ACCURACY:.4f})")
        else:
            with ph_status.container():
                st.metric("Status", "❌ Meta não atingida!")
                st.caption(f"Não superou CV accuracy")

        if last_stats["done"]:
            st.success(
                f"🏆 Meta superada na geração **{gen}**! "
                f"CV accuracy: **{best_fit:.4f}** > {last_stats['target_cv']:.4f}"
            )
        else:
            st.warning(
                f"⚠️ Limite de gerações atingido ({max_gen}). "
                f"Melhor CV acc: **{best_fit:.4f}** (meta: {last_stats['target_cv']:.4f})"
            )

        st.subheader("📊 Avaliação no conjunto de teste")
        params = decode(best_ind)
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred   = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        r1, r2 = st.columns(2)
        r1.metric("Acurácia no teste", f"{test_acc:.4f}")
        r2.metric("CV accuracy (treino)", f"{best_fit:.4f}")

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
