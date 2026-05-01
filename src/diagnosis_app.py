"""Interface Streamlit para diagnóstico de diabetes com interpretação LLM.

Este módulo implementa a interface de diagnóstico que permite:
- Selecionar um modelo RandomForest treinado (arquivos .pkl em models/);
- Inserir dados clínicos de um paciente via formulário interativo;
- Obter uma predição de diabetes com probabilidades associadas;
- Receber uma interpretação médica contextualizada em português via API OpenAI (gpt-4o-mini).

A interface é completamente independente do dashboard GA existente (src/app.py)
e reutiliza os modelos .pkl gerados pelo pipeline de otimização.
A chave da API OpenAI é carregada do arquivo .env via python-dotenv.

Execute com:
    streamlit run src/diagnosis_app.py
"""
import glob
import os
import pickle
import sys
from dataclasses import dataclass

import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

# Resolução de imports independente do diretório de trabalho
sys.path.insert(0, os.path.dirname(__file__))

# Carrega variáveis de ambiente do .env automaticamente
load_dotenv()

# ---------------------------------------------------------------------------
# Configuração da página Streamlit — deve ser a PRIMEIRA chamada st.* do script
# ---------------------------------------------------------------------------

# Detecta se o módulo está sendo executado dentro do runtime do Streamlit
# (evita erros ao importar o módulo em testes pytest)
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
    _STREAMLIT_RUNTIME = _get_ctx() is not None
except Exception:
    _STREAMLIT_RUNTIME = False

if _STREAMLIT_RUNTIME:
    st.set_page_config(
        page_title="Diagnóstico de Diabetes",
        page_icon="🩺",
        layout="wide",
    )

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Encapsula o resultado da predição do modelo RandomForest.

    Attributes:
        prediction: Classe predita — 0 = não diabético, 1 = diabético.
        probability_negative: Probabilidade da classe negativa P(Outcome=0).
        probability_positive: Probabilidade da classe positiva P(Outcome=1).
        label: Rótulo legível — "Diabético" ou "Não Diabético".
        confidence: Confiança da predição = max(probability_negative, probability_positive).
    """

    prediction: int
    probability_negative: float
    probability_positive: float
    label: str
    confidence: float


@dataclass
class PatientData:
    """Dados clínicos de um paciente para diagnóstico de diabetes.

    Os intervalos de valores refletem os dados observados em diabetes_treated.csv.

    Attributes:
        Pregnancies: Número de gestações (int), intervalo 0–17.
        Glucose: Concentração de glicose plasmática (mg/dL), intervalo 44–199.
        BloodPressure: Pressão arterial diastólica (mm Hg), intervalo 24–122.
        SkinThickness: Espessura da dobra cutânea do tríceps (mm), intervalo 7–99.
        Insulin: Insulina sérica de 2 horas (μU/mL), intervalo 14–846.
        BMI: Índice de massa corporal (kg/m²), intervalo 18.2–67.1.
        DiabetesPedigreeFunction: Função pedigree de diabetes, intervalo 0.078–2.42.
        Age: Idade em anos, intervalo 21–81.
    """

    Pregnancies: float               # int, 0–17
    Glucose: float                   # mg/dL, 44–199
    BloodPressure: float             # mm Hg, 24–122
    SkinThickness: float             # mm, 7–99
    Insulin: float                   # μU/mL, 14–846
    BMI: float                       # kg/m², 18.2–67.1
    DiabetesPedigreeFunction: float  # 0.078–2.42
    Age: float                       # anos, 21–81


# ---------------------------------------------------------------------------
# ModelLoader
# ---------------------------------------------------------------------------


def list_available_models(models_dir: str) -> list[tuple[str, str]]:
    """Retorna lista de tuplas (label_exibição, caminho_absoluto) ordenada do
    mais recente para o mais antigo (ordem decrescente por nome).

    Args:
        models_dir: Caminho para o diretório contendo os arquivos .pkl.

    Returns:
        Lista de tuplas (label, caminho_absoluto) ordenada por nome de arquivo
        em ordem decrescente. Retorna lista vazia se não houver arquivos .pkl.
    """
    pattern = os.path.join(models_dir, "*.pkl")
    pkl_files = glob.glob(pattern)

    if not pkl_files:
        return []

    # Build (label, absolute_path) pairs
    model_list: list[tuple[str, str]] = []
    for file_path in pkl_files:
        filename = os.path.basename(file_path)
        suffix = "(otimizado)" if "optimized" in filename else "(original)"
        label = f"{filename} {suffix}"
        absolute_path = os.path.abspath(file_path)
        model_list.append((label, absolute_path))

    # Sort by filename in descending order (most recent timestamp first)
    model_list.sort(key=lambda t: os.path.basename(t[1]), reverse=True)

    return model_list


# ---------------------------------------------------------------------------
# FeatureBuilder
# ---------------------------------------------------------------------------


def build_feature_dataframe(patient_data: dict[str, float]) -> pd.DataFrame:
    """Converte os dados clínicos do paciente em um DataFrame com o formato
    exato esperado pelo modelo RandomForest.

    Args:
        patient_data: Dicionário mapeando nome do atributo para valor numérico.
                      Deve conter exatamente as 8 chaves definidas em FEATURE_COLUMNS.

    Returns:
        pd.DataFrame com shape (1, 8), colunas na ordem de FEATURE_COLUMNS e
        todos os valores convertidos para dtype float64.

    Raises:
        ValueError: Se um ou mais atributos obrigatórios de FEATURE_COLUMNS
                    estiverem ausentes em patient_data.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in patient_data]
    if missing:
        raise ValueError(
            f"Atributos obrigatórios ausentes em patient_data: {missing}"
        )

    df = pd.DataFrame([patient_data])[FEATURE_COLUMNS].astype("float64")

    assert df.shape == (1, 8), (
        f"Shape inesperado do DataFrame de features: {df.shape}"
    )

    return df


# ---------------------------------------------------------------------------
# RFPredictor
# ---------------------------------------------------------------------------


def predict_diabetes(
    clf: RandomForestClassifier,
    feature_df: pd.DataFrame,
) -> PredictionResult:
    """Executa a predição de diabetes usando o modelo RandomForest carregado.

    Args:
        clf: Instância de RandomForestClassifier treinada e carregada.
        feature_df: DataFrame com shape (1, 8) e colunas na ordem de FEATURE_COLUMNS.

    Returns:
        PredictionResult com prediction (0 ou 1), probabilidades, label e confidence.
    """
    prediction: int = int(clf.predict(feature_df)[0])
    probabilities = clf.predict_proba(feature_df)[0]

    probability_negative: float = float(probabilities[0])
    probability_positive: float = float(probabilities[1])
    label: str = "Diabético" if prediction == 1 else "Não Diabético"
    confidence: float = max(probability_negative, probability_positive)

    return PredictionResult(
        prediction=prediction,
        probability_negative=probability_negative,
        probability_positive=probability_positive,
        label=label,
        confidence=confidence,
    )


@st.cache_resource
def load_model(model_path: str) -> RandomForestClassifier:
    """Carrega e retorna o modelo pickle. Cacheado pelo Streamlit
    para evitar recarregamentos entre re-renders.

    Args:
        model_path: Caminho absoluto ou relativo para o arquivo .pkl.

    Returns:
        Instância de RandomForestClassifier desserializada do arquivo.

    Raises:
        FileNotFoundError: Se o arquivo em model_path não existir.
        pickle.UnpicklingError: Se o arquivo não for um pickle válido.
    """
    with open(model_path, "rb") as f:
        model: RandomForestClassifier = pickle.load(f)
    return model


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------


def build_diagnosis_prompt(
    patient_data: dict[str, float],
    result: PredictionResult,
) -> list[dict]:
    """Constrói a lista de mensagens no formato OpenAI chat para interpretação médica.

    Monta um system message com contexto médico e instruções de idioma, e um
    user message com os dados clínicos do paciente e o resultado da predição RF.

    Args:
        patient_data: Dicionário com os 8 atributos clínicos do paciente.
                      Deve conter as chaves de FEATURE_COLUMNS.
        result: PredictionResult com prediction, probabilidades e label.

    Returns:
        Lista de exatamente 2 dicts no formato OpenAI:
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    system_message = (
        "Você é um assistente médico especializado em diabetes. "
        "Analise os dados clínicos fornecidos e o resultado de um modelo de "
        "machine learning (RandomForest) treinado no dataset Pima Indians Diabetes. "
        "Forneça uma interpretação clara, em português, sobre o risco de diabetes "
        "do paciente. Destaque os fatores de risco mais relevantes com base nos "
        "valores fornecidos. Inclua recomendações gerais de acompanhamento médico. "
        "IMPORTANTE: Deixe claro que esta é uma análise auxiliar e não substitui "
        "consulta médica profissional."
    )

    prediction_label = (
        "POSITIVO para diabetes" if result.prediction == 1 else "NEGATIVO para diabetes"
    )

    user_message = (
        "Dados clínicos do paciente:\n"
        f"- Gestações: {patient_data['Pregnancies']}\n"
        f"- Glicose: {patient_data['Glucose']} mg/dL\n"
        f"- Pressão arterial: {patient_data['BloodPressure']} mm Hg\n"
        f"- Espessura da pele (tríceps): {patient_data['SkinThickness']} mm\n"
        f"- Insulina: {patient_data['Insulin']} μU/mL\n"
        f"- IMC: {patient_data['BMI']} kg/m²\n"
        f"- Função pedigree de diabetes: {patient_data['DiabetesPedigreeFunction']}\n"
        f"- Idade: {patient_data['Age']} anos\n\n"
        f"Resultado do modelo RandomForest: {prediction_label}\n"
        f"Probabilidade de diabetes: {result.probability_positive * 100:.1f}%\n"
        f"Probabilidade de não diabetes: {result.probability_negative * 100:.1f}%\n\n"
        "Por favor, forneça uma interpretação médica detalhada deste resultado."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def get_llm_interpretation(
    patient_data: dict[str, float],
    result: PredictionResult,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Envia o prompt para a API OpenAI e retorna a interpretação médica em texto.

    Args:
        patient_data: Dicionário com os 8 atributos clínicos do paciente.
        result: PredictionResult com prediction, probabilidades e label.
        api_key: Chave de autenticação da API OpenAI.
        model: Identificador do modelo OpenAI a usar. Padrão: "gpt-4o-mini".

    Returns:
        String não vazia com a interpretação médica gerada pela LLM.

    Raises:
        openai.OpenAIError: Em caso de falha na chamada à API OpenAI.
    """
    client = openai.OpenAI(api_key=api_key)
    messages = build_diagnosis_prompt(patient_data, result)
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# DiagnosisUI — Orquestração Streamlit
# ---------------------------------------------------------------------------


def main() -> None:
    """Ponto de entrada principal da interface Streamlit de diagnóstico de diabetes.

    Orquestra todos os componentes:
    - Carrega a chave da API OpenAI do ambiente (9.1)
    - Renderiza sidebar com seleção de modelo (9.2)
    - Renderiza formulário de entrada dos dados clínicos (9.3)
    - Executa pipeline de predição e exibe resultados (9.4)
    - Chama a LLM e exibe a interpretação médica (9.5)
    """
    # -----------------------------------------------------------------------
    # 9.1 — Carregar variáveis de ambiente
    # -----------------------------------------------------------------------
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    api_key_available: bool = bool(api_key)

    # -----------------------------------------------------------------------
    # 9.2 — Sidebar: seleção de modelo
    # -----------------------------------------------------------------------
    st.sidebar.title("⚙️ Configurações")
    st.sidebar.markdown("---")

    available_models = list_available_models("models/")

    analyze_disabled = False

    if not available_models:
        st.sidebar.warning(
            "⚠️ Nenhum modelo encontrado em `models/`. "
            "Execute o AG primeiro para gerar um modelo."
        )
        analyze_disabled = True
        selected_model_path: str = ""
    else:
        model_labels = [label for label, _ in available_models]
        model_paths = {label: path for label, path in available_models}

        selected_label = st.sidebar.selectbox(
            "Selecione o modelo:",
            options=model_labels,
            index=0,
        )
        selected_model_path = model_paths[selected_label]

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Modelos disponíveis em `models/`. "
        "Modelos com `optimized` no nome foram gerados pelo AG."
    )

    # -----------------------------------------------------------------------
    # Cabeçalho principal
    # -----------------------------------------------------------------------
    st.title("🩺 Diagnóstico de Diabetes")
    st.markdown(
        "Preencha os dados clínicos do paciente abaixo e clique em **Analisar** "
        "para obter a predição do modelo RandomForest e uma interpretação médica "
        "gerada por IA."
    )
    st.markdown("---")

    # -----------------------------------------------------------------------
    # 9.3 — Formulário de entrada dos dados clínicos
    # -----------------------------------------------------------------------
    st.subheader("📋 Dados Clínicos do Paciente")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Gestações",
            min_value=0,
            max_value=17,
            value=1,
            step=1,
            help="Número de gestações (0–17)",
        )
        glucose = st.number_input(
            "Glicose (mg/dL)",
            min_value=44,
            max_value=199,
            value=120,
            step=1,
            help="Concentração de glicose plasmática (44–199 mg/dL)",
        )
        blood_pressure = st.number_input(
            "Pressão Arterial (mm Hg)",
            min_value=24,
            max_value=122,
            value=70,
            step=1,
            help="Pressão arterial diastólica (24–122 mm Hg)",
        )
        skin_thickness = st.number_input(
            "Espessura da Pele — Tríceps (mm)",
            min_value=7,
            max_value=99,
            value=20,
            step=1,
            help="Espessura da dobra cutânea do tríceps (7–99 mm)",
        )

    with col2:
        insulin = st.number_input(
            "Insulina (μU/mL)",
            min_value=14,
            max_value=846,
            value=80,
            step=1,
            help="Insulina sérica de 2 horas (14–846 μU/mL)",
        )
        bmi = st.number_input(
            "IMC (kg/m²)",
            min_value=18.2,
            max_value=67.1,
            value=32.0,
            step=0.1,
            format="%.1f",
            help="Índice de massa corporal (18.2–67.1 kg/m²)",
        )
        dpf = st.number_input(
            "Função Pedigree de Diabetes",
            min_value=0.078,
            max_value=2.420,
            value=0.500,
            step=0.001,
            format="%.3f",
            help="Função pedigree de diabetes (0.078–2.42)",
        )
        age = st.number_input(
            "Idade (anos)",
            min_value=21,
            max_value=81,
            value=33,
            step=1,
            help="Idade em anos (21–81)",
        )

    st.markdown("---")

    # Botão "Analisar" — desabilitado se não houver modelos disponíveis
    analyze_clicked = st.button(
        "🔍 Analisar",
        disabled=analyze_disabled,
        type="primary",
    )

    # -----------------------------------------------------------------------
    # 9.4 — Pipeline de predição e exibição de resultados
    # -----------------------------------------------------------------------
    if analyze_clicked:
        patient_data: dict[str, float] = {
            "Pregnancies": float(pregnancies),
            "Glucose": float(glucose),
            "BloodPressure": float(blood_pressure),
            "SkinThickness": float(skin_thickness),
            "Insulin": float(insulin),
            "BMI": float(bmi),
            "DiabetesPedigreeFunction": float(dpf),
            "Age": float(age),
        }

        # Carregar modelo
        try:
            clf = load_model(selected_model_path)
        except (FileNotFoundError, pickle.UnpicklingError, Exception) as e:
            st.error(f"❌ Erro ao carregar o modelo: {e}")
            return

        # Construir DataFrame e executar predição
        try:
            feature_df = build_feature_dataframe(patient_data)
            result = predict_diabetes(clf, feature_df)
        except Exception as e:
            st.error(f"❌ Erro durante a predição: {e}")
            return

        # Exibir resultados da predição
        st.subheader("📊 Resultado da Predição")

        # Ícone de diagnóstico
        diagnosis_icon = "🔴" if result.prediction == 1 else "🟢"
        st.markdown(f"### {diagnosis_icon} {result.label}")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        with res_col1:
            st.metric(
                label="Diagnóstico",
                value=result.label,
            )
        with res_col2:
            st.metric(
                label="Probabilidade Positiva",
                value=f"{result.probability_positive * 100:.1f}%",
                help="Probabilidade de diabetes (Outcome=1)",
            )
        with res_col3:
            st.metric(
                label="Probabilidade Negativa",
                value=f"{result.probability_negative * 100:.1f}%",
                help="Probabilidade de não diabetes (Outcome=0)",
            )
        with res_col4:
            st.metric(
                label="Confiança",
                value=f"{result.confidence * 100:.1f}%",
                help="max(prob_negativa, prob_positiva)",
            )

        st.markdown("---")

        # -------------------------------------------------------------------
        # 9.5 — Chamada LLM e exibição da interpretação
        # -------------------------------------------------------------------
        if not api_key_available:
            st.error(
                "⚠️ `OPENAI_API_KEY` não encontrada. "
                "Adicione a chave ao arquivo `.env` para obter a interpretação médica."
            )
        else:
            try:
                with st.spinner("Consultando OpenAI..."):
                    interpretation = get_llm_interpretation(
                        patient_data, result, api_key
                    )

                st.subheader("🤖 Interpretação Médica (IA)")
                st.markdown(interpretation)

            except openai.OpenAIError as e:
                st.error(
                    f"❌ Erro ao consultar a API OpenAI: {e}\n\n"
                    "O resultado da predição RF acima ainda é válido."
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
elif _STREAMLIT_RUNTIME:
    # Quando executado via `streamlit run`, o módulo é importado diretamente
    # (não como __main__), então chamamos main() aqui também.
    main()
