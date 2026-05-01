"""
Testes para src/diagnosis_app.py
Execução:
    pytest src/test/test_diagnosis_app.py -v
"""
import os
import string
import sys
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Garante que src/ esteja no path independentemente de onde pytest é chamado
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diagnosis_app import FEATURE_COLUMNS, build_feature_dataframe, list_available_models


# ---------------------------------------------------------------------------
# Estratégias Hypothesis
# ---------------------------------------------------------------------------

# Caracteres seguros para nomes de arquivo no filesystem (sem separadores de path,
# sem nulos, sem caracteres reservados no Windows)
_SAFE_CHARS = string.ascii_letters + string.digits + "_-"

filename_strategy = st.text(alphabet=_SAFE_CHARS, min_size=1, max_size=50)


# ---------------------------------------------------------------------------
# Propriedade 5: Ordenação de modelos por nome de arquivo
# Valida: Requisito 2.1
# ---------------------------------------------------------------------------


@given(filenames=st.lists(filename_strategy, min_size=1, max_size=20))
@settings(max_examples=100)
def test_property5_list_available_models_descending_order(filenames):
    """**Validates: Requirements 2.1**

    Para qualquer conjunto não-vazio de nomes de arquivo .pkl gerados
    arbitrariamente, list_available_models() deve retornar os modelos
    ordenados em ordem decrescente de nome de arquivo.
    """
    # Deduplica para evitar colisões de nome no filesystem
    unique_filenames = list(dict.fromkeys(filenames))

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Cria os arquivos .pkl no diretório temporário
        for name in unique_filenames:
            filepath = os.path.join(tmp_dir, f"{name}.pkl")
            open(filepath, "w").close()  # arquivo vazio — só o nome importa

        result = list_available_models(tmp_dir)

        # Extrai os nomes de arquivo dos caminhos retornados
        returned_filenames = [os.path.basename(path) for _, path in result]

        # Verifica que estão em ordem decrescente
        assert returned_filenames == sorted(returned_filenames, reverse=True), (
            f"Esperado ordem decrescente, obtido: {returned_filenames}"
        )


@given(filenames=st.lists(filename_strategy, min_size=0, max_size=0))
@settings(max_examples=1)
def test_property5_empty_directory_returns_empty_list(filenames):
    """**Validates: Requirements 2.1**

    Quando o diretório não contém arquivos .pkl, list_available_models()
    deve retornar lista vazia.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = list_available_models(tmp_dir)
        assert result == []


# ---------------------------------------------------------------------------
# Propriedade 6: Rotulagem de modelos
# Valida: Requisito 2.3
# ---------------------------------------------------------------------------

_LABEL_SAFE_CHARS = string.ascii_letters + string.digits + "_-"

label_stem_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_-",
    ),
    min_size=1,
)


@given(stem=label_stem_strategy)
@settings(max_examples=200)
def test_property6_model_labeling(stem):
    """**Validates: Requirements 2.3**

    Para qualquer nome de arquivo `.pkl`, list_available_models() deve
    adicionar o sufixo `"(otimizado)"` ao label se e somente se o nome
    do arquivo contém `"optimized"`. Caso contrário, o sufixo deve ser
    `"(original)"`.

    Verificação bicondicional:
      - "optimized" in filename  ↔  "(otimizado)" in label
      - "optimized" not in filename  ↔  "(original)" in label
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        filename = f"{stem}.pkl"
        filepath = os.path.join(tmp_dir, filename)
        open(filepath, "w").close()  # arquivo vazio — só o nome importa

        result = list_available_models(tmp_dir)

        assert len(result) == 1, (
            f"Esperado exatamente 1 resultado, obtido {len(result)}"
        )

        label, _ = result[0]
        has_optimized_in_name = "optimized" in filename

        if has_optimized_in_name:
            assert "(otimizado)" in label, (
                f"Esperado '(otimizado)' no label para arquivo '{filename}', "
                f"mas obtido label='{label}'"
            )
            assert "(original)" not in label, (
                f"Label não deve conter '(original)' para arquivo '{filename}', "
                f"mas obtido label='{label}'"
            )
        else:
            assert "(original)" in label, (
                f"Esperado '(original)' no label para arquivo '{filename}', "
                f"mas obtido label='{label}'"
            )
            assert "(otimizado)" not in label, (
                f"Label não deve conter '(otimizado)' para arquivo '{filename}', "
                f"mas obtido label='{label}'"
            )
    finally:
        # Limpeza do diretório temporário
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Propriedade 4: Ordem e shape do DataFrame de features
# Valida: Requisitos 3.2, 3.3
# ---------------------------------------------------------------------------

# Estratégia para valores numéricos válidos (finitos, sem NaN/inf)
_valid_float = st.floats(
    min_value=-1e9,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
)

# Estratégia que gera um dicionário completo com os 8 atributos clínicos
_patient_data_strategy = st.fixed_dictionaries(
    {col: _valid_float for col in FEATURE_COLUMNS}
)


@given(patient_data=_patient_data_strategy)
@settings(max_examples=200)
def test_property4_build_feature_dataframe_shape_and_order(patient_data):
    """**Validates: Requirements 3.2, 3.3**

    Para qualquer dicionário patient_data contendo todos os 8 atributos
    obrigatórios com valores numéricos finitos, build_feature_dataframe()
    deve retornar um DataFrame com:
      - shape exatamente (1, 8)
      - colunas na ordem exata de FEATURE_COLUMNS
      - dtype float64 em todas as colunas
    """
    df = build_feature_dataframe(patient_data)

    # Shape invariant
    assert df.shape == (1, 8), (
        f"Esperado shape (1, 8), obtido {df.shape}"
    )

    # Column order invariant
    assert list(df.columns) == FEATURE_COLUMNS, (
        f"Esperado colunas {FEATURE_COLUMNS}, obtido {list(df.columns)}"
    )

    # dtype invariant
    for col in FEATURE_COLUMNS:
        assert df[col].dtype == "float64", (
            f"Esperado dtype float64 para coluna '{col}', obtido {df[col].dtype}"
        )


@given(
    patient_data=_patient_data_strategy,
    extra_key=st.text(min_size=1, max_size=20).filter(
        lambda k: k not in FEATURE_COLUMNS
    ),
    extra_value=_valid_float,
)
@settings(max_examples=100)
def test_property4_extra_keys_are_ignored(patient_data, extra_key, extra_value):
    """**Validates: Requirements 3.2, 3.3**

    Chaves extras em patient_data além das 8 obrigatórias devem ser ignoradas:
    o DataFrame retornado ainda deve ter shape (1, 8) e apenas as colunas
    de FEATURE_COLUMNS.
    """
    patient_data_with_extra = {**patient_data, extra_key: extra_value}

    df = build_feature_dataframe(patient_data_with_extra)

    assert df.shape == (1, 8), (
        f"Esperado shape (1, 8) mesmo com chave extra, obtido {df.shape}"
    )
    assert list(df.columns) == FEATURE_COLUMNS, (
        f"Colunas extras não devem aparecer no DataFrame: {list(df.columns)}"
    )


@given(
    patient_data=_patient_data_strategy,
    missing_col=st.sampled_from(FEATURE_COLUMNS),
)
@settings(max_examples=100)
def test_property4_missing_key_raises_value_error(patient_data, missing_col):
    """**Validates: Requirement 3.4**

    Se qualquer atributo obrigatório estiver ausente em patient_data,
    build_feature_dataframe() deve lançar ValueError identificando o
    atributo ausente.
    """
    incomplete_data = {k: v for k, v in patient_data.items() if k != missing_col}

    with pytest.raises(ValueError) as exc_info:
        build_feature_dataframe(incomplete_data)

    assert missing_col in str(exc_info.value), (
        f"ValueError deve mencionar o atributo ausente '{missing_col}', "
        f"mas a mensagem foi: '{exc_info.value}'"
    )


# ---------------------------------------------------------------------------
# Estratégias para predict_diabetes (Propriedades 1, 2, 3)
# ---------------------------------------------------------------------------

# Intervalos válidos derivados de diabetes_treated.csv
_patient_valid_strategy = st.fixed_dictionaries(
    {
        "Pregnancies": st.floats(min_value=0.0, max_value=17.0, allow_nan=False, allow_infinity=False),
        "Glucose": st.floats(min_value=44.0, max_value=199.0, allow_nan=False, allow_infinity=False),
        "BloodPressure": st.floats(min_value=24.0, max_value=122.0, allow_nan=False, allow_infinity=False),
        "SkinThickness": st.floats(min_value=7.0, max_value=99.0, allow_nan=False, allow_infinity=False),
        "Insulin": st.floats(min_value=14.0, max_value=846.0, allow_nan=False, allow_infinity=False),
        "BMI": st.floats(min_value=18.2, max_value=67.1, allow_nan=False, allow_infinity=False),
        "DiabetesPedigreeFunction": st.floats(min_value=0.078, max_value=2.42, allow_nan=False, allow_infinity=False),
        "Age": st.floats(min_value=21.0, max_value=81.0, allow_nan=False, allow_infinity=False),
    }
)


def _load_real_model():
    """Carrega o modelo real uma única vez para os testes de propriedade."""
    import pickle

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "model_diabetes_rf_original.pkl"
    )
    model_path = os.path.abspath(model_path)
    with open(model_path, "rb") as f:
        return pickle.load(f)


# Carrega o modelo uma vez no nível de módulo para reutilização nos testes
_REAL_MODEL = _load_real_model()


# ---------------------------------------------------------------------------
# Propriedade 1: Conservação de probabilidades
# Valida: Requisito 4.2
# ---------------------------------------------------------------------------


@given(patient_data=_patient_valid_strategy)
@settings(max_examples=100)
def test_property1_probability_conservation(patient_data):
    """**Validates: Requirements 4.2**

    Para qualquer entrada válida, a soma de probability_negative e
    probability_positive deve ser igual a 1.0 dentro de uma tolerância
    numérica de 1e-6.
    """
    from diagnosis_app import predict_diabetes

    feature_df = build_feature_dataframe(patient_data)
    result = predict_diabetes(_REAL_MODEL, feature_df)

    assert abs(result.probability_negative + result.probability_positive - 1.0) < 1e-6, (
        f"Conservação de probabilidades violada: "
        f"prob_neg={result.probability_negative}, prob_pos={result.probability_positive}, "
        f"soma={result.probability_negative + result.probability_positive}"
    )


# ---------------------------------------------------------------------------
# Propriedade 2: Consistência label/prediction
# Valida: Requisito 4.3
# ---------------------------------------------------------------------------


@given(patient_data=_patient_valid_strategy)
@settings(max_examples=100)
def test_property2_label_prediction_consistency(patient_data):
    """**Validates: Requirements 4.3**

    Para qualquer entrada válida, result.label deve ser "Diabético" se e
    somente se result.prediction == 1; caso contrário deve ser "Não Diabético".
    """
    from diagnosis_app import predict_diabetes

    feature_df = build_feature_dataframe(patient_data)
    result = predict_diabetes(_REAL_MODEL, feature_df)

    if result.prediction == 1:
        assert result.label == "Diabético", (
            f"Esperado label='Diabético' quando prediction=1, obtido label='{result.label}'"
        )
    else:
        assert result.label == "Não Diabético", (
            f"Esperado label='Não Diabético' quando prediction={result.prediction}, "
            f"obtido label='{result.label}'"
        )

    # Verificação bidirecional: label "Diabético" implica prediction == 1
    if result.label == "Diabético":
        assert result.prediction == 1, (
            f"Esperado prediction=1 quando label='Diabético', "
            f"obtido prediction={result.prediction}"
        )
    else:
        assert result.prediction == 0, (
            f"Esperado prediction=0 quando label='Não Diabético', "
            f"obtido prediction={result.prediction}"
        )


# ---------------------------------------------------------------------------
# Propriedade 3: Consistência confidence
# Valida: Requisito 4.4
# ---------------------------------------------------------------------------


@given(patient_data=_patient_valid_strategy)
@settings(max_examples=100)
def test_property3_confidence_consistency(patient_data):
    """**Validates: Requirements 4.4**

    Para qualquer entrada válida, result.confidence deve ser igual a
    max(result.probability_negative, result.probability_positive).
    """
    from diagnosis_app import predict_diabetes

    feature_df = build_feature_dataframe(patient_data)
    result = predict_diabetes(_REAL_MODEL, feature_df)

    expected_confidence = max(result.probability_negative, result.probability_positive)

    assert result.confidence == expected_confidence, (
        f"Consistência de confidence violada: "
        f"confidence={result.confidence}, "
        f"max(prob_neg, prob_pos)={expected_confidence}"
    )


# ---------------------------------------------------------------------------
# Estratégias para build_diagnosis_prompt (Propriedades 7 e 8)
# ---------------------------------------------------------------------------

from diagnosis_app import PredictionResult, build_diagnosis_prompt

# Estratégia para PredictionResult com probabilidades válidas (somam ~1.0)
_prob_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _prediction_result_strategy(draw):
    """Gera um PredictionResult arbitrário com probabilidades consistentes."""
    prob_positive = draw(_prob_strategy)
    prob_negative = 1.0 - prob_positive
    prediction = draw(st.integers(min_value=0, max_value=1))
    label = "Diabético" if prediction == 1 else "Não Diabético"
    confidence = max(prob_negative, prob_positive)
    return PredictionResult(
        prediction=prediction,
        probability_negative=prob_negative,
        probability_positive=prob_positive,
        label=label,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Propriedade 7: Estrutura do prompt LLM
# Valida: Requisito 5.1
# ---------------------------------------------------------------------------


@given(
    patient_data=_patient_data_strategy,
    result=_prediction_result_strategy(),
)
@settings(max_examples=100)
def test_property7_prompt_structure(patient_data, result):
    """**Validates: Requirements 5.1**

    Para qualquer patient_data válido e PredictionResult, build_diagnosis_prompt()
    deve retornar uma lista de exatamente 2 dicts, cada um com chaves "role" e
    "content" não vazias, com roles "system" e "user" respectivamente.
    """
    messages = build_diagnosis_prompt(patient_data, result)

    # Deve ser uma lista de exatamente 2 elementos
    assert isinstance(messages, list), (
        f"Esperado list, obtido {type(messages)}"
    )
    assert len(messages) == 2, (
        f"Esperado exatamente 2 mensagens, obtido {len(messages)}"
    )

    # Cada elemento deve ser um dict com "role" e "content" não vazios
    for i, msg in enumerate(messages):
        assert isinstance(msg, dict), (
            f"Mensagem {i} deve ser dict, obtido {type(msg)}"
        )
        assert "role" in msg, f"Mensagem {i} deve ter chave 'role'"
        assert "content" in msg, f"Mensagem {i} deve ter chave 'content'"
        assert msg["role"], f"Mensagem {i}: 'role' não deve ser vazio"
        assert msg["content"], f"Mensagem {i}: 'content' não deve ser vazio"

    # Primeira mensagem deve ter role "system", segunda "user"
    assert messages[0]["role"] == "system", (
        f"Primeira mensagem deve ter role 'system', obtido '{messages[0]['role']}'"
    )
    assert messages[1]["role"] == "user", (
        f"Segunda mensagem deve ter role 'user', obtido '{messages[1]['role']}'"
    )


# ---------------------------------------------------------------------------
# Propriedade 8: Completude do prompt — dados clínicos e resultado
# Valida: Requisitos 5.2, 5.3
# ---------------------------------------------------------------------------


@given(
    patient_data=_patient_data_strategy,
    result=_prediction_result_strategy(),
)
@settings(max_examples=100)
def test_property8_prompt_completeness(patient_data, result):
    """**Validates: Requirements 5.2, 5.3**

    Para qualquer patient_data e PredictionResult, o conteúdo da mensagem
    "user" deve conter a representação em string de todos os 8 valores
    clínicos de patient_data, o label da predição e os valores de
    probabilidade formatados.
    """
    messages = build_diagnosis_prompt(patient_data, result)
    user_content = messages[1]["content"]

    # Verifica que todos os 8 valores clínicos estão presentes no user message
    for col in FEATURE_COLUMNS:
        value = patient_data[col]
        assert str(value) in user_content, (
            f"Valor de '{col}' ({value}) não encontrado no user message"
        )

    # Verifica que o label da predição está presente
    prediction_label = (
        "POSITIVO para diabetes" if result.prediction == 1 else "NEGATIVO para diabetes"
    )
    assert prediction_label in user_content, (
        f"Label da predição '{prediction_label}' não encontrado no user message"
    )

    # Verifica que as probabilidades formatadas estão presentes
    prob_pos_formatted = f"{result.probability_positive * 100:.1f}%"
    prob_neg_formatted = f"{result.probability_negative * 100:.1f}%"
    assert prob_pos_formatted in user_content, (
        f"Probabilidade positiva formatada '{prob_pos_formatted}' não encontrada no user message"
    )
    assert prob_neg_formatted in user_content, (
        f"Probabilidade negativa formatada '{prob_neg_formatted}' não encontrada no user message"
    )


# ===========================================================================
# UNIT TESTS — Subtask 10.1: list_available_models
# Requisitos: 2.1, 2.3
# ===========================================================================


class TestListAvailableModels:
    """Testes unitários para list_available_models."""

    def test_descending_sort_order(self, tmp_path):
        """Verifica que os modelos são retornados em ordem decrescente por nome de arquivo."""
        filenames = [
            "model_diabetes_rf_optimized_2603182045.pkl",
            "model_diabetes_rf_optimized_2603172000.pkl",
            "model_diabetes_rf_original.pkl",
            "model_diabetes_rf_optimized_2603182032.pkl",
        ]
        for name in filenames:
            (tmp_path / name).write_bytes(b"")

        result = list_available_models(str(tmp_path))

        returned_filenames = [os.path.basename(path) for _, path in result]
        assert returned_filenames == sorted(returned_filenames, reverse=True), (
            f"Esperado ordem decrescente, obtido: {returned_filenames}"
        )

    def test_optimized_suffix_in_label(self, tmp_path):
        """Verifica que arquivos com 'optimized' no nome recebem sufixo '(otimizado)'."""
        (tmp_path / "model_diabetes_rf_optimized_2603182045.pkl").write_bytes(b"")

        result = list_available_models(str(tmp_path))

        assert len(result) == 1
        label, _ = result[0]
        assert "(otimizado)" in label, (
            f"Esperado '(otimizado)' no label, obtido: '{label}'"
        )
        assert "(original)" not in label

    def test_original_suffix_in_label(self, tmp_path):
        """Verifica que arquivos sem 'optimized' no nome recebem sufixo '(original)'."""
        (tmp_path / "model_diabetes_rf_original.pkl").write_bytes(b"")

        result = list_available_models(str(tmp_path))

        assert len(result) == 1
        label, _ = result[0]
        assert "(original)" in label, (
            f"Esperado '(original)' no label, obtido: '{label}'"
        )
        assert "(otimizado)" not in label

    def test_empty_list_when_no_pkl_files(self, tmp_path):
        """Verifica que lista vazia é retornada quando não há arquivos .pkl."""
        # Cria arquivos com outras extensões — não devem ser listados
        (tmp_path / "model.txt").write_bytes(b"")
        (tmp_path / "model.json").write_bytes(b"")

        result = list_available_models(str(tmp_path))

        assert result == [], (
            f"Esperado lista vazia, obtido: {result}"
        )

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """Verifica que lista vazia é retornada para diretório completamente vazio."""
        result = list_available_models(str(tmp_path))
        assert result == []

    def test_returns_list_of_tuples(self, tmp_path):
        """Verifica que o retorno é uma lista de tuplas (label, caminho_absoluto)."""
        (tmp_path / "model_diabetes_rf_original.pkl").write_bytes(b"")

        result = list_available_models(str(tmp_path))

        assert isinstance(result, list)
        assert len(result) == 1
        label, path = result[0]
        assert isinstance(label, str)
        assert isinstance(path, str)
        assert os.path.isabs(path), f"Caminho deve ser absoluto, obtido: '{path}'"

    def test_multiple_models_mixed_labels(self, tmp_path):
        """Verifica ordenação e labels corretos com mix de modelos otimizados e original."""
        filenames = [
            "model_diabetes_rf_optimized_2603182045.pkl",
            "model_diabetes_rf_original.pkl",
        ]
        for name in filenames:
            (tmp_path / name).write_bytes(b"")

        result = list_available_models(str(tmp_path))

        assert len(result) == 2

        # Verifica que a ordem é decrescente por nome de arquivo
        returned_filenames = [os.path.basename(path) for _, path in result]
        assert returned_filenames == sorted(returned_filenames, reverse=True), (
            f"Esperado ordem decrescente, obtido: {returned_filenames}"
        )

        # Verifica que cada modelo tem o label correto
        labels_and_paths = {os.path.basename(path): label for label, path in result}
        assert "(otimizado)" in labels_and_paths["model_diabetes_rf_optimized_2603182045.pkl"]
        assert "(original)" in labels_and_paths["model_diabetes_rf_original.pkl"]


# ===========================================================================
# UNIT TESTS — Subtask 10.3: build_feature_dataframe
# Requisitos: 3.2, 3.3, 3.4
# ===========================================================================


class TestBuildFeatureDataframe:
    """Testes unitários para build_feature_dataframe."""

    _VALID_PATIENT_DATA = {
        "Pregnancies": 2.0,
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 79.8,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50.0,
    }

    def test_correct_return_with_valid_dict(self):
        """Verifica retorno correto com dicionário completo e válido."""
        df = build_feature_dataframe(self._VALID_PATIENT_DATA)

        assert df is not None
        assert df.shape == (1, 8), f"Esperado shape (1, 8), obtido {df.shape}"
        assert list(df.columns) == FEATURE_COLUMNS

    def test_values_are_preserved(self):
        """Verifica que os valores do dicionário são preservados no DataFrame."""
        df = build_feature_dataframe(self._VALID_PATIENT_DATA)

        for col in FEATURE_COLUMNS:
            assert df[col].iloc[0] == float(self._VALID_PATIENT_DATA[col]), (
                f"Valor de '{col}' não preservado: esperado {self._VALID_PATIENT_DATA[col]}, "
                f"obtido {df[col].iloc[0]}"
            )

    def test_all_dtypes_are_float64(self):
        """Verifica que o dtype de todas as colunas é float64."""
        df = build_feature_dataframe(self._VALID_PATIENT_DATA)

        for col in FEATURE_COLUMNS:
            assert df[col].dtype == "float64", (
                f"Esperado dtype float64 para coluna '{col}', obtido {df[col].dtype}"
            )

    def test_raises_value_error_when_single_attribute_missing(self):
        """Verifica que ValueError é lançado quando um atributo obrigatório está ausente."""
        incomplete_data = {k: v for k, v in self._VALID_PATIENT_DATA.items()
                          if k != "Glucose"}

        with pytest.raises(ValueError) as exc_info:
            build_feature_dataframe(incomplete_data)

        assert "Glucose" in str(exc_info.value), (
            f"ValueError deve mencionar 'Glucose', mas a mensagem foi: '{exc_info.value}'"
        )

    def test_raises_value_error_when_multiple_attributes_missing(self):
        """Verifica que ValueError é lançado quando múltiplos atributos estão ausentes."""
        incomplete_data = {
            "Pregnancies": 2.0,
            "Glucose": 148.0,
        }

        with pytest.raises(ValueError) as exc_info:
            build_feature_dataframe(incomplete_data)

        error_msg = str(exc_info.value)
        # Pelo menos um dos atributos ausentes deve ser mencionado
        missing_cols = [col for col in FEATURE_COLUMNS if col not in incomplete_data]
        assert any(col in error_msg for col in missing_cols), (
            f"ValueError deve mencionar atributos ausentes, mas a mensagem foi: '{error_msg}'"
        )

    def test_raises_value_error_when_empty_dict(self):
        """Verifica que ValueError é lançado para dicionário vazio."""
        with pytest.raises(ValueError):
            build_feature_dataframe({})

    def test_extra_keys_are_ignored(self):
        """Verifica que chaves extras no dicionário são ignoradas."""
        data_with_extra = {**self._VALID_PATIENT_DATA, "ExtraFeature": 99.0}

        df = build_feature_dataframe(data_with_extra)

        assert df.shape == (1, 8)
        assert list(df.columns) == FEATURE_COLUMNS
        assert "ExtraFeature" not in df.columns

    def test_integer_values_converted_to_float64(self):
        """Verifica que valores inteiros são convertidos para float64."""
        int_data = {col: int(v) for col, v in self._VALID_PATIENT_DATA.items()}

        df = build_feature_dataframe(int_data)

        for col in FEATURE_COLUMNS:
            assert df[col].dtype == "float64", (
                f"Esperado float64 para '{col}' após conversão de int, obtido {df[col].dtype}"
            )


# ===========================================================================
# UNIT TESTS — Subtask 10.5: predict_diabetes
# Requisitos: 4.1, 4.2, 4.3, 4.4
# ===========================================================================


class TestPredictDiabetes:
    """Testes unitários para predict_diabetes usando modelo real."""

    # Dados de paciente típico para testes determinísticos
    _PATIENT_DATA_POSITIVE = {
        "Pregnancies": 6.0,
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 79.8,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50.0,
    }

    _PATIENT_DATA_NEGATIVE = {
        "Pregnancies": 1.0,
        "Glucose": 85.0,
        "BloodPressure": 66.0,
        "SkinThickness": 29.0,
        "Insulin": 79.8,
        "BMI": 26.6,
        "DiabetesPedigreeFunction": 0.351,
        "Age": 31.0,
    }

    def _get_result(self, patient_data):
        """Helper: constrói DataFrame e executa predição."""
        from diagnosis_app import predict_diabetes
        feature_df = build_feature_dataframe(patient_data)
        return predict_diabetes(_REAL_MODEL, feature_df)

    def test_prediction_is_0_or_1(self):
        """Verifica que prediction é 0 ou 1."""
        result = self._get_result(self._PATIENT_DATA_POSITIVE)
        assert result.prediction in (0, 1), (
            f"Esperado prediction em {{0, 1}}, obtido {result.prediction}"
        )

    def test_probabilities_sum_to_one(self):
        """Verifica que probability_negative + probability_positive ≈ 1.0."""
        result = self._get_result(self._PATIENT_DATA_POSITIVE)
        total = result.probability_negative + result.probability_positive
        assert abs(total - 1.0) < 1e-6, (
            f"Soma das probabilidades deve ser ≈1.0, obtido {total}"
        )

    def test_probabilities_sum_to_one_negative_case(self):
        """Verifica conservação de probabilidades para caso negativo."""
        result = self._get_result(self._PATIENT_DATA_NEGATIVE)
        total = result.probability_negative + result.probability_positive
        assert abs(total - 1.0) < 1e-6, (
            f"Soma das probabilidades deve ser ≈1.0, obtido {total}"
        )

    def test_label_consistency_with_prediction_diabetico(self):
        """Verifica que label='Diabético' quando prediction=1."""
        from diagnosis_app import predict_diabetes, PredictionResult
        feature_df = build_feature_dataframe(self._PATIENT_DATA_POSITIVE)
        result = predict_diabetes(_REAL_MODEL, feature_df)

        if result.prediction == 1:
            assert result.label == "Diabético", (
                f"Esperado label='Diabético' quando prediction=1, obtido '{result.label}'"
            )
        else:
            assert result.label == "Não Diabético", (
                f"Esperado label='Não Diabético' quando prediction=0, obtido '{result.label}'"
            )

    def test_label_consistency_with_prediction_nao_diabetico(self):
        """Verifica que label='Não Diabético' quando prediction=0."""
        from diagnosis_app import predict_diabetes
        feature_df = build_feature_dataframe(self._PATIENT_DATA_NEGATIVE)
        result = predict_diabetes(_REAL_MODEL, feature_df)

        if result.prediction == 0:
            assert result.label == "Não Diabético", (
                f"Esperado label='Não Diabético' quando prediction=0, obtido '{result.label}'"
            )
        else:
            assert result.label == "Diabético"

    def test_confidence_equals_max_probability(self):
        """Verifica que confidence == max(prob_neg, prob_pos)."""
        result = self._get_result(self._PATIENT_DATA_POSITIVE)
        expected = max(result.probability_negative, result.probability_positive)
        assert result.confidence == expected, (
            f"Esperado confidence={expected}, obtido {result.confidence}"
        )

    def test_confidence_equals_max_probability_negative_case(self):
        """Verifica confidence para caso negativo."""
        result = self._get_result(self._PATIENT_DATA_NEGATIVE)
        expected = max(result.probability_negative, result.probability_positive)
        assert result.confidence == expected

    def test_probabilities_are_between_0_and_1(self):
        """Verifica que as probabilidades estão no intervalo [0, 1]."""
        result = self._get_result(self._PATIENT_DATA_POSITIVE)
        assert 0.0 <= result.probability_negative <= 1.0
        assert 0.0 <= result.probability_positive <= 1.0

    def test_confidence_is_between_0_and_1(self):
        """Verifica que confidence está no intervalo [0, 1]."""
        result = self._get_result(self._PATIENT_DATA_POSITIVE)
        assert 0.0 <= result.confidence <= 1.0

    def test_returns_prediction_result_instance(self):
        """Verifica que o retorno é uma instância de PredictionResult."""
        from diagnosis_app import predict_diabetes, PredictionResult
        feature_df = build_feature_dataframe(self._PATIENT_DATA_POSITIVE)
        result = predict_diabetes(_REAL_MODEL, feature_df)
        assert isinstance(result, PredictionResult)


# ===========================================================================
# UNIT TESTS — Subtask 10.7: build_diagnosis_prompt
# Requisitos: 5.1, 5.2, 5.3, 5.4
# ===========================================================================


class TestBuildDiagnosisPrompt:
    """Testes unitários para build_diagnosis_prompt."""

    _PATIENT_DATA = {
        "Pregnancies": 2.0,
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 79.8,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50.0,
    }

    _RESULT_POSITIVE = PredictionResult(
        prediction=1,
        probability_negative=0.23,
        probability_positive=0.77,
        label="Diabético",
        confidence=0.77,
    )

    _RESULT_NEGATIVE = PredictionResult(
        prediction=0,
        probability_negative=0.82,
        probability_positive=0.18,
        label="Não Diabético",
        confidence=0.82,
    )

    def test_returns_list_of_exactly_2_dicts(self):
        """Verifica que o retorno é uma lista de exatamente 2 dicts."""
        messages = build_diagnosis_prompt(self._PATIENT_DATA, self._RESULT_POSITIVE)

        assert isinstance(messages, list), f"Esperado list, obtido {type(messages)}"
        assert len(messages) == 2, f"Esperado 2 mensagens, obtido {len(messages)}"
        for i, msg in enumerate(messages):
            assert isinstance(msg, dict), f"Mensagem {i} deve ser dict, obtido {type(msg)}"

    def test_roles_are_system_and_user(self):
        """Verifica que os roles são 'system' e 'user' respectivamente."""
        messages = build_diagnosis_prompt(self._PATIENT_DATA, self._RESULT_POSITIVE)

        assert messages[0]["role"] == "system", (
            f"Primeira mensagem deve ter role 'system', obtido '{messages[0]['role']}'"
        )
        assert messages[1]["role"] == "user", (
            f"Segunda mensagem deve ter role 'user', obtido '{messages[1]['role']}'"
        )

    def test_user_message_contains_all_8_clinical_values(self):
        """Verifica que o conteúdo da mensagem 'user' contém todos os 8 valores clínicos."""
        messages = build_diagnosis_prompt(self._PATIENT_DATA, self._RESULT_POSITIVE)
        user_content = messages[1]["content"]

        for col in FEATURE_COLUMNS:
            value = self._PATIENT_DATA[col]
            assert str(value) in user_content, (
                f"Valor de '{col}' ({value}) não encontrado no user message"
            )

    def test_user_message_contains_prediction_label(self):
        """Verifica que o conteúdo da mensagem 'user' contém o label da predição."""
        messages = build_diagnosis_prompt(self._PATIENT_DATA, self._RESULT_POSITIVE)
        user_content = messages[1]["content"]

        assert "POSITIVO para diabetes" in user_content, (
            "Label da predição positiva não encontrado no user message"
        )

    def test_user_message_contains_prediction_label_negative(self):
        """Verifica que o label negativo aparece corretamente no user message."""
        messages = build_diagnosis_prompt(self._PATIENT_DATA, self._RESULT_NEGATIVE)
        user_content = messages[1]["content"]

        assert "NEGATIVO para diabetes" in user_content, (
            "Label da predição negativa não encontrado no user message"
        )

    def test_user_message_contains_probabilities(self):
        """Verifica que o conteúdo da mensagem 'user' contém as probabilidades formatadas."""
        messages = build_diagnosis_prompt(self._PATIENT_DATA, self._RESULT_POSITIVE)
        user_content = messages[1]["content"]

        prob_pos_formatted = f"{self._RESULT_POSITIVE.probability_positive * 100:.1f}%"
        prob_neg_formatted = f"{self._RESULT_POSITIVE.probability_negative * 100:.1f}%"

        assert prob_pos_formatted in user_content, (
            f"Probabilidade positiva '{prob_pos_formatted}' não encontrada no user message"
        )
        assert prob_neg_formatted in user_content, (
            f"Probabilidade negativa '{prob_neg_formatted}' não encontrada no user message"
        )

    def test_system_message_mentions_portuguese(self):
        """Verifica que a mensagem 'system' menciona o idioma português."""
        messages = build_diagnosis_prompt(self._PATIENT_DATA, self._RESULT_POSITIVE)
        system_content = messages[0]["content"]

        assert "português" in system_content.lower() or "portuguese" in system_content.lower(), (
            "Mensagem system deve mencionar português"
        )

    def test_system_message_mentions_disclaimer(self):
        """Verifica que a mensagem 'system' menciona que não substitui consulta médica."""
        messages = build_diagnosis_prompt(self._PATIENT_DATA, self._RESULT_POSITIVE)
        system_content = messages[0]["content"]

        # Verifica presença de aviso sobre não substituição de consulta médica
        disclaimer_keywords = ["não substitui", "consulta médica", "profissional"]
        assert any(kw in system_content.lower() for kw in disclaimer_keywords), (
            f"Mensagem system deve mencionar que não substitui consulta médica. "
            f"Conteúdo: '{system_content[:200]}...'"
        )

    def test_both_messages_have_non_empty_content(self):
        """Verifica que ambas as mensagens têm conteúdo não vazio."""
        messages = build_diagnosis_prompt(self._PATIENT_DATA, self._RESULT_POSITIVE)

        assert messages[0]["content"], "Mensagem 'system' não deve ter conteúdo vazio"
        assert messages[1]["content"], "Mensagem 'user' não deve ter conteúdo vazio"

    def test_messages_have_role_and_content_keys(self):
        """Verifica que cada mensagem tem as chaves 'role' e 'content'."""
        messages = build_diagnosis_prompt(self._PATIENT_DATA, self._RESULT_POSITIVE)

        for i, msg in enumerate(messages):
            assert "role" in msg, f"Mensagem {i} deve ter chave 'role'"
            assert "content" in msg, f"Mensagem {i} deve ter chave 'content'"
