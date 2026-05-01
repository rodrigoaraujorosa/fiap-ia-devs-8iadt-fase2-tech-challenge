# Documento de Requisitos

## Introdução

Este documento descreve os requisitos funcionais e de qualidade para a **Interface de Diagnóstico de Diabetes** — uma aplicação Streamlit (`src/diagnosis_app.py`) que permite ao usuário inserir dados clínicos de um paciente, obter uma predição de diabetes a partir de um modelo RandomForest treinado (arquivos `.pkl` em `models/`) e receber uma interpretação médica contextualizada em português gerada pela API OpenAI (modelo `gpt-4o-mini`).

A interface é independente do dashboard GA existente (`src/app.py`) e reutiliza os modelos já gerados pelo pipeline de otimização. A chave da API OpenAI é carregada do arquivo `.env` via `python-dotenv`.

---

## Glossário

- **DiagnosisApp**: A aplicação Streamlit principal definida em `src/diagnosis_app.py`, responsável por orquestrar todos os componentes.
- **ModelLoader**: Componente responsável por descobrir, listar e carregar modelos `.pkl` do diretório `models/`.
- **FeatureBuilder**: Componente responsável por converter os dados do paciente em um `pd.DataFrame` com as colunas exatas esperadas pelo modelo.
- **RFPredictor**: Componente responsável por executar a predição e obter probabilidades usando o `RandomForestClassifier` carregado.
- **OpenAIClient**: Componente responsável por construir o prompt, enviar para a API OpenAI e retornar a interpretação médica em texto.
- **PredictionResult**: Estrutura de dados que encapsula o resultado da predição: `prediction`, `probability_negative`, `probability_positive`, `label` e `confidence`.
- **PatientData**: Dicionário com os 8 atributos clínicos do paciente: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
- **FEATURE_COLUMNS**: Lista ordenada dos 8 nomes de colunas esperados pelo modelo, na mesma ordem do treinamento.
- **OPENAI_API_KEY**: Chave de autenticação da API OpenAI, carregada do arquivo `.env` via `python-dotenv`.

---

## Requisitos

### Requisito 1: Carregamento de Configuração e Chave de API

**História de Usuário:** Como desenvolvedor, quero que a aplicação carregue automaticamente a chave da API OpenAI do arquivo `.env`, para que eu não precise configurar variáveis de ambiente manualmente a cada execução.

#### Critérios de Aceitação

1. O **DiagnosisApp** DEVE carregar a `OPENAI_API_KEY` do arquivo `.env` via `python-dotenv` durante a inicialização da aplicação.
2. SE a `OPENAI_API_KEY` estiver ausente ou vazia no ambiente, ENTÃO o **DiagnosisApp** DEVE exibir uma mensagem de erro indicando a chave ausente e desabilitar o bloco de interpretação LLM.

---

### Requisito 2: Descoberta e Carregamento de Modelos

**História de Usuário:** Como usuário, quero selecionar qual modelo RandomForest usar para o diagnóstico, para que eu possa comparar resultados entre diferentes modelos gerados pelo pipeline de otimização.

#### Critérios de Aceitação

1. O **ModelLoader** DEVE varrer o diretório `models/` usando glob e retornar todos os arquivos `.pkl` como uma lista de tuplas `(label, caminho)` ordenada pelo nome do arquivo em ordem decrescente (mais recente primeiro).
2. QUANDO o diretório `models/` estiver vazio ou não existir, ENTÃO o **DiagnosisApp** DEVE exibir uma mensagem de aviso e desabilitar o botão "Analisar".
3. O **ModelLoader** DEVE adicionar o sufixo `"(otimizado)"` ao label de exibição de qualquer arquivo `.pkl` cujo nome contenha `"optimized"`, e `"(original)"` para todos os demais arquivos `.pkl`.
4. QUANDO um caminho de modelo válido for fornecido, o **ModelLoader** DEVE carregar e retornar a instância do `RandomForestClassifier` usando `pickle.load`, armazenando o resultado em cache com `@st.cache_resource` para evitar desserializações repetidas.
5. SE um arquivo `.pkl` estiver corrompido ou for incompatível com a versão atual do `scikit-learn`, ENTÃO o **DiagnosisApp** DEVE exibir uma mensagem de erro e manter o seletor de modelos ativo para que o usuário possa escolher outro modelo.

---

### Requisito 3: Construção do DataFrame de Atributos

**História de Usuário:** Como sistema, quero converter os dados clínicos inseridos pelo usuário em um DataFrame com o formato exato esperado pelo modelo, para que a predição seja executada corretamente.

#### Critérios de Aceitação

1. O **FeatureBuilder** DEVE definir `FEATURE_COLUMNS` como a lista ordenada `["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]`.
2. QUANDO um dicionário `patient_data` contendo todos os 8 atributos obrigatórios for fornecido, o **FeatureBuilder** DEVE retornar um `pd.DataFrame` com shape `(1, 8)` e colunas na ordem exata definida por `FEATURE_COLUMNS`.
3. O **FeatureBuilder** DEVE retornar um DataFrame onde todos os valores das colunas possuam dtype `float64`.
4. SE o dicionário `patient_data` estiver faltando um ou mais atributos obrigatórios de `FEATURE_COLUMNS`, ENTÃO o **FeatureBuilder** DEVE lançar um `ValueError` identificando os atributos ausentes.

---

### Requisito 4: Predição com RandomForest

**História de Usuário:** Como usuário, quero obter uma predição de diabetes com probabilidades associadas, para que eu possa entender o grau de confiança do modelo no diagnóstico.

#### Critérios de Aceitação

1. QUANDO um DataFrame de atributos válido e um `RandomForestClassifier` carregado forem fornecidos, o **RFPredictor** DEVE retornar um `PredictionResult` com `prediction` igual a `0` ou `1`.
2. O **RFPredictor** DEVE garantir que `probability_negative + probability_positive` seja igual a `1.0` dentro de uma tolerância numérica de `1e-6` para qualquer entrada válida.
3. O **RFPredictor** DEVE definir `result.label` como `"Diabético"` quando `result.prediction` for `1`, e como `"Não Diabético"` quando `result.prediction` for `0`.
4. O **RFPredictor** DEVE definir `result.confidence` como `max(result.probability_negative, result.probability_positive)` para qualquer entrada válida.

---

### Requisito 5: Construção do Prompt e Interpretação LLM

**História de Usuário:** Como usuário, quero receber uma interpretação médica em português do resultado da predição gerada por uma LLM, para que eu possa compreender os fatores de risco e obter recomendações de acompanhamento.

#### Critérios de Aceitação

1. QUANDO um dicionário `patient_data` e um `PredictionResult` forem fornecidos, o **OpenAIClient** DEVE construir uma lista de mensagens contendo exatamente 2 entradas: uma com `role` `"system"` e outra com `role` `"user"`, ambas com `content` não vazio.
2. O **OpenAIClient** DEVE incluir todos os 8 valores de atributos clínicos de `patient_data` no conteúdo da mensagem `"user"`.
3. O **OpenAIClient** DEVE incluir o label da predição e os valores de probabilidade formatados (`probability_positive` e `probability_negative`) no conteúdo da mensagem `"user"`.
4. O **OpenAIClient** DEVE incluir uma mensagem de sistema instruindo o modelo a responder em português, atuar como assistente médico especializado em diabetes e incluir um aviso de que a análise não substitui consulta médica profissional.
5. QUANDO a chamada à API OpenAI for bem-sucedida, o **OpenAIClient** DEVE retornar uma string não vazia contendo a interpretação médica.
6. O **OpenAIClient** DEVE chamar `openai.chat.completions.create()` usando o modelo `"gpt-4o-mini"` por padrão.
7. SE a chamada à API OpenAI lançar um `openai.OpenAIError`, ENTÃO o **DiagnosisApp** DEVE exibir uma mensagem de erro mantendo o resultado da predição RF visível ao usuário.

---

### Requisito 6: Interface do Usuário (DiagnosisUI)

**História de Usuário:** Como usuário, quero interagir com uma interface Streamlit clara e responsiva para inserir dados do paciente e visualizar os resultados do diagnóstico, para que o processo de análise seja intuitivo e eficiente.

#### Critérios de Aceitação

1. O **DiagnosisApp** DEVE renderizar uma barra lateral contendo um seletor (selectbox) para escolher entre os modelos `.pkl` disponíveis listados pelo ModelLoader.
2. O **DiagnosisApp** DEVE renderizar controles de entrada (sliders ou campos numéricos) para cada um dos 8 atributos clínicos em `FEATURE_COLUMNS`, com valores mínimos e máximos derivados dos intervalos observados no dataset `diabetes_treated.csv`.
3. QUANDO o usuário clicar no botão "Analisar", o **DiagnosisApp** DEVE executar o pipeline completo em sequência: carregar modelo → construir DataFrame de atributos → realizar predição → obter interpretação LLM → exibir resultados.
4. O **DiagnosisApp** DEVE exibir o label de `prediction`, os valores de `probability_positive`, `probability_negative` e `confidence` do `PredictionResult` na seção de resultados.
5. O **DiagnosisApp** DEVE renderizar o texto de interpretação da LLM usando `st.markdown` para suportar saída formatada.
6. ENQUANTO a chamada à API OpenAI estiver em andamento, o **DiagnosisApp** DEVE exibir um indicador de carregamento (spinner) com uma mensagem descritiva para fornecer feedback visual ao usuário.
