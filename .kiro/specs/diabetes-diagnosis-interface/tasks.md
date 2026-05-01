# Plano de Implementação: Diabetes Diagnosis Interface

## Visão Geral

Implementar `src/diagnosis_app.py` — uma aplicação Streamlit independente que carrega modelos RandomForest treinados (`.pkl`), recebe dados clínicos do paciente via formulário, executa predição local e envia os dados para a API OpenAI (`gpt-4o-mini`) para obter uma interpretação médica em português.

A implementação segue a arquitetura de 5 componentes definida no design: `ModelLoader`, `FeatureBuilder`, `RFPredictor`, `OpenAIClient` e `DiagnosisUI`.

## Tarefas

- [x] 1. Preparar dependências e configuração do ambiente
  - Adicionar `openai>=1.0.0` ao `requirements.txt`
  - Adicionar a variável `OPENAI_API_KEY=` ao `.env.example` com comentário explicativo
  - _Requisitos: 1.1_

- [x] 2. Criar estrutura base do arquivo `src/diagnosis_app.py`
  - Criar o arquivo `src/diagnosis_app.py` com os imports necessários: `os`, `sys`, `pickle`, `glob`, `dataclasses`, `pandas`, `streamlit`, `sklearn`, `openai`, `dotenv`
  - Adicionar `sys.path.insert(0, os.path.dirname(__file__))` para resolução de imports
  - Chamar `load_dotenv()` no topo do módulo para carregar `.env` automaticamente
  - Definir a constante `FEATURE_COLUMNS` com a lista ordenada dos 8 atributos clínicos
  - Definir os dataclasses `PredictionResult` e `PatientData` conforme especificado no design
  - _Requisitos: 1.1, 3.1_

- [ ] 3. Implementar o componente `ModelLoader`
  - [x] 3.1 Implementar a função `list_available_models(models_dir: str) -> list[tuple[str, str]]`
    - Usar `glob.glob` para varrer `models_dir/*.pkl`
    - Ordenar os arquivos por nome em ordem decrescente (timestamp mais recente primeiro)
    - Adicionar sufixo `"(otimizado)"` se `"optimized"` estiver no nome do arquivo, `"(original)"` caso contrário
    - Retornar lista de tuplas `(label, caminho_absoluto)`
    - _Requisitos: 2.1, 2.3_

  - [x] 3.2 Escrever testes de propriedade para `list_available_models`
    - **Propriedade 5: Ordenação de modelos por nome de arquivo**
    - **Valida: Requisito 2.1**
    - Usar `hypothesis` com `st.lists(st.text(...))` para gerar conjuntos arbitrários de nomes de arquivo `.pkl`
    - Verificar que o resultado está sempre em ordem decrescente de nome

  - [x] 3.3 Escrever testes de propriedade para rotulagem de modelos
    - **Propriedade 6: Rotulagem de modelos**
    - **Valida: Requisito 2.3**
    - Para qualquer nome de arquivo `.pkl`, verificar que o label contém `"(otimizado)"` se e somente se o nome contém `"optimized"`

  - [x] 3.4 Implementar a função `load_model(model_path: str) -> RandomForestClassifier`
    - Decorar com `@st.cache_resource` para cache por sessão Streamlit
    - Usar `pickle.load` para desserializar o modelo
    - Propagar exceções (`FileNotFoundError`, `pickle.UnpicklingError`) para tratamento pela UI
    - _Requisitos: 2.4_

- [x] 4. Implementar o componente `FeatureBuilder`
  - [x] 4.1 Implementar a função `build_feature_dataframe(patient_data: dict[str, float]) -> pd.DataFrame`
    - Verificar que todas as 8 chaves de `FEATURE_COLUMNS` estão presentes em `patient_data`; lançar `ValueError` com lista dos atributos ausentes caso contrário
    - Construir `pd.DataFrame([patient_data])[FEATURE_COLUMNS]` para garantir ordem correta das colunas
    - Converter todos os valores para `float64` com `.astype("float64")`
    - Verificar que o shape retornado é `(1, 8)`
    - _Requisitos: 3.1, 3.2, 3.3, 3.4_

  - [x] 4.2 Escrever testes de propriedade para `build_feature_dataframe`
    - **Propriedade 4: Ordem e shape do DataFrame de features**
    - **Valida: Requisitos 3.2, 3.3**
    - Usar `hypothesis` com `st.floats` para gerar valores válidos para cada um dos 8 atributos
    - Verificar que o DataFrame retornado sempre tem shape `(1, 8)`, colunas na ordem de `FEATURE_COLUMNS` e dtype `float64`

- [x] 5. Implementar o componente `RFPredictor`
  - [x] 5.1 Implementar a função `predict_diabetes(clf, feature_df) -> PredictionResult`
    - Chamar `clf.predict(feature_df)[0]` para obter `prediction` (0 ou 1)
    - Chamar `clf.predict_proba(feature_df)[0]` para obter o array de probabilidades
    - Montar `PredictionResult` com `prediction`, `probability_negative` (índice 0), `probability_positive` (índice 1), `label` (`"Diabético"` se prediction==1, `"Não Diabético"` caso contrário) e `confidence` (`max(prob_neg, prob_pos)`)
    - _Requisitos: 4.1, 4.2, 4.3, 4.4_

  - [x] 5.2 Escrever testes de propriedade para `predict_diabetes`
    - **Propriedade 1: Conservação de probabilidades**
    - **Valida: Requisito 4.2**
    - Usar modelo real carregado de `models/model_diabetes_rf_original.pkl` com entradas geradas por `hypothesis`
    - Verificar que `abs(probability_negative + probability_positive - 1.0) < 1e-6` para qualquer entrada válida

  - [x] 5.3 Escrever testes de propriedade para consistência label/prediction
    - **Propriedade 2: Consistência label/prediction**
    - **Valida: Requisito 4.3**
    - Verificar que `result.label == "Diabético"` se e somente se `result.prediction == 1`

  - [x] 5.4 Escrever testes de propriedade para consistência confidence
    - **Propriedade 3: Consistência confidence**
    - **Valida: Requisito 4.4**
    - Verificar que `result.confidence == max(result.probability_negative, result.probability_positive)`

- [x] 6. Checkpoint — Verificar componentes de dados e predição
  - Garantir que todos os testes passam, perguntar ao usuário se houver dúvidas.

- [x] 7. Implementar o componente `OpenAIClient`
  - [x] 7.1 Implementar a função `build_diagnosis_prompt(patient_data, result) -> list[dict]`
    - Construir `system_message` com instruções de contexto médico, idioma português e aviso de que não substitui consulta médica
    - Construir `user_message` com todos os 8 valores clínicos formatados com unidades, o label da predição e as probabilidades formatadas com uma casa decimal (`:.1f`)
    - Retornar lista de exatamente 2 dicts: `[{"role": "system", "content": ...}, {"role": "user", "content": ...}]`
    - _Requisitos: 5.1, 5.2, 5.3, 5.4_

  - [x] 7.2 Escrever testes de propriedade para `build_diagnosis_prompt`
    - **Propriedade 7: Estrutura do prompt LLM**
    - **Valida: Requisito 5.1**
    - Usar `hypothesis` para gerar `patient_data` e `PredictionResult` arbitrários válidos
    - Verificar que o retorno é sempre uma lista de exatamente 2 dicts com chaves `"role"` e `"content"` não vazias, com roles `"system"` e `"user"` respectivamente

  - [x] 7.3 Escrever testes de propriedade para completude do prompt
    - **Propriedade 8: Completude do prompt — dados clínicos e resultado**
    - **Valida: Requisitos 5.2, 5.3**
    - Verificar que o conteúdo da mensagem `"user"` contém a representação em string de todos os 8 valores de `patient_data`, o label da predição e os valores de probabilidade formatados

  - [x] 7.4 Implementar a função `get_llm_interpretation(patient_data, result, api_key, model="gpt-4o-mini") -> str`
    - Instanciar `openai.OpenAI(api_key=api_key)`
    - Chamar `build_diagnosis_prompt` para obter as mensagens
    - Chamar `client.chat.completions.create(model=model, messages=messages)`
    - Retornar `response.choices[0].message.content`
    - Propagar `openai.OpenAIError` para tratamento pela UI
    - _Requisitos: 5.5, 5.6, 5.7_

- [x] 8. Checkpoint — Verificar componente OpenAIClient
  - Garantir que todos os testes de propriedade do prompt passam, perguntar ao usuário se houver dúvidas.

- [x] 9. Implementar o componente `DiagnosisUI` (orquestração Streamlit)
  - [x] 9.1 Configurar página e carregar variáveis de ambiente
    - Chamar `st.set_page_config(page_title="Diagnóstico de Diabetes", page_icon="🩺", layout="wide")`
    - Ler `OPENAI_API_KEY` do ambiente com `os.getenv("OPENAI_API_KEY")`
    - Exibir `st.error(...)` e interromper o bloco LLM se a chave estiver ausente ou vazia
    - _Requisitos: 1.1, 1.2_

  - [x] 9.2 Implementar sidebar com seleção de modelo
    - Chamar `list_available_models("models/")` para obter a lista de modelos disponíveis
    - Renderizar `st.sidebar.selectbox` com os labels dos modelos disponíveis
    - Exibir `st.warning(...)` e desabilitar o botão "Analisar" (`disabled=True`) se a lista estiver vazia
    - Tratar exceção de modelo corrompido com `st.error(...)` mantendo o seletor ativo
    - _Requisitos: 2.1, 2.2, 2.5, 6.1_

  - [x] 9.3 Implementar formulário de entrada dos dados clínicos
    - Renderizar controles de entrada (sliders ou `st.number_input`) para cada um dos 8 atributos de `FEATURE_COLUMNS` com valores mínimos e máximos derivados dos intervalos do dataset `diabetes_treated.csv` conforme especificado no design
    - Organizar os controles em colunas para melhor aproveitamento do espaço
    - Renderizar botão "Analisar" com `st.button`
    - _Requisitos: 6.2, 6.3_

  - [x] 9.4 Implementar pipeline de predição e exibição de resultados
    - Ao clicar "Analisar": chamar `load_model`, `build_feature_dataframe` e `predict_diabetes` em sequência
    - Exibir `result.label`, `result.probability_positive`, `result.probability_negative` e `result.confidence` usando `st.metric` ou `st.columns`
    - Tratar exceções de carregamento de modelo com `st.error(...)`
    - _Requisitos: 4.1, 6.3, 6.4_

  - [x] 9.5 Implementar chamada LLM e exibição da interpretação
    - Exibir `st.spinner("Consultando OpenAI...")` durante a chamada à API
    - Chamar `get_llm_interpretation(patient_data, result, api_key)` dentro do bloco do spinner
    - Renderizar o texto retornado com `st.markdown(interpretation)`
    - Tratar `openai.OpenAIError` com `st.error(...)` mantendo o resultado da predição RF visível
    - _Requisitos: 5.5, 5.7, 6.5, 6.6_

- [x] 10. Criar arquivo de testes `src/test/test_diagnosis_app.py`
  - [x] 10.1 Escrever testes unitários para `list_available_models`
    - Usar `tmp_path` do pytest para criar diretório temporário com arquivos `.pkl` fictícios
    - Testar ordenação decrescente por nome de arquivo
    - Testar sufixos `"(otimizado)"` e `"(original)"` nos labels
    - Testar retorno de lista vazia quando o diretório não contém `.pkl`
    - _Requisitos: 2.1, 2.3_

  - [x] 10.2 Escrever testes de propriedade para `list_available_models` (Propriedades 5 e 6)
    - Implementar os testes de propriedade especificados nas tarefas 3.2 e 3.3
    - _Requisitos: 2.1, 2.3_

  - [x] 10.3 Escrever testes unitários para `build_feature_dataframe`
    - Testar retorno correto com dicionário completo e válido
    - Testar que `ValueError` é lançado quando atributos obrigatórios estão ausentes
    - Testar que o dtype de todas as colunas é `float64`
    - _Requisitos: 3.2, 3.3, 3.4_

  - [x] 10.4 Escrever testes de propriedade para `build_feature_dataframe` (Propriedade 4)
    - Implementar o teste de propriedade especificado na tarefa 4.2
    - _Requisitos: 3.2, 3.3_

  - [x] 10.5 Escrever testes unitários para `predict_diabetes`
    - Usar `models/model_diabetes_rf_original.pkl` como modelo real para os testes
    - Testar que `prediction` é 0 ou 1
    - Testar que `probability_negative + probability_positive ≈ 1.0` (tolerância `1e-6`)
    - Testar consistência entre `label` e `prediction`
    - Testar que `confidence == max(prob_neg, prob_pos)`
    - _Requisitos: 4.1, 4.2, 4.3, 4.4_

  - [x] 10.6 Escrever testes de propriedade para `predict_diabetes` (Propriedades 1, 2 e 3)
    - Implementar os testes de propriedade especificados nas tarefas 5.2, 5.3 e 5.4
    - _Requisitos: 4.2, 4.3, 4.4_

  - [x] 10.7 Escrever testes unitários para `build_diagnosis_prompt`
    - Testar que o retorno é uma lista de exatamente 2 dicts
    - Testar que os roles são `"system"` e `"user"` respectivamente
    - Testar que o conteúdo da mensagem `"user"` contém todos os 8 valores clínicos
    - Testar que o conteúdo da mensagem `"user"` contém o label da predição e as probabilidades
    - Testar que a mensagem `"system"` menciona português e aviso de não substituição de consulta médica
    - _Requisitos: 5.1, 5.2, 5.3, 5.4_

  - [x] 10.8 Escrever testes de propriedade para `build_diagnosis_prompt` (Propriedades 7 e 8)
    - Implementar os testes de propriedade especificados nas tarefas 7.2 e 7.3
    - _Requisitos: 5.1, 5.2, 5.3_

- [x] 11. Checkpoint final — Garantir que todos os testes passam
  - Executar `pytest src/test/test_diagnosis_app.py -v` e verificar que todos os testes passam
  - Garantir que todos os testes passam, perguntar ao usuário se houver dúvidas.

## Notas

- Tarefas marcadas com `*` são opcionais e podem ser puladas para um MVP mais rápido
- Cada tarefa referencia requisitos específicos para rastreabilidade
- Os checkpoints garantem validação incremental antes de avançar para a próxima fase
- Os testes de propriedade validam invariantes universais; os testes unitários validam exemplos e casos de borda específicos
- O arquivo `src/diagnosis_app.py` é completamente independente de `src/app.py` — não importar nem modificar o app existente
- O modelo `models/model_diabetes_rf_original.pkl` deve ser usado nos testes que requerem um modelo real carregado
- A propriedade 9 (idempotência do cache) não é testável via pytest puro pois depende do runtime do Streamlit; está coberta pela especificação de `@st.cache_resource` na tarefa 3.4
