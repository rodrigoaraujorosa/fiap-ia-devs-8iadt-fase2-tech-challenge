# Diagrama do Algoritmo Genético

Critério de parada: `cv_acc > target_cv`, onde `target_cv = PHASE1_CV_ACCURACY × (1 + target_improvement)`.

```mermaid
flowchart TD
    A([Início]) --> B[Gerar população inicial<br/>n_pop indivíduos aleatórios]
    B --> C[Injetar semente GridSearch<br/>melhor resultado Fase 1]
    C --> D[Avaliar fitness de todos<br/>CV 5-fold: mean − 0.1·std]

    D --> E{fitness_melhor<br/>> target_cv?}
    E -- Sim --> Z([Retornar melhor indivíduo])

    E -- Não --> F[Seleção por Torneio<br/>tournsize=3, k=n_pop]
    F --> G[Crossover de dois pontos<br/>cx_pb por par]
    G --> H[Mutação uniforme inteira<br/>mut_pb · mut_indpb por gene]
    H --> I[Avaliar filhos modificados<br/>fitness == None]
    I --> J[Atualizar melhor indivíduo<br/>Hall of Fame]
    J --> K[Exibir geração atual<br/>log de progresso]
    K --> E
```

## Cromossomo

Cada indivíduo é uma lista de 5 genes que codificam os hiperparâmetros do modelo:

| Gene | Hiperparâmetro       | Tipo       | Intervalo / Opções        |
|------|----------------------|------------|---------------------------|
| [0]  | `n_estimators`       | inteiro    | 20 – 60                   |
| [1]  | `max_depth`          | categórico | 5, 10, 15, 20, 25         |
| [2]  | `min_samples_leaf`   | inteiro    | 1 – 10                    |
| [3]  | `min_samples_split`  | inteiro    | 2 – 20                    |
| [4]  | `max_features`       | binário    | 0 = `sqrt`, 1 = `log2`    |

## Função de Fitness

$$\text{fitness} = \overline{\text{CV}} - 0.1 \times \sigma_{\text{CV}}$$

Penalizar o desvio padrão premia modelos acurados e estáveis entre os folds.

## Parâmetros Padrão do AG

| Parâmetro    | Valor | Descrição                                          |
|--------------|-------|----------------------------------------------------|
| `n_pop`              | 10       | Tamanho da população (padrão)                                                    |
| `cx_pb`              | 0.7      | Probabilidade de crossover entre dois indivíduos *(configurável)*               |
| `mut_pb`             | 0.6      | Probabilidade de um indivíduo sofrer mutação *(configurável)*                   |
| `mut_indpb`          | 0.5      | Probabilidade de mutar cada gene individualmente *(configurável)*               |
| `tournsize`          | 3        | Candidatos por torneio na seleção                                               |
| `target_improvement` | 0.0      | Melhoria % desejada sobre a meta base (0.0 = apenas superar 0.7867) *(configurável)* |
| Meta base (CV)       | 0.7867   | Acurácia CV 5-fold do GridSearch da Fase 1                                      |
| `target_cv`          | dinâmica | `PHASE1_CV_ACCURACY × (1 + target_improvement)` — critério de parada            |
