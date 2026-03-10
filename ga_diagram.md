# Diagrama do Algoritmo Genético

Fluxo completo da execução do AG para otimização de hiperparâmetros do `RandomForestClassifier`.

```mermaid
flowchart TD
    A([Início]) --> B[Gerar população inicial<br/>n_pop indivíduos aleatórios]
    B --> C[Injetar semente GridSearch<br/>melhor resultado Fase 1]
    C --> D[Avaliar fitness de todos<br/>CV 5-fold: mean − 0.1·std]

    D --> E{fitness_melhor<br/>> 0.7867?}
    E -- Sim --> Z([Retornar melhor indivíduo])

    E -- Não --> F[Seleção por Torneio<br/>tournsize=3, k=n_pop]
    F --> G[Crossover de dois pontos<br/>CX_PB=0.5 por par]
    G --> H[Mutação uniforme inteira<br/>MUT_PB=0.4 · MUT_INDPB=0.5]
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

## Parâmetros do AG

| Parâmetro    | Valor | Descrição                                          |
|--------------|-------|----------------------------------------------------|
| `n_pop`      | 20    | Tamanho da população (padrão)                      |
| `CX_PB`      | 0.5   | Probabilidade de crossover entre dois indivíduos   |
| `MUT_PB`     | 0.4   | Probabilidade de um indivíduo sofrer mutação       |
| `MUT_INDPB`  | 0.5   | Probabilidade de mutar cada gene individualmente   |
| `tournsize`  | 3     | Candidatos por torneio na seleção                  |
| Meta CV      | 0.7867 | Acurácia CV 5-fold do GridSearch da Fase 1        |
