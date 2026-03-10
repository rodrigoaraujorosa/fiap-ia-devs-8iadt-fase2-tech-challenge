"""Implementação manual (sem DEAP) de um Algoritmo Genético para otimização
de hiperparâmetros do RandomForestClassifier.

Estrutura do cromossomo (5 genes):
  [0] n_estimators      — número de árvores da floresta      (int,  20–60)
  [1] max_depth         — profundidade máxima das árvores    (int,  5|10|15|20|25)
  [2] min_samples_leaf  — mínimo de amostras por folha       (int,  1–10)
  [3] min_samples_split — mínimo de amostras para dividir nó (int,  2–20)
  [4] max_features      — critério de seleção de features    (bin,  0='sqrt' | 1='log2')

Fluxo do AG por geração:
  1. Seleção por torneio  → escolhe os pais mais aptos
  2. Crossover de 2 pontos → combina segmentos dos cromossomos
  3. Mutação uniforme inteira → diversifica a população
  4. Avaliação (CV 5-fold)  → mede a aptidão de cada novo indivíduo

Critério de parada: CV acc > PHASE1_CV_ACCURACY (78,67 %) ou interrupção manual.
"""
import copy
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Constantes e configurações do AG
# ---------------------------------------------------------------------------

# Espaço de busca de cada hiperparâmetro — definem os limites do cromossomo.
N_ESTIMATORS_LOW,  N_ESTIMATORS_HIGH  = 20, 60          # quantidade de árvores na floresta
MAX_DEPTH_OPTIONS = [5, 10, 15, 20, 25]                 # valores possíveis para profundidade máxima
MIN_SAMPLES_LEAF_LOW, MIN_SAMPLES_LEAF_HIGH = 1, 10     # min de amostras em cada folha
MIN_SAMPLES_SPLIT_LOW, MIN_SAMPLES_SPLIT_HIGH = 2, 20   # min de amostras para dividir um nó interno

# Vetores de limites usados pela mutação uniforme: GENE_LOW[i] e GENE_HIGH[i]
# correspondem ao gene i do cromossomo. max_features (gene 4) é binário: 0='sqrt', 1='log2'.
GENE_LOW  = [N_ESTIMATORS_LOW,  min(MAX_DEPTH_OPTIONS), MIN_SAMPLES_LEAF_LOW,  MIN_SAMPLES_SPLIT_LOW,  0]
GENE_HIGH = [N_ESTIMATORS_HIGH, max(MAX_DEPTH_OPTIONS), MIN_SAMPLES_LEAF_HIGH, MIN_SAMPLES_SPLIT_HIGH, 1]

# Probabilidades dos operadores genéticos.
# MUT_INDPB controla a granularidade da mutação: com 0.5, em média metade dos
# genes de um indivíduo são alterados a cada aplicação do operador.
MUT_INDPB = 0.5   # chance de cada gene ser mutado individualmente
MUT_PB    = 0.4   # chance de um indivíduo ser submetido à mutação
CX_PB     = 0.5   # chance de dois indivíduos realizarem crossover

# Penalidade de estabilidade na função de aptidão.
# Subtrair um múltiplo do desvio padrão do CV penaliza soluções instáveis
# (que acertam muito em alguns folds e erram em outros).
# fitness = mean_cv - CV_STD_PENALTY × std_cv
CV_STD_PENALTY = 0.1

# Cromossomo semente: melhor solução encontrada pelo GridSearch na Fase 1
# (CV acc = 78,67 %). Injetado como primeiro indivíduo para acelerar a convergência.
# Parâmetros: n_estimators=30, max_depth=15, min_samples_leaf=1,
#             min_samples_split=5, max_features='log2' (codificado como 1).
GRIDSEARCH_SEED = [30, 15, 1, 5, 1]

# Meta de acurácia CV 5-fold que o AG precisa superar para encerrar.
# Valor obtido pelo GridSearch no Tech Challenge da Fase 1.
PHASE1_CV_ACCURACY = 0.7867


# ---------------------------------------------------------------------------
# Estrutura de indivíduo
# ---------------------------------------------------------------------------
class Individual(list):
    """Cromossomo do AG: lista de 5 genes inteiros representando os hiperparâmetros
    do RandomForestClassifier, com atributo `fitness` associado.

    Herda de `list` para que os operadores genéticos (crossover, mutação)
    possam indexar e fatiar o cromossomo diretamente, seguindo a mesma
    interface da biblioteca DEAP.

    Genes (índice → hiperparâmetro → tipo — intervalo):
        [0] n_estimators      — número de árvores          (int,  20–60)
        [1] max_depth         — profundidade máxima         (int,  5|10|15|20|25)
        [2] min_samples_leaf  — mínimo de amostras/folha    (int,  1–10)
        [3] min_samples_split — mínimo de amostras p/ split (int,  2–20)
        [4] max_features      — critério de features        (bin,  0='sqrt'|1='log2')

    Atributos:
        fitness: float | None — valor da função de aptidão; None indica que
                                o indivíduo ainda precisa ser avaliado.
    """

    def __init__(self, genes):
        super().__init__(genes)
        # fitness começa como None e é preenchido por evaluate().
        # Indivíduos com fitness=None são re-avaliados pelo loop geracional.
        self.fitness: float | None = None

    def __copy__(self):
        # Copia rasa dos genes + preserva o fitness já calculado,
        # evitando re-avaliações desnecessárias no crossover/mutação.
        new_ind = Individual(self)
        new_ind.fitness = self.fitness
        return new_ind


# ---------------------------------------------------------------------------
# Geração de indivíduos e população
# ---------------------------------------------------------------------------
def _random_individual() -> Individual:
    """Gera um indivíduo com genes sorteados uniformemente dentro dos limites
    definidos para cada hiperparâmetro. max_depth é amostrado da lista discreta
    MAX_DEPTH_OPTIONS; os demais genes são inteiros contínuos.
    """
    return Individual([
        random.randint(N_ESTIMATORS_LOW, N_ESTIMATORS_HIGH),
        random.choice(MAX_DEPTH_OPTIONS),           # valor categórico discreto
        random.randint(MIN_SAMPLES_LEAF_LOW, MIN_SAMPLES_LEAF_HIGH),
        random.randint(MIN_SAMPLES_SPLIT_LOW, MIN_SAMPLES_SPLIT_HIGH),
        random.randint(0, 1),                        # max_features binário
    ])


def create_seeded_pop(n_pop: int) -> list[Individual]:
    """Cria a população inicial com elitismo de semente.

    Todos os indivíduos são gerados aleatoriamente, exceto o primeiro, que
    recebe o cromossomo GRIDSEARCH_SEED — a melhor solução encontrada pelo
    GridSearch na Fase 1. Essa estratégia garante que o AG parta de um ponto
    já sabidamente competitivo, reduzindo o tempo até a convergência.

    Semente injetada (GridSearch Fase 1, CV acc = 78,67 %):
        n_estimators=30, max_depth=15, min_samples_leaf=1,
        min_samples_split=5, max_features='log2' → codificado como [30,15,1,5,1]
    """
    pop = [_random_individual() for _ in range(n_pop)]
    # Substitui o primeiro indivíduo pela semente do GridSearch.
    # Os demais continuam aleatórios para garantir diversidade genética.
    pop[0] = Individual(GRIDSEARCH_SEED)
    return pop


# ---------------------------------------------------------------------------
# Função de aptidão
# ---------------------------------------------------------------------------
def evaluate(individual: Individual, X, y) -> float:
    """Calcula o fitness de um indivíduo treinando um RandomForest com seus genes.

    Utiliza validação cruzada estratificada de 5 folds para obter uma estimativa
    robusta da acurácia. O fitness combina média e desvio padrão:

        fitness = mean_cv - CV_STD_PENALTY × std_cv

    A penalidade no desvio padrão (CV_STD_PENALTY = 0.1) desfavorece soluções
    instáveis — aquelas que acertam muito em alguns folds mas erram em outros —
    priorizando modelos consistentes ao longo de toda a base.

    Parâmetros
    ----------
    individual : Individual — cromossomo com os 5 genes de hiperparâmetros
    X          : array-like — features de treino
    y          : array-like — rótulos de treino

    Retorna
    -------
    float — valor do fitness (quanto maior, melhor)
    """
    # Decodifica gene 4: 0 → 'sqrt', 1 → 'log2'
    feat = "sqrt" if individual[4] == 0 else "log2"
    clf = RandomForestClassifier(
        n_estimators=individual[0],
        max_depth=individual[1],
        min_samples_leaf=individual[2],
        min_samples_split=individual[3],
        max_features=feat,
        random_state=42,
        n_jobs=-1,           # usa todos os núcleos disponíveis
    )
    scores = cross_val_score(clf, X, y, cv=5)
    # Penaliza a variância: premia acurácia alta E consistente entre folds
    return scores.mean() - CV_STD_PENALTY * scores.std()


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------
def crossover(ind1: Individual, ind2: Individual) -> tuple[Individual, Individual]:
    """Crossover de dois pontos: recombina material genético entre dois pais.

    Sorteia dois pontos de corte cx1 < cx2 e troca o segmento [cx1, cx2)
    entre os dois indivíduos in-place. O algoritmo garante cx1 < cx2 mesmo
    quando os pontos saem fora de ordem.

    Exemplo com cromossomo de tamanho 5 e cortes em 1 e 3:
        pai1: [A, B, C, D, E]      pai2: [a, b, c, d, e]
              ↕ troca [1:3) ↕
        filho1: [A, b, c, D, E]   filho2: [a, B, C, d, e]

    Os indivíduos são modificados in-place; o fitness é invalidado (→ None)
    pelo chamador (_gen_loop) para forçar re-avaliação.
    """
    size = len(ind1)
    # Sorteia dois índices distintos garantindo cx1 < cx2
    cx1 = random.randint(1, size)
    cx2 = random.randint(1, size - 1)
    if cx2 >= cx1:
        cx2 += 1          # empurra cx2 para frente se coincidiu com cx1
    else:
        cx1, cx2 = cx2, cx1   # ordena para que cx1 < cx2
    # Troca o segmento entre os dois pais
    ind1[cx1:cx2], ind2[cx1:cx2] = ind2[cx1:cx2][:], ind1[cx1:cx2][:]
    return ind1, ind2


def mutate(individual: Individual) -> tuple[Individual]:
    """Mutação uniforme inteira: introduz diversidade alterando genes aleatoriamente.

    Percorre cada gene e, com probabilidade MUT_INDPB, substitui seu valor por
    um inteiro sorteado uniformemente dentro dos limites GENE_LOW[i]–GENE_HIGH[i].
    O gene é modificado in-place; o fitness é invalidado (→ None) pelo chamador.

    Com MUT_INDPB = 0.5 e cromossomo de 5 genes, em média 2–3 genes são
    alterados a cada aplicação do operador.
    """
    for i in range(len(individual)):
        if random.random() < MUT_INDPB:
            # Novo valor respeitando os limites do espaço de busca do gene i
            individual[i] = random.randint(GENE_LOW[i], GENE_HIGH[i])
    return (individual,)


def selection(population: list[Individual], k: int, tournsize: int = 3) -> list[Individual]:
    """Seleção por torneio: pressão seletiva sem eliminar completamente a diversidade.

    Para cada um dos k slots do offspring:
      1. Sorteia `tournsize` candidatos aleatórios da população (sem reposição).
      2. Seleciona o campeão do torneio (maior fitness).

    Vantagem sobre a roleta: não exige que os fitness sejam positivos nem
    normalizados, e o parâmetro `tournsize` controla a pressão seletiva —
    valores maiores favorecem mais os melhores indivíduos.

    Parâmetros
    ----------
    population : list[Individual] — população atual (com fitness calculado)
    k          : int              — número de indivíduos a selecionar
    tournsize  : int              — tamanho de cada torneio (padrão 3)
    """
    chosen = []
    for _ in range(k):
        # Candidatos sorteados sem reposição para evitar torneios triviais
        aspirants = random.sample(population, tournsize)
        chosen.append(max(aspirants, key=lambda ind: ind.fitness))  # type: ignore[arg-type]
    return chosen


# ---------------------------------------------------------------------------
# Loop geracional
# ---------------------------------------------------------------------------
def _gen_loop(
    population: list[Individual],
    X,
    y,
) -> list[Individual]:
    """Executa um ciclo completo de uma geração (equivalente ao eaSimple do DEAP).

    Implementa o fluxo clássico: Seleção → Crossover → Mutação → Avaliação.
    A população original não é alterada; o retorno é uma nova lista de offspring.

    Passos
    ------
    1. Seleção   — gera offspring do mesmo tamanho da população via torneio.
    2. Crossover — aplica crossover de 2 pontos em pares consecutivos com P = CX_PB.
                   O fitness dos filhos modificados é invalidado (→ None).
    3. Mutação   — aplica mutação uniforme em cada indivíduo com P = MUT_PB.
                   O fitness do indivíduo mutado é invalidado (→ None).
    4. Avaliação — reavalia apenas os indivíduos com fitness == None,
                   economizando chamadas ao modelo para os não modificados.

    Parâmetros
    ----------
    population : list[Individual] — população da geração atual
    X          : array-like       — features de treino
    y          : array-like       — rótulos de treino

    Retorna
    -------
    list[Individual] — nova população (offspring avaliado)
    """
    # 1 — Seleção: clona para não modificar os indivíduos da geração anterior
    offspring = [copy.copy(ind) for ind in selection(population, len(population))]

    # 2 — Crossover em pares consecutivos (i-1, i) com passo 2
    for i in range(1, len(offspring), 2):
        if random.random() < CX_PB:
            offspring[i - 1], offspring[i] = crossover(offspring[i - 1], offspring[i])
            # Invalida o fitness: os filhos foram modificados e precisam ser reavaliados
            offspring[i - 1].fitness = None
            offspring[i].fitness = None

    # 3 — Mutação: cada indivíduo é candidato independentemente
    for ind in offspring:
        if random.random() < MUT_PB:
            mutate(ind)
            # Invalida o fitness: genes alterados → resultado do CV anterior não é mais válido
            ind.fitness = None

    # 4 — Avaliação lazy: avalia somente quem foi modificado (fitness == None)
    for ind in offspring:
        if ind.fitness is None:
            ind.fitness = evaluate(ind, X, y)

    return offspring


def run_ga(X_train, y_train, n_pop: int = 20) -> Individual:
    """Ponto de entrada do AG: executa o ciclo evolutivo até atingir a meta de acurácia.

    Fluxo principal
    ---------------
    1. Inicialização — cria população com semente do GridSearch + indivíduos aleatórios.
    2. Avaliação inicial — calcula o fitness de toda a população (1 vez, antes do loop).
    3. Loop geracional — repete _gen_loop() e atualiza o melhor indivíduo global
       até que cv_acc > PHASE1_CV_ACCURACY ou o usuário interrompa com Ctrl+C.

    O melhor indivíduo global (best_ind) é mantido fora da população corrente:
    mesmo que gerações futuras percam diversidade, o melhor resultado já visto
    é sempre preservado (elitismo implícito via cópia).

    Parâmetros
    ----------
    X_train : array-like — features de treino
    y_train : array-like — rótulos de treino
    n_pop   : int        — tamanho da população (padrão 20)

    Retorna
    -------
    Individual — cromossomo com o maior fitness observado em toda a execução
    """
    population = create_seeded_pop(n_pop)

    # Avaliação inicial: todos os indivíduos precisam de fitness antes do 1º torneio
    for ind in population:
        ind.fitness = evaluate(ind, X_train, y_train)

    # Melhor indivíduo global: copiado para não ser sobrescrito pelo loop geracional
    best_ind = copy.copy(max(population, key=lambda ind: ind.fitness))  # type: ignore[arg-type]

    gen = 0
    try:
        while True:
            population = _gen_loop(population, X_train, y_train)
            gen += 1

            # Atualiza o melhor global se a geração atual produziu um indivíduo superior
            current_best = max(population, key=lambda ind: ind.fitness)  # type: ignore[arg-type]
            if current_best.fitness > best_ind.fitness:  # type: ignore[operator]
                best_ind = copy.copy(current_best)

            cv_acc    = round(best_ind.fitness, 4)  # type: ignore[arg-type]
            feat_name = "sqrt" if best_ind[4] == 0 else "log2"
            print(
                f"      Geração {gen:>3} | CV 5-fold: {cv_acc:.4f}"
                f" | n_estimators={best_ind[0]} max_depth={best_ind[1]}"
                f" min_samples_leaf={best_ind[2]} min_samples_split={best_ind[3]} max_features={feat_name}"
            )
            # Critério de parada: meta de acurácia da Fase 1 superada
            if cv_acc > PHASE1_CV_ACCURACY:
                print(f"      Meta superada na geração {gen} (CV: {cv_acc:.4f} > {PHASE1_CV_ACCURACY})")
                break
    except KeyboardInterrupt:
        # Interrupção manual: retorna o melhor indivíduo encontrado até o momento
        print(f"\n      Interrompido na geração {gen}. Retornando melhor indivíduo encontrado até agora.")

    return best_ind
