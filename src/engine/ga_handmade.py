import copy
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Constantes e configurações do AG
# ---------------------------------------------------------------------------
N_ESTIMATORS_LOW,  N_ESTIMATORS_HIGH  = 20, 60          # intervalo do número de árvores na floresta
MAX_DEPTH_OPTIONS = [5, 10, 15, 20, 25]                 # profundidades máximas permitidas para as árvores
MIN_SAMPLES_LEAF_LOW, MIN_SAMPLES_LEAF_HIGH = 1, 10     # intervalo do mínimo de amostras por folha
MIN_SAMPLES_SPLIT_LOW, MIN_SAMPLES_SPLIT_HIGH = 2, 20   # intervalo do mínimo de amostras para dividir um nó

# Limites inferior e superior de cada gene do cromossomo, usados pela mutação uniforme.
# Ordem: [n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features]
# max_features é binário: 0 = 'sqrt', 1 = 'log2'
GENE_LOW  = [N_ESTIMATORS_LOW,  min(MAX_DEPTH_OPTIONS), MIN_SAMPLES_LEAF_LOW,  MIN_SAMPLES_SPLIT_LOW,  0]
GENE_HIGH = [N_ESTIMATORS_HIGH, max(MAX_DEPTH_OPTIONS), MIN_SAMPLES_LEAF_HIGH, MIN_SAMPLES_SPLIT_HIGH, 1]

MUT_INDPB = 0.5   # probabilidade de mutar cada gene individualmente
MUT_PB    = 0.4   # probabilidade de um indivíduo sofrer mutação
CX_PB     = 0.5   # probabilidade de crossover entre dois indivíduos selecionados

# Peso da penalidade pelo desvio padrão do CV:
# fitness = mean_cv - CV_STD_PENALTY * std_cv
CV_STD_PENALTY = 0.1

# Melhor configuração encontrada pelo GridSearch no Tech Challenge da Fase 1 (CV acurácia: 78,67%):
#   n_estimators=30, max_depth=15, min_samples_leaf=1, min_samples_split=5, max_features='log2'
# Codificação: max_features → 1 (log2)
GRIDSEARCH_SEED = [30, 15, 1, 5, 1]

# Meta de acurácia CV 5-fold que o AG deve superar para encerrar
PHASE1_CV_ACCURACY = 0.7867


# ---------------------------------------------------------------------------
# Estrutura de indivíduo
# ---------------------------------------------------------------------------
class Individual(list):
    """Lista de genes com atributo fitness associado.

    A ordem dos genes no cromossomo é:
        [0] n_estimators      — número de árvores          (inteiro, 20–60)
        [1] max_depth         — profundidade máxima         (categórico: 5,10,15,20,25)
        [2] min_samples_leaf  — mínimo de amostras por folha (inteiro, 1–10)
        [3] min_samples_split — mínimo para dividir nó      (inteiro, 2–20)
        [4] max_features      — critério de features        (binário: 0='sqrt', 1='log2')
    """

    def __init__(self, genes):
        super().__init__(genes)
        self.fitness: float | None = None

    def __copy__(self):
        new_ind = Individual(self)
        new_ind.fitness = self.fitness
        return new_ind


# ---------------------------------------------------------------------------
# Geração de indivíduos e população
# ---------------------------------------------------------------------------
def _random_individual() -> Individual:
    return Individual([
        random.randint(N_ESTIMATORS_LOW, N_ESTIMATORS_HIGH),
        random.choice(MAX_DEPTH_OPTIONS),
        random.randint(MIN_SAMPLES_LEAF_LOW, MIN_SAMPLES_LEAF_HIGH),
        random.randint(MIN_SAMPLES_SPLIT_LOW, MIN_SAMPLES_SPLIT_HIGH),
        random.randint(0, 1),
    ])


def create_seeded_pop(n_pop: int) -> list[Individual]:
    """Cria a população inicial e injeta o melhor resultado conhecido do GridSearch
    como primeiro indivíduo (elitismo de semente). Isso acelera a convergência
    pois o AG parte de um ponto já sabidamente bom.
    Melhor resultado anterior obtido no GridSearch do Tech Challenge da Fase 1:
        {'max_depth': 15, 'max_features': 'log2', 'min_samples_leaf': 1,
         'min_samples_split': 5, 'n_estimators': 30}
    Codificado como: [n_estimators=30, max_depth_idx=2→15, min_samples_leaf=1, min_samples_split=5, max_features=1→log2]
    """
    pop = [_random_individual() for _ in range(n_pop)]
    pop[0] = Individual(GRIDSEARCH_SEED)
    return pop


# ---------------------------------------------------------------------------
# Função de aptidão
# ---------------------------------------------------------------------------
def evaluate(individual: Individual, X, y) -> float:
    """Função de aptidão: treina um RandomForest com os hiperparâmetros do indivíduo
    e retorna  mean_cv - CV_STD_PENALTY * std_cv  (validação cruzada de 5 folds).
    Combinar média e desvio padrão premia modelos acurados e estáveis entre os folds,
    evitando soluções que acertam em alguns folds mas erram muito em outros.
    """
    feat = "sqrt" if individual[4] == 0 else "log2"
    clf = RandomForestClassifier(
        n_estimators=individual[0],
        max_depth=individual[1],
        min_samples_leaf=individual[2],
        min_samples_split=individual[3],
        max_features=feat,
        random_state=42,
        n_jobs=-1,
    )
    scores = cross_val_score(clf, X, y, cv=5)
    return scores.mean() - CV_STD_PENALTY * scores.std()


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------
def crossover(ind1: Individual, ind2: Individual) -> tuple[Individual, Individual]:
    """Crossover de dois pontos: troca o segmento [cx1, cx2) entre os dois indivíduos
    (in-place).
    """
    size = len(ind1)
    cx1 = random.randint(1, size)
    cx2 = random.randint(1, size - 1)
    if cx2 >= cx1:
        cx2 += 1
    else:
        cx1, cx2 = cx2, cx1
    ind1[cx1:cx2], ind2[cx1:cx2] = ind2[cx1:cx2][:], ind1[cx1:cx2][:]
    return ind1, ind2


def mutate(individual: Individual) -> tuple[Individual]:
    """Mutação uniforme inteira: altera cada gene com probabilidade MUT_INDPB,
    respeitando os limites GENE_LOW / GENE_HIGH de cada posição.
    """
    for i in range(len(individual)):
        if random.random() < MUT_INDPB:
            individual[i] = random.randint(GENE_LOW[i], GENE_HIGH[i])
    return (individual,)


def selection(population: list[Individual], k: int, tournsize: int = 3) -> list[Individual]:
    """Seleção por torneio: escolhe k indivíduos, cada um sendo o melhor de
    tournsize competidores sorteados aleatoriamente sem reposição.
    """
    chosen = []
    for _ in range(k):
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
    """Executa uma geração do eaSimple:
    1. Seleciona offspring por torneio (mesmo tamanho da população).
    2. Aplica crossover em pares consecutivos com probabilidade CX_PB.
    3. Aplica mutação em cada indivíduo com probabilidade MUT_PB.
    4. Avalia apenas os indivíduos com fitness inválido (invalidade = None).
    5. Retorna o novo conjunto de offspring como nova população.
    """
    # 1 — Seleção: clona os indivíduos escolhidos para não modificar os originais
    offspring = [copy.copy(ind) for ind in selection(population, len(population))]

    # 2 — Crossover em pares consecutivos
    for i in range(1, len(offspring), 2):
        if random.random() < CX_PB:
            offspring[i - 1], offspring[i] = crossover(offspring[i - 1], offspring[i])
            offspring[i - 1].fitness = None
            offspring[i].fitness = None

    # 3 — Mutação individual
    for ind in offspring:
        if random.random() < MUT_PB:
            mutate(ind)
            ind.fitness = None

    # 4 — Avaliação dos indivíduos modificados
    for ind in offspring:
        if ind.fitness is None:
            ind.fitness = evaluate(ind, X, y)

    return offspring


def run_ga(X_train, y_train, n_pop: int = 20) -> Individual:
    """Executa o algoritmo genético geração a geração até superar PHASE1_CV_ACCURACY em CV 5-fold.

    Parâmetros
    ----------
    X_train : array-like  — features de treino
    y_train : array-like  — rótulos de treino
    n_pop   : int         — tamanho da população (padrão 20)

    Retorna
    -------
    best_ind : Individual — cromossomo com maior fitness já visto
    """
    population = create_seeded_pop(n_pop)

    # Avaliação inicial de toda a população
    for ind in population:
        ind.fitness = evaluate(ind, X_train, y_train)

    best_ind = copy.copy(max(population, key=lambda ind: ind.fitness))  # type: ignore[arg-type]

    gen = 0
    try:
        while True:
            population = _gen_loop(population, X_train, y_train)
            gen += 1

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
            if cv_acc > PHASE1_CV_ACCURACY:
                print(f"      Meta superada na geração {gen} (CV: {cv_acc:.4f} > {PHASE1_CV_ACCURACY})")
                break
    except KeyboardInterrupt:
        print(f"\n      Interrompido na geração {gen}. Retornando melhor indivíduo encontrado até agora.")

    return best_ind
