import random
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Tipos do DEAP
# Criamos apenas uma vez para evitar erros ao reimportar o módulo (ex: notebooks).
# FitnessMax: maximiza um único objetivo (weights=(1.0,) — quanto maior, melhor).
# Individual: lista de genes que carrega um objeto FitnessMax associado.
# ---------------------------------------------------------------------------
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore

# ---------------------------------------------------------------------------
# Constantes dos genes
# ---------------------------------------------------------------------------
N_ESTIMATORS_LOW,  N_ESTIMATORS_HIGH  = 20, 60
MAX_DEPTH_OPTIONS = [5, 10, 15, 20, 25]
MIN_SAMPLES_LEAF_LOW, MIN_SAMPLES_LEAF_HIGH = 1, 10
MIN_SAMPLES_SPLIT_LOW, MIN_SAMPLES_SPLIT_HIGH = 2, 20

GENE_LOW  = [N_ESTIMATORS_LOW,  min(MAX_DEPTH_OPTIONS), MIN_SAMPLES_LEAF_LOW,  MIN_SAMPLES_SPLIT_LOW,  0]
GENE_HIGH = [N_ESTIMATORS_HIGH, max(MAX_DEPTH_OPTIONS), MIN_SAMPLES_LEAF_HIGH, MIN_SAMPLES_SPLIT_HIGH, 1]

MUT_INDPB = 0.2   # probabilidade de mutar cada gene individualmente
MUT_PB    = 0.2   # probabilidade de um indivíduo sofrer mutação
CX_PB     = 0.5   # probabilidade de crossover entre dois indivíduos selecionados

# Melhor configuração encontrada pelo GridSearch na Fase 1 (acurácia: 75,32 %):
#   n_estimators=30, max_depth=15, min_samples_leaf=1, min_samples_split=5, max_features='log2'
# Codificação: max_features → 1 (log2)
GRIDSEARCH_SEED = [30, 15, 1, 5, 1]

# Toolbox: registro central de operadores e geradores usados pelo DEAP
toolbox = base.Toolbox()

# ---------------------------------------------------------------------------
# Definição dos genes
# Cada gene representa um hiperparâmetro do RandomForestClassifier.
# A ordem no cromossomo é:
#   [0] n_estimators      — número de árvores          (inteiro, 20–60)
#   [1] max_depth         — profundidade máxima         (categórico: 5,10,15,20,25)
#   [2] min_samples_leaf  — mínimo de amostras por folha (inteiro, 1–10)
#   [3] min_samples_split — mínimo para dividir nó      (inteiro, 2–20)
#   [4] max_features      — critério de features        (binário: 0='sqrt', 1='log2')
# ---------------------------------------------------------------------------

toolbox.register("attr_n_estimators", random.randint, N_ESTIMATORS_LOW, N_ESTIMATORS_HIGH)
toolbox.register("attr_max_depth",    random.choice,  MAX_DEPTH_OPTIONS)
toolbox.register("attr_min_leaf",     random.randint, MIN_SAMPLES_LEAF_LOW, MIN_SAMPLES_LEAF_HIGH)
toolbox.register("attr_min_split",    random.randint, MIN_SAMPLES_SPLIT_LOW, MIN_SAMPLES_SPLIT_HIGH)
toolbox.register("attr_features",     random.randint, 0, 1)

# Cria um indivíduo percorrendo os 5 geradores de genes exatamente 1 vez (n=1)
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,  # type: ignore
    (
        getattr(toolbox, "attr_n_estimators"),
        getattr(toolbox, "attr_max_depth"),
        getattr(toolbox, "attr_min_leaf"),
        getattr(toolbox, "attr_min_split"),
        getattr(toolbox, "attr_features"),
    ),
    n=1,
)
# Cria uma população como lista de N indivíduos gerados pelo toolbox.individual
toolbox.register("population", tools.initRepeat, list, getattr(toolbox, "individual"))


def evaluate(individual, X, y):
    """Função de aptidão: treina um RandomForest com os hiperparâmetros do indivíduo
    e retorna a acurácia média em validação cruzada de 3 folds.
    O DEAP exige que o retorno seja uma tupla, mesmo com um único valor.
    """
    # Decodifica o gene binário de max_features para o valor esperado pelo sklearn
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
    scores = cross_val_score(clf, X, y, cv=3)
    return (scores.mean(),)


def create_seeded_pop(n_pop):
    """Cria a população inicial e injeta o melhor resultado conhecido do GridSearch
    como primeiro indivíduo (elitismo de semente). Isso acelera a convergência
    pois o AG parte de um ponto já sabidamente bom.
    Melhor resultado anterior:
        {'max_depth': 15, 'max_features': 'log2', 'min_samples_leaf': 1,
         'min_samples_split': 5, 'n_estimators': 30}
    Codificado como: [n_estimators=30, max_depth_idx=3→15, leaf=1, split=5, feat=1→log2]
    """
    pop = getattr(toolbox, "population")(n=n_pop)

    # Semente: cromossomo com os hiperparâmetros do melhor resultado do GridSearch
    seed_ind = creator.Individual(GRIDSEARCH_SEED)  # type: ignore
    pop[0] = seed_ind
    return pop


def run_ga(X_train, y_train, n_pop=20, ngen=10):
    """Executa o algoritmo genético e retorna o melhor indivíduo encontrado.

    Parâmetros
    ----------
    X_train : array-like  — features de treino
    y_train : array-like  — rótulos de treino
    n_pop   : int         — tamanho da população (padrão 20)
    ngen    : int         — número de gerações  (padrão 10)

    Retorna
    -------
    hof[0] : Individual — cromossomo com maior fitness já visto
    """
    # Registra a função de aptidão com os dados de treino via closure
    toolbox.register("evaluate", evaluate, X=X_train, y=y_train)

    # Crossover de dois pontos: troca segmentos entre dois pais para gerar filhos
    toolbox.register("mate", tools.cxTwoPoint)

    # Mutação uniforme inteira: altera cada gene com probabilidade indpb,
    # respeitando os limites [low, up] de cada posição do cromossomo
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=GENE_LOW,
        up=GENE_HIGH,
        indpb=MUT_INDPB,
    )

    # Seleção por torneio: escolhe o melhor entre 3 indivíduos sorteados
    toolbox.register("select", tools.selTournament, tournsize=3)

    populacao = create_seeded_pop(n_pop=n_pop)

    # Hall of Fame: armazena o melhor indivíduo de todas as gerações
    hof = tools.HallOfFame(1)

    # eaSimple: loop geracional padrão
    #   cxpb=CX_PB  — probabilidade de crossover entre dois indivíduos selecionados
    #   mutpb=MUT_PB — probabilidade de um indivíduo sofrer mutação
    algorithms.eaSimple(
        populacao, toolbox, cxpb=CX_PB, mutpb=MUT_PB, ngen=ngen, halloffame=hof
    )

    return hof[0]