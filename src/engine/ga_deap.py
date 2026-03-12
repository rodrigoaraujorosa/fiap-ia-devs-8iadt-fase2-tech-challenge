import random
import time
from deap import base, creator, tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from .ga_logger import GALogger

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
# Constantes e configurações do AG
# ---------------------------------------------------------------------------
N_ESTIMATORS_LOW,  N_ESTIMATORS_HIGH  = 20, 60          # intervalo do número de árvores na floresta
MAX_DEPTH_OPTIONS = [5, 10, 15, 20, 25]                 # profundidades máximas permitidas para as árvores
MIN_SAMPLES_LEAF_LOW, MIN_SAMPLES_LEAF_HIGH = 1, 10     # intervalo do mínimo de amostras por folha
MIN_SAMPLES_SPLIT_LOW, MIN_SAMPLES_SPLIT_HIGH = 2, 20   # intervalo do mínimo de amostras para dividir um nó

# Limites inferior e superior de cada gene do indivíduo, usados pelo mutUniformInt.
# Ordem: [n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features]
# max_features é binário: 0 = 'sqrt', 1 = 'log2'
GENE_LOW  = [N_ESTIMATORS_LOW,  min(MAX_DEPTH_OPTIONS), MIN_SAMPLES_LEAF_LOW,  MIN_SAMPLES_SPLIT_LOW,  0]
GENE_HIGH = [N_ESTIMATORS_HIGH, max(MAX_DEPTH_OPTIONS), MIN_SAMPLES_LEAF_HIGH, MIN_SAMPLES_SPLIT_HIGH, 1]

MUT_INDPB = 0.5   # probabilidade de mutar cada gene individualmente (↑ para maior diversidade nos genes)
MUT_PB    = 0.6   # probabilidade de um indivíduo sofrer mutação    (↑ para mais exploração por geração)
CX_PB     = 0.7   # probabilidade de crossover entre dois indivíduos selecionados

# Peso da penalidade pelo desvio padrão do CV:
# fitness = mean_cv - CV_STD_PENALTY * std_cv
# Penaliza soluções instáveis entre os folds sem dominar a métrica principal.
CV_STD_PENALTY = 0.1

# Melhor configuração encontrada pelo GridSearch no Tech Challenge da Fase 1 (CV acurácia: 78,67%; acurácia: 75,32 %):
#   n_estimators=30, max_depth=15, min_samples_leaf=1, min_samples_split=5, max_features='log2'
# Codificação: max_features → 1 (log2)
GRIDSEARCH_SEED = [30, 15, 1, 5, 1]

# Meta de acurácia CV 5-fold que o AG deve superar para encerrar
# (média obtida pelo GridSearch no Tech Challenge da Fase 1)
PHASE1_CV_ACCURACY = 0.7867

# Toolbox: registro central de operadores e geradores usados pelo DEAP
toolbox = base.Toolbox()

# ---------------------------------------------------------------------------
# Definição dos genes
# Cada gene representa um hiperparâmetro do RandomForestClassifier.
# A ordem no indivíduo é:
#   [0] n_estimators      — número de árvores          (inteiro, 20–60)
#   [1] max_depth         — profundidade máxima         (categórico: 5,10,15,20,25)
#   [2] min_samples_leaf  — mínimo de amostras por folha (inteiro, 1–10)
#   [3] min_samples_split — mínimo de amostras para dividir um nó interno (inteiro, 2–20)
#   [4] max_features      — critério de features (variáveis)       (binário: 0='sqrt', 1='log2')
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


def _cx_effective(ind1, ind2):
    """Crossover de 2 pontos garantindo troca em pelo menos um gene divergente.

    Substitui o ``tools.cxTwoPoint`` do DEAP para evitar crossovers nulos:
    quando os cortes aleatórios caem numa região onde os genes já são idênticos,
    o resultado é genéticamente inútil. Esta função ancora um dos cortes numa
    posição onde os pais diferem, garantindo diversidade real nos filhos.
    """
    size = len(ind1)
    diff = [i for i in range(size) if ind1[i] != ind2[i]]
    if not diff:
        return ind1, ind2  # pais idênticos: nada a trocar
    pivot = random.choice(diff)
    cx1 = random.randint(0, pivot)
    cx2 = random.randint(pivot + 1, size)
    ind1[cx1:cx2], ind2[cx1:cx2] = list(ind2[cx1:cx2]), list(ind1[cx1:cx2])
    return ind1, ind2


def evaluate(individual, X, y):
    """Função de aptidão: treina um RandomForest com os hiperparâmetros do indivíduo
    e retorna  mean_cv - CV_STD_PENALTY * std_cv  (validação cruzada de 5 folds).
    Combinar média e desvio padrão premia modelos acurados E estáveis entre os folds,
    evitando soluções que acertam em alguns folds mas erram muito em outros.
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
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return (scores.mean() - CV_STD_PENALTY * scores.std(),)


def create_seeded_pop(n_pop):
    """Cria a população inicial e injeta o melhor resultado conhecido do GridSearch
    como primeiro indivíduo (elitismo de semente). Isso acelera a convergência
    pois o AG parte de um ponto já sabidamente bom.
    Melhor resultado anterior obtido no GridSearch do Tech Challenge da Fase 1:
        {'max_depth': 15, 'max_features': 'log2', 'min_samples_leaf': 1,
         'min_samples_split': 5, 'n_estimators': 30}
    Codificado como: [n_estimators=30, max_depth_idx=2→15, min_samples_leaf=1, min_samples_split=5, max_features=1→log2]
    """
    pop = getattr(toolbox, "population")(n=n_pop)

    # Semente: indivíduo com os hiperparâmetros do melhor resultado do GridSearch
    seed_ind = creator.Individual(GRIDSEARCH_SEED)  # type: ignore
    pop[0] = seed_ind
    return pop


def run_ga(
    X_train,
    y_train,
    n_pop=10,
    target_improvement: float = 0.000,
    cx_pb: float = CX_PB,
    mut_pb: float = MUT_PB,
    mut_indpb: float = MUT_INDPB,
):
    """Executa o algoritmo genético geração a geração até superar a meta de acurácia dinâmica.

    Parâmetros
    ----------
    X_train            : array-like — features de treino
    y_train            : array-like — rótulos de treino
    n_pop              : int        — tamanho da população (padrão 10)
    target_improvement : float      — melhoria percentual desejada sobre PHASE1_CV_ACCURACY
                                      ex.: 0.10 = meta 10 % acima de 0.7867 → 0.8654
    cx_pb              : float      — probabilidade de crossover (padrão CX_PB = 0.7)
    mut_pb             : float      — probabilidade de mutação por indivíduo (padrão MUT_PB = 0.6)
    mut_indpb          : float      — probabilidade de mutação por gene (padrão MUT_INDPB = 0.5)

    Retorna
    -------
    hof[0] : Individual — indivíduo com maior fitness já visto
    """
    print(f"      Iniciando AG com população={n_pop}, target_improvement={target_improvement:.3f}, "
          f"cx_pb={cx_pb:.2f}, mut_pb={mut_pb:.2f}, mut_indpb={mut_indpb:.2f}")
    # Meta dinâmica: PHASE1_CV_ACCURACY elevada pelo percentual solicitado
    target_cv = round(PHASE1_CV_ACCURACY * (1 + target_improvement), 4)

    _ga_logger = GALogger("ga_deap")
    _ga_logger.log_run_start(
        n_pop=n_pop, cx_pb=cx_pb, mut_pb=mut_pb, mut_indpb=mut_indpb,
        target_improvement=target_improvement, target_cv=target_cv,
    )
    _t0 = time.time()

    # Registra a função de aptidão com os dados de treino via closure
    toolbox.register("evaluate", evaluate, X=X_train, y=y_train)

    # Crossover de dois pontos com garantia de troca em gene divergente
    toolbox.register("mate", _cx_effective)

    # Mutação uniforme inteira: altera cada gene com probabilidade indpb,
    # respeitando os limites [low, up] de cada posição do indivíduo
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=GENE_LOW,
        up=GENE_HIGH,
        indpb=mut_indpb,
    )

    # Seleção por torneio: escolhe o melhor entre 3 indivíduos sorteados
    toolbox.register("select", tools.selTournament, tournsize=3)

    populacao = create_seeded_pop(n_pop=n_pop)

    # Hall of Fame: armazena o melhor indivíduo de todas as gerações
    hof = tools.HallOfFame(1)

    # Avaliação inicial
    _t_eval0 = time.perf_counter()
    for ind in populacao:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind) # type: ignore
    _ga_logger._phase_times["avaliacao"]  = time.perf_counter() - _t_eval0
    _ga_logger._phase_counts["avaliacao"] = 1
    hof.update(populacao)

    # Log da população inicial (geração 0)
    _ga_logger.log_generation_start(0)
    _ga_logger.log_population(0, populacao)
    fits0 = [ind.fitness.values[0] for ind in populacao]
    _ga_logger.log_generation_stats(
        0, hof[0].fitness.values[0], sum(fits0) / len(fits0), target_cv, hof[0]
    )

    gen = 0
    try:
        while True:
            gen += 1
            _ga_logger.log_generation_start(gen)

            # Seleção
            _t_sel = time.perf_counter()
            selected = toolbox.select(populacao, len(populacao)) # type: ignore
            _ga_logger._phase_times["selecao"]  = _ga_logger._phase_times.get("selecao", 0.0)  + (time.perf_counter() - _t_sel)
            _ga_logger._phase_counts["selecao"] = _ga_logger._phase_counts.get("selecao", 0)   + 1
            _ga_logger.log_selection(gen, n_selected=len(selected), tournsize=3)
            _ga_logger.log_selected_individuals(gen, selected)
            offspring = [toolbox.clone(ind) for ind in selected] # type: ignore

            # Crossover (estilo varAnd)
            _t_cx = time.perf_counter()
            for i in range(1, len(offspring), 2):
                before1 = list(offspring[i - 1])
                before2 = list(offspring[i])
                # Pais idênticos: crossover não geraria diversidade — pula o operador
                identical = before1 == before2
                if (not identical) and random.random() < cx_pb:
                    offspring[i - 1], offspring[i] = toolbox.mate( # type: ignore
                        offspring[i - 1], offspring[i]
                    )
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values
                    _ga_logger.log_crossover(
                        gen, (i - 1, i), before1, before2,
                        list(offspring[i - 1]), list(offspring[i]), True,
                    )
                else:
                    _ga_logger.log_crossover(
                        gen, (i - 1, i), before1, before2,
                        list(offspring[i - 1]), list(offspring[i]), False,
                    )
            _ga_logger._phase_times["crossover"]  = _ga_logger._phase_times.get("crossover", 0.0)  + (time.perf_counter() - _t_cx)
            _ga_logger._phase_counts["crossover"] = _ga_logger._phase_counts.get("crossover", 0)   + 1

            # Mutação
            _t_mut = time.perf_counter()
            for idx in range(len(offspring)):
                before = list(offspring[idx])
                if random.random() < mut_pb:
                    offspring[idx], = toolbox.mutate(offspring[idx]) # type: ignore
                    del offspring[idx].fitness.values
                    _ga_logger.log_mutation(gen, idx, before, list(offspring[idx]), True)
                else:
                    _ga_logger.log_mutation(gen, idx, before, list(offspring[idx]), False)
            _ga_logger._phase_times["mutacao"]  = _ga_logger._phase_times.get("mutacao", 0.0)  + (time.perf_counter() - _t_mut)
            _ga_logger._phase_counts["mutacao"] = _ga_logger._phase_counts.get("mutacao", 0)   + 1

            # Avaliação lazy
            _t_eval = time.perf_counter()
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind) # type: ignore
            _ga_logger._phase_times["avaliacao"]  = _ga_logger._phase_times.get("avaliacao", 0.0)  + (time.perf_counter() - _t_eval)
            _ga_logger._phase_counts["avaliacao"] = _ga_logger._phase_counts.get("avaliacao", 0)   + 1

            # Atualiza população e HoF
            populacao[:] = offspring
            hof.update(populacao)

            best_ind = hof[0]
            cv_acc   = round(best_ind.fitness.values[0], 4)
            fits     = [ind.fitness.values[0] for ind in populacao]
            mean_f   = sum(fits) / len(fits)

            _ga_logger.log_generation_stats(gen, cv_acc, mean_f, target_cv, best_ind)
            _ga_logger.log_population(gen, populacao)

            feat_name = "sqrt" if best_ind[4] == 0 else "log2"
            print(
                f"      Geração {gen:>3} | CV 5-fold: {cv_acc:.4f} (meta:>{target_cv:.4f})"
                f" | n_estimators={best_ind[0]} max_depth={best_ind[1]}"
                f" min_samples_leaf={best_ind[2]} min_samples_split={best_ind[3]} max_features={feat_name}"
            )
            if cv_acc > target_cv:
                print(f"      Meta superada na geração {gen} (CV: {cv_acc:.4f} > {target_cv:.4f})")
                break
    except KeyboardInterrupt:
        print(f"\n      Interrompido na geração {gen}. Retornando melhor indivíduo encontrado até agora.")

    best_ind = hof[0]
    _ga_logger.log_run_end(
        gen, best_ind.fitness.values[0], time.time() - _t0,
        meta_atingida=round(best_ind.fitness.values[0], 4) > target_cv,
        best_ind=best_ind,
    )
    return hof[0], _ga_logger