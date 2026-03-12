"""
Demo visual dos três operadores genéticos em conjunto — ga_handmade.py

Simula uma geração completa do AG:
    1. Seleção por torneio  — escolhe os pais da próxima geração
    2. Crossover dois pontos — combina pares de pais para gerar filhos
    3. Mutação uniforme     — introduz variações aleatórias nos filhos

Execução:
    python src/test/demo_ga_operators.py
"""
import copy
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.ga_handmade import (
    CX_PB,
    MUT_INDPB,
    MUT_PB,
    Individual,
    crossover,
    mutate,
    selection,
)

GENE_LABELS = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]
SEP = "=" * 72
SEP_THIN = "-" * 72


def _fmt(genes) -> str:
    # Formata os genes como string legível com nome=valor por gene.
    # max_features é decodificado de binário (0=sqrt, 1=log2) para o nome real.
    parts = []
    for i, label in enumerate(GENE_LABELS):
        if i == 4:
            val = "sqrt" if genes[i] == 0 else "log2"
            parts.append(f'{label}="{val}"')
        else:
            parts.append(f"{label}={genes[i]}")
    return "[" + ", ".join(parts) + "]"


def _fmt_diff(before: list, after: Individual) -> str:
    # Compara gene a gene com 'before'; genes alterados recebem o marcador (*).
    # Utilizado tanto após crossover quanto após mutação para destacar mudanças.
    parts = []
    for i, label in enumerate(GENE_LABELS):
        marker = " (*)" if after[i] != before[i] else ""
        if i == 4:
            val = "sqrt" if after[i] == 0 else "log2"
            parts.append(f'{label}="{val}"{marker}')
        else:
            parts.append(f"{label}={after[i]}{marker}")
    return "[" + ", ".join(parts) + "]"


def _make_ind(genes: list, fitness: float) -> Individual:
    # Cria um indivíduo já inicializado com fitness pré-definido,
    # simulando uma população que já passou pela avaliação (cross-validation).
    ind = Individual(genes)
    ind.fitness = fitness
    return ind


# ---------------------------------------------------------------------------
# População inicial de 6 indivíduos com fitness já atribuídos.
# Os fitnesses cobrem um espectro variado para que a pressão seletiva
# do torneio seja claramente observada no Passo 1.
# ---------------------------------------------------------------------------
POPULATION = [
    _make_ind([20,  5,  1,  2, 0], fitness=0.7000),
    _make_ind([30, 15,  1,  5, 1], fitness=0.7900),
    _make_ind([44, 20,  3, 10, 0], fitness=0.7500),
    _make_ind([55, 25,  8, 18, 1], fitness=0.6800),
    _make_ind([35, 10,  4,  8, 0], fitness=0.7700),
    _make_ind([60, 25, 10, 20, 1], fitness=0.7200),
]

random.seed(42)  # garante reprodutibilidade: mesmos resultados a cada execução

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
print(SEP)
print("  Demo Completo: Operadores Genéticos em Ação — ga_handmade.py")
print("  Uma geração completa: Seleção → Crossover → Mutação")
print(SEP)

# ---------------------------------------------------------------------------
# PASSO 0 — População inicial
# ---------------------------------------------------------------------------
print("\n  PASSO 0 — População inicial")
print(f"  {'#':<4} {'Fitness':<10}  Indivíduo")
print("  " + SEP_THIN)
for i, ind in enumerate(POPULATION):
    best_marker = "  ← melhor" if ind.fitness == max(p.fitness for p in POPULATION) else "" # type: ignore[arg-type]
    print(f"  {i:<4} {ind.fitness:<10.4f}  {_fmt(ind)}{best_marker}")

# ---------------------------------------------------------------------------
# PASSO 1 — Seleção por torneio.
# A seleção não modifica a população original: gera uma nova lista de pais
# do mesmo tamanho, favorecendo os indivíduos mais aptos.
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PASSO 1 — Seleção por Torneio  (tournsize=3, k={len(POPULATION)})")
print(f"  Cada vaga é disputada por 3 candidatos sorteados; vence o maior fitness.")
print(SEP)

parents = [copy.copy(ind) for ind in selection(POPULATION, k=len(POPULATION))]
# Cópias rasas dos selecionados: preserva os indivíduos originais intactos
# para que crossover e mutação operem sobre os pais, não sobre a população.

print(f"\n  {'#':<4} {'Fitness':<10}  Indivíduo  (pais selecionados)")
print("  " + SEP_THIN)
for i, ind in enumerate(parents):
    print(f"  {i:<4} {ind.fitness:<10.4f}  {_fmt(ind)}")

fitness_before = sorted([ind.fitness for ind in POPULATION], reverse=True) # type: ignore[arg-type]
fitness_after  = sorted([ind.fitness for ind in parents], reverse=True) # type: ignore[arg-type]
print(f"\n  Fitness anterior (pop. original) : {fitness_before}")
print(f"  Fitness após seleção (pais)      : {fitness_after}")
print(f"  → Os pais selecionados têm fitness médio mais alto que a população.")

# ---------------------------------------------------------------------------
# PASSO 2 — Crossover em pares consecutivos.
# Pares são formados por índices (0,1), (2,3), (4,5).
# Cada par tem CX_PB de chance de cruzar; se não cruzar, os filhos
# são cópias diretas dos pais (sem alteração de genes).
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PASSO 2 — Crossover de Dois Pontos  (CX_PB={CX_PB})")
print(f"  Pares consecutivos (0+1, 2+3, 4+5) têm {int(CX_PB*100)}% de chance de cruzar.")
print(f"  Genes marcados com (*) foram herdados do outro pai.")
print(SEP)

offspring = [copy.copy(ind) for ind in parents]  # copia os pais para não modificá-los
n_crossovers = 0  # contador de crossovers efetivamente aplicados nesta geração

for i in range(1, len(offspring), 2):
    # Registra os genes antes do crossover para comparar depois com _fmt_diff.
    pai1_genes = list(offspring[i - 1])
    pai2_genes = list(offspring[i])
    aplicou = random.random() < CX_PB  # decide estocàsticamente se o par cruza

    print(f"\n  Par ({i-1}, {i})  {'→ CROSSOVER APLICADO' if aplicou else '→ sem crossover (chance não atingida)'}")
    print(f"  {'Pai ' + str(i-1):<12}: {_fmt(pai1_genes)}")
    print(f"  {'Pai ' + str(i):<12}: {_fmt(pai2_genes)}")

    if aplicou:
        offspring[i - 1], offspring[i] = crossover(offspring[i - 1], offspring[i])
        # Após o crossover, o fitness dos filhos é invalidado (None) porque
        # os genes mudaram e o valor anterior do CV não é mais válido.
        offspring[i - 1].fitness = None
        offspring[i].fitness = None
        n_crossovers += 1
        print(f"  {'Child ' + str(i-1):<12}: {_fmt_diff(pai1_genes, offspring[i-1])}")
        print(f"  {'Child ' + str(i):<12}: {_fmt_diff(pai2_genes, offspring[i])}")
    else:
        print(f"  {'Child ' + str(i-1):<12}: {_fmt(offspring[i-1])}  (inalterado)")
        print(f"  {'Child ' + str(i):<12}: {_fmt(offspring[i])}  (inalterado)")

print(f"\n  Total de crossovers aplicados: {n_crossovers} de {len(offspring)//2} pares")

# ---------------------------------------------------------------------------
# PASSO 3 — Mutação individual.
# Cada filho tem MUT_PB de chance de ser mutado. Se mutado, cada gene
# é alterado independentemente com probabilidade MUT_INDPB, para um
# valor aleatório dentro dos limites do espaço de busca daquele gene.
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PASSO 3 — Mutação Uniforme Inteira  (MUT_PB={MUT_PB}, MUT_INDPB={MUT_INDPB})")
print(f"  Cada filho tem {int(MUT_PB*100)}% de chance de sofrer mutação;")
print(f"  se mutado, cada gene é alterado individualmente com {int(MUT_INDPB*100)}% de chance.")
print(f"  Genes marcados com (*) foram mutados.")
print(SEP)

n_mutations = 0  # contador de mutações efetivamente aplicadas nesta geração
for i, ind in enumerate(offspring):
    genes_before = list(ind)  # salva o estado antes da mutação para comparar depois
    aplicou = random.random() < MUT_PB  # decide estocàsticamente se o filho é mutado

    if aplicou:
        mutate(ind, mut_indpb=MUT_INDPB)
        # Fitness invalidado: genes alterados → o CV anterior não é mais válido.
        ind.fitness = None
        n_mutations += 1
        changed = [GENE_LABELS[j] for j in range(5) if ind[j] != genes_before[j]]
        changed_str = ", ".join(changed) if changed else "nenhum gene alterado"
        print(f"\n  Filho {i}  → MUTAÇÃO APLICADA")
        print(f"  {'antes':<8}: {_fmt(genes_before)}")
        print(f"  {'após':<8}: {_fmt_diff(genes_before, ind)}")
        print(f"  Genes alterados: {changed_str}")
    else:
        print(f"\n  Filho {i}  → sem mutação (chance não atingida)")
        print(f"  {'genes':<8}: {_fmt(ind)}")

print(f"\n  Total de mutações aplicadas: {n_mutations} de {len(offspring)} filhos")

# ---------------------------------------------------------------------------
# Resumo da geração
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  RESUMO DA GERAÇÃO")
print(SEP)
print(f"\n  {'Etapa':<30} {'Indivíduos modificados'}")
print("  " + SEP_THIN)
print(f"  {'Seleção (torneio)':<30} {len(parents)} pais escolhidos da população")
print(f"  {'Crossover':<30} {n_crossovers * 2} filhos gerados por recombinação")
print(f"  {'Mutação':<30} {n_mutations} filhos com genes alterados")

fitness_offspring = [ind.fitness for ind in offspring if ind.fitness is not None]
fitness_offspring_str = [f"{f:.4f}" for f in sorted(fitness_offspring, reverse=True)]
print(f"\n  Fitness dos filhos com fitness válido (não reavaliados nesta demo):")
if fitness_offspring_str:
    print(f"  {fitness_offspring_str}")
else:
    print("  Todos precisam ser reavaliados (fitness=None após cx/mut).")
print(f"\n  → Na execução real do AG, os filhos sem fitness")
print(f"    são reavaliados via cross-validation antes da próxima geração.")

print("\n" + SEP)
print("  Fim da demonstração.")
print(SEP)
