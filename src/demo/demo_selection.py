"""
Demo visual do operador de Seleção por Torneio — ga_handmade.py

A seleção por torneio escolhe k indivíduos da população. Para cada vaga,
sorteia tournsize=3 candidatos aleatoriamente e elege o de maior fitness.
Indivíduos mais aptos têm maior chance de ser selecionados, mas até os
menos aptos podem ser escolhidos ocasionalmente (pressão seletiva controlada).

Execução:
    python src/test/demo_selection.py
"""
import random
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.ga_handmade import Individual, selection

GENE_LABELS = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]


def _fmt(ind: Individual) -> str:
    # Formata o indivíduo como string legível com nome=valor por gene.
    # max_features é decodificado de binário (0=sqrt, 1=log2) para o nome real.
    parts = []
    for i, label in enumerate(GENE_LABELS):
        if i == 4:
            val = "sqrt" if ind[i] == 0 else "log2"
            parts.append(f'{label}="{val}"')
        else:
            parts.append(f"{label}={ind[i]}")
    return "[" + ", ".join(parts) + "]"


def _make_ind(genes: list, fitness: float) -> Individual:
    # Cria um indivíduo já inicializado com fitness pré-definido,
    # simulando uma população que já passou por avaliação (cross-validation).
    ind = Individual(genes)
    ind.fitness = fitness
    return ind


# ---------------------------------------------------------------------------
# População de exemplo com fitnesses variados.
# Os valores cobrem um espectro de aptidão para que a pressão seletiva
# do torneio seja claramente observada nas partes 2 e 3 do demo.
# ---------------------------------------------------------------------------
POPULATION = [
    _make_ind([20,  5,  1,  2, 0], fitness=0.7000),
    _make_ind([30, 15,  1,  5, 1], fitness=0.7900),
    _make_ind([44, 20,  3, 10, 0], fitness=0.7500),
    _make_ind([55, 25,  8, 18, 1], fitness=0.6800),
    _make_ind([35, 10,  4,  8, 0], fitness=0.7700),
    _make_ind([60, 25, 10, 20, 1], fitness=0.7200),
]

print("=" * 72)
print("  Demo: Operador de Seleção por Torneio — ga_handmade.py")
print("  tournsize=3 | k = tamanho da população")
print("=" * 72)

# ---------------------------------------------------------------------------
# Parte 1 — Exibe a população com os fitness.
# O ← melhor marca visualmente o indivíduo de referência para comparar
# com os selecionados pelo torneio nas partes seguintes.
# ---------------------------------------------------------------------------
print("\n  População inicial:")
print(f"  {'#':<4} {'Fitness':<10}  Indivíduo")
print("  " + "-" * 68)
for i, ind in enumerate(POPULATION):
    marker = "  ← melhor" if ind.fitness == max(p.fitness for p in POPULATION) else "" # type: ignore[arg-type]
    print(f"  {i:<4} {ind.fitness:<10.4f}  {_fmt(ind)}{marker}")

# ---------------------------------------------------------------------------
# Parte 2 — Exemplo único de seleção (seed fixo, k = tamanho da população).
# seed=42 garante reprodutibilidade: o mesmo resultado a cada execução.
# Com k igual ao tamanho da população, cria-se uma nova geração de mesmo tamanho.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  Exemplo de seleção (seed=42, k=6):")
print("=" * 72)

random.seed(42)
chosen = selection(POPULATION, k=len(POPULATION))

print(f"\n  {'#':<4} {'Fitness':<10}  Indivíduo")
print("  " + "-" * 68)
for i, ind in enumerate(chosen):
    print(f"  {i:<4} {ind.fitness:<10.4f}  {_fmt(ind)}")

fitness_chosen = sorted([ind.fitness for ind in chosen], reverse=True) # type: ignore[arg-type]
fitness_pop    = sorted([ind.fitness for ind in POPULATION], reverse=True) # type: ignore[arg-type]
print(f"\n  Fitness selecionados (ord. desc.): {fitness_chosen}")
print(f"  Fitness da população (ord. desc.): {fitness_pop}")

# ---------------------------------------------------------------------------
# Parte 3 — Frequência de seleção em 1000 torneios (pressão seletiva).
# Repete a seleção N_ROUNDS vezes e conta quantas vezes cada indivíduo
# foi escolhido. Permite visualizar que o torneio favorece os mais aptos
# mas ainda permite que indivíduos fracos sejam ocasionalmente eleitos,
# mantendo a diversidade genética da população.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  Frequência de seleção em 1.000 torneios (k=6 por rodada):")
print("  Mostra a pressão seletiva: os mais aptos devem aparecer mais vezes.")
print("=" * 72)

random.seed(0)
N_ROUNDS = 1000
counts: Counter = Counter()
for _ in range(N_ROUNDS):
    chosen_round = selection(POPULATION, k=len(POPULATION))
    for ind in chosen_round:
        counts[id(ind)] += 1

total = sum(counts.values())
print(f"\n  {'Fitness':<10} {'Selecionado':<14} {'Frequência'}")
print("  " + "-" * 40)
for ind in sorted(POPULATION, key=lambda x: x.fitness, reverse=True): # type: ignore[arg-type]
    cnt  = counts[id(ind)]
    freq = cnt / total
    bar  = "█" * int(freq * 50)
    print(f"  {ind.fitness:<10.4f} {cnt:<14}  {freq:.1%}  {bar}")

print("\n" + "=" * 72)
print("  Conclusão: indivíduos com maior fitness são selecionados com")
print("  maior frequência, mas há chance de qualquer indivíduo ser")
print("  escolhido — o torneio mantém diversidade genética na população.")
print("=" * 72)
