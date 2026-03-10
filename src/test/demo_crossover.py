"""
Demo visual do operador de Crossover (dois pontos) — ga_handmade.py

O crossover de dois pontos seleciona aleatoriamente um segmento contíguo
[cx1, cx2) do cromossomo e troca esse segmento entre os dois pais.
Com cromossomos de 5 genes, o segmento pode ter largura 1, 2, 3 ou 4 genes.
Este demo encontra automaticamente um exemplo para cada largura possível,
usando pais em que todos os 5 genes diferem — assim qualquer troca é visível.

Execução:
    python src/test/demo_crossover.py
"""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.ga_handmade import Individual, crossover

GENE_LABELS = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]

# Pais com todos os 5 genes diferentes para que qualquer troca fique evidente
PARENT1 = [20,  5,  1,  2, 0]   # n_estimators=20  max_depth=5   min_samples_leaf=1  min_samples_split=2   max_features=sqrt
PARENT2 = [60, 25, 10, 20, 1]   # n_estimators=60  max_depth=25  min_samples_leaf=10  min_samples_split=20   max_features=log2


def _fmt(genes: list) -> str:
    parts = []
    for i, label in enumerate(GENE_LABELS):
        if i == 4:
            val = "sqrt" if genes[i] == 0 else "log2"
            parts.append(f'{label}="{val}"')
        else:
            parts.append(f"{label}={genes[i]}")
    return "[" + ", ".join(parts) + "]"


def _fmt_diff(before: list, after: Individual) -> str:
    """Igual a _fmt mas marca com (*) os genes que mudaram em relação a 'before'."""
    parts = []
    for i, label in enumerate(GENE_LABELS):
        marker = "  (*)" if after[i] != before[i] else ""
        if i == 4:
            val = "sqrt" if after[i] == 0 else "log2"
            parts.append(f'{label}="{val}"{marker}')
        else:
            parts.append(f"{label}={after[i]}{marker}")
    return "[" + ", ".join(parts) + "]"


def _find_seed_for_n_changed(n_target: int, parent1: list, parent2: list) -> int:
    """Busca o menor seed que produz exatamente n_target genes trocados no Child 1."""
    for seed in range(10_000):
        random.seed(seed)
        c1, _ = crossover(Individual(parent1[:]), Individual(parent2[:]))
        n_changed = sum(1 for i in range(5) if c1[i] != parent1[i])
        if n_changed == n_target:
            return seed
    raise RuntimeError(f"Nenhum seed encontrado para n_target={n_target}")


# ---------------------------------------------------------------------------
# Encontra um seed para cada largura de segmento (1, 2, 3 e 4 genes trocados)
# ---------------------------------------------------------------------------
print("=" * 72)
print("  Demo: Operador de Crossover (dois pontos) — ga_handmade.py")
print("=" * 72)
print("  Pais usados em todos os exemplos:")
print(f"  {'Pai 1':<10}: {_fmt(PARENT1)}")
print(f"  {'Pai 2':<10}: {_fmt(PARENT2)}")
print()
print("  O segmento trocado é contíguo: [cx1 … cx2).")
print("  Genes marcados com (*) foram herdados do outro pai.")
print("=" * 72)

for n in range(1, 5):
    seed = _find_seed_for_n_changed(n, PARENT1, PARENT2)
    random.seed(seed)
    ind1 = Individual(PARENT1[:])
    ind2 = Individual(PARENT2[:])
    c1, c2 = crossover(ind1, ind2)
    n_changed = sum(1 for i in range(5) if c1[i] != PARENT1[i])

    print(f"\n  Segmento de {n} gene(s) trocado(s)  (seed={seed})")
    print(f"  {'Child 1':<10}: {_fmt_diff(PARENT1, c1)}")
    print(f"  {'Child 2':<10}: {_fmt_diff(PARENT2, c2)}")

print("\n" + "=" * 72)
print("  Conclusão: com 5 genes e crossover de dois pontos, o segmento")
print("  trocado pode ter de 1 a 4 genes — a largura depende dos pontos")
print("  de corte sorteados aleatoriamente a cada geração.")
print("=" * 72)
