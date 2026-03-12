"""
Demo visual do operador de Mutação (uniforme inteira) — ga_rf_optimizer.py

Execução:
    python src/test/demo_mutate.py
"""
import copy
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.ga_rf_optimizer import Individual, MUT_INDPB, mutate

GENE_LABELS = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]


def _fmt(ind: Individual) -> str:
    # Formata o indivíduo como string legível com nome=valor por gene.
    # max_features é decodificado de binário (0=sqrt, 1=log2) para o nome real.
    parts = []
    for i, label in enumerate(GENE_LABELS):
        if i == 4:
            val = "sqrt" if ind[i] == 0 else "log2"
        else:
            val = str(ind[i])
        parts.append(f'{label}="{val}"' if i == 4 else f"{label}={val}")
    return "[" + ", ".join(parts) + "]"


def _fmt_diff(before: list, after: Individual) -> str:
    """Igual a _fmt, mas marca com (*) os genes que foram mutados."""
    # Compara gene a gene com o estado anterior para identificar quais foram
    # alterados pela mutação, destacando visualmente as mudanças no terminal.
    parts = []
    for i, label in enumerate(GENE_LABELS):
        if i == 4:
            val = "sqrt" if after[i] == 0 else "log2"
        else:
            val = str(after[i])
        marker = "  (*)" if after[i] != before[i] else ""
        parts.append((f'{label}="{val}"{marker}' if i == 4 else f"{label}={val}{marker}"))
    return "[" + ", ".join(parts) + "]"


def _changed_genes(before: list, after: Individual) -> list[str]:
    # Retorna os nomes dos genes que mudaram após a mutação,
    # comparando o cromossomo original com o mutado posição a posição.
    return [GENE_LABELS[i] for i in range(5) if before[i] != after[i]]


# ---------------------------------------------------------------------------
# Exemplos fixos com seeds diferentes para mostrar variedade de mutações.
# Cada seed produz uma combinação diferente de genes mutados,
# ilustrando que a mutação é estocástica e pode afetar 0 a 5 genes.
# ---------------------------------------------------------------------------
EXAMPLES = [
    # (genes_originais,                    seed)
    ([30, 15,  1,  5, 1], 0),
    ([44, 20,  3, 10, 0], 7),
    ([20,  5,  1,  2, 0], 42),
    ([55, 25,  8, 18, 1], 13),
    ([35, 10,  4,  8, 0], 99),
]

print("=" * 72)
print("  Demo: Operador de Mutação (uniforme inteira) — ga_rf_optimizer.py")
print(f"  Probabilidade de mutação por gene (MUT_INDPB): {MUT_INDPB}")
print("=" * 72)
print("  Genes marcados com (*) foram mutados.")
print("=" * 72)

for idx, (genes, seed) in enumerate(EXAMPLES, 1):
    random.seed(seed)  # fixa o gerador para reprodutibilidade do demo
    ind = Individual(genes[:])
    mutate(ind, mut_indpb=MUT_INDPB)  # aplica mutação uniforme inteira gene a gene

    changed = _changed_genes(genes, ind)
    changed_str = ", ".join(changed) if changed else "nenhum gene alterado"

    print(f"\n  Exemplo {idx}  (seed={seed})")
    print(f"  {'Original':<12}: {_fmt(Individual(genes))}")
    print(f"  {'Após mutação':<12}: {_fmt_diff(genes, ind)}")
    print(f"  Genes alterados : {changed_str}")

print("\n" + "=" * 72)
print("  Fim da demonstração.")
print("=" * 72)
