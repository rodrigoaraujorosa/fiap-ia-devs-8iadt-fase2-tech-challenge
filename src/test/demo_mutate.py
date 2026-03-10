"""
Demo visual do operador de Mutação (uniforme inteira) — ga_handmade.py

Execução:
    python src/test/demo_mutate.py
"""
import copy
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.ga_handmade import Individual, MUT_INDPB, mutate

GENE_LABELS = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]


def _fmt(ind: Individual) -> str:
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
    return [GENE_LABELS[i] for i in range(5) if before[i] != after[i]]


# ---------------------------------------------------------------------------
# Exemplos fixos com seeds diferentes para mostrar variedade de mutações
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
print("  Demo: Operador de Mutação (uniforme inteira) — ga_handmade.py")
print(f"  Probabilidade de mutação por gene (MUT_INDPB): {MUT_INDPB}")
print("=" * 72)
print("  Genes marcados com (*) foram mutados.")
print("=" * 72)

for idx, (genes, seed) in enumerate(EXAMPLES, 1):
    random.seed(seed)
    ind = Individual(genes[:])
    mutate(ind, mut_indpb=MUT_INDPB)

    changed = _changed_genes(genes, ind)
    changed_str = ", ".join(changed) if changed else "nenhum gene alterado"

    print(f"\n  Exemplo {idx}  (seed={seed})")
    print(f"  {'Original':<12}: {_fmt(Individual(genes))}")
    print(f"  {'Após mutação':<12}: {_fmt_diff(genes, ind)}")
    print(f"  Genes alterados : {changed_str}")

print("\n" + "=" * 72)
print("  Fim da demonstração.")
print("=" * 72)
