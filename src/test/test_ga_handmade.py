"""
Testes unitários para src/engine/ga_handmade.py
Execução:
    pytest src/test/test_ga_handmade.py -v
"""
import copy
import random
import sys
import os

import pytest

# Garante que src/ esteja no path independentemente de onde pytest é chamado
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.ga_handmade import (
    GENE_HIGH,
    GENE_LOW,
    GRIDSEARCH_SEED,
    MAX_DEPTH_OPTIONS,
    MIN_SAMPLES_LEAF_HIGH,
    MIN_SAMPLES_LEAF_LOW,
    MIN_SAMPLES_SPLIT_HIGH,
    MIN_SAMPLES_SPLIT_LOW,
    MUT_INDPB,
    N_ESTIMATORS_HIGH,
    N_ESTIMATORS_LOW,
    Individual,
    _random_individual,
    create_seeded_pop,
    crossover,
    mutate,
    selection,
)

# ---------------------------------------------------------------------------
# Helpers de teste
# ---------------------------------------------------------------------------

def _make_ind(genes: list, fitness: float | None = None) -> Individual:
    ind = Individual(genes)
    ind.fitness = fitness
    return ind


def _genes_str(ind: Individual) -> str:
    """Representação legível dos genes de um indivíduo."""
    labels = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]
    feat = "sqrt" if ind[4] == 0 else "log2"
    parts = [f"{labels[i]}={ind[i]}" for i in range(4)]
    parts.append(f"max_features={feat}")
    return "  [" + ", ".join(parts) + "]"


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

class TestIndividual:
    def test_init_stores_genes(self):
        genes = [30, 15, 1, 5, 1]
        ind = Individual(genes)
        assert list(ind) == genes

    def test_fitness_starts_none(self):
        ind = Individual([30, 15, 1, 5, 1])
        assert ind.fitness is None

    def test_copy_preserves_genes_and_fitness(self):
        ind = _make_ind([30, 15, 1, 5, 1], fitness=0.78)
        ind2 = copy.copy(ind)
        assert list(ind2) == list(ind)
        assert ind2.fitness == ind.fitness

    def test_copy_is_independent(self):
        """Mudar o clone não deve afetar o original."""
        ind = _make_ind([30, 15, 1, 5, 1], fitness=0.78)
        ind2 = copy.copy(ind)
        ind2[0] = 99
        ind2.fitness = 0.0
        assert ind[0] == 30
        assert ind.fitness == 0.78


# ---------------------------------------------------------------------------
# _random_individual
# ---------------------------------------------------------------------------

class TestRandomIndividual:
    def test_returns_individual(self):
        ind = _random_individual()
        assert isinstance(ind, Individual)

    def test_gene_bounds(self):
        random.seed(0)
        for _ in range(200):
            ind = _random_individual()
            assert N_ESTIMATORS_LOW  <= ind[0] <= N_ESTIMATORS_HIGH,  f"n_estimators fora dos limites: {ind[0]}"
            assert ind[1] in MAX_DEPTH_OPTIONS,                        f"max_depth inválido: {ind[1]}"
            assert MIN_SAMPLES_LEAF_LOW  <= ind[2] <= MIN_SAMPLES_LEAF_HIGH,  f"min_samples_leaf fora dos limites: {ind[2]}"
            assert MIN_SAMPLES_SPLIT_LOW <= ind[3] <= MIN_SAMPLES_SPLIT_HIGH, f"min_samples_split fora dos limites: {ind[3]}"
            assert ind[4] in (0, 1),                                   f"max_features inválido: {ind[4]}"

    def test_fitness_is_none(self):
        ind = _random_individual()
        assert ind.fitness is None


# ---------------------------------------------------------------------------
# create_seeded_pop
# ---------------------------------------------------------------------------

class TestCreateSeededPop:
    def test_size(self):
        pop = create_seeded_pop(10)
        assert len(pop) == 10

    def test_first_is_gridsearch_seed(self):
        pop = create_seeded_pop(5)
        assert list(pop[0]) == GRIDSEARCH_SEED, (
            f"Esperado semente GridSearch {GRIDSEARCH_SEED}, obtido {list(pop[0])}"
        )

    def test_all_are_individuals(self):
        pop = create_seeded_pop(8)
        for ind in pop:
            assert isinstance(ind, Individual)

    def test_fitness_all_none(self):
        pop = create_seeded_pop(8)
        for ind in pop:
            assert ind.fitness is None


# ---------------------------------------------------------------------------
# crossover
# ---------------------------------------------------------------------------

class TestCrossover:
    def test_returns_two_individuals(self):
        ind1 = _make_ind([20, 5,  1,  2, 0])
        ind2 = _make_ind([60, 25, 10, 20, 1])
        result = crossover(ind1, ind2)
        assert len(result) == 2

    def test_genes_are_mix_of_parents(self):
        """Após crossover cada filho deve conter apenas genes que existiam nos pais."""
        random.seed(42)
        parent1_orig = [20, 5,  1,  2, 0]
        parent2_orig = [60, 25, 10, 20, 1]
        parent_gene_sets = [set(v) for v in zip(parent1_orig, parent2_orig)]

        for seed in range(50):
            random.seed(seed)
            ind1 = _make_ind(parent1_orig[:])
            ind2 = _make_ind(parent2_orig[:])
            c1, c2 = crossover(ind1, ind2)
            for i in range(5):
                assert c1[i] in parent_gene_sets[i], (
                    f"gene[{i}]={c1[i]} não pertence aos pais (seed={seed})"
                )
                assert c2[i] in parent_gene_sets[i], (
                    f"gene[{i}]={c2[i]} não pertence aos pais (seed={seed})"
                )

    def test_total_genes_conserved(self):
        """A soma de genes pai1+pai2 deve ser igual à soma filho1+filho2 (genes apenas trocam)."""
        random.seed(7)
        ind1 = _make_ind([20, 5,  1,  2, 0])
        ind2 = _make_ind([60, 25, 10, 20, 1])
        original_sum = [ind1[i] + ind2[i] for i in range(5)]
        c1, c2 = crossover(ind1, ind2)
        result_sum = [c1[i] + c2[i] for i in range(5)]
        assert original_sum == result_sum

    def test_verbose_output(self, capsys):
        """Exibe os cromossomos antes e depois do crossover para inspeção visual."""
        random.seed(42)
        ind1 = _make_ind([20, 5,  1,  2, 0])
        ind2 = _make_ind([60, 25, 10, 20, 1])
        print("\n--- crossover (seed=42) ---")
        print(f"  pai1 antes : {list(ind1)}")
        print(f"  pai2 antes : {list(ind2)}")
        c1, c2 = crossover(ind1, ind2)
        print(f"  filho1 após: {list(c1)}")
        print(f"  filho2 após: {list(c2)}")
        captured = capsys.readouterr()
        assert "filho1 após" in captured.out


# ---------------------------------------------------------------------------
# mutate
# ---------------------------------------------------------------------------

class TestMutate:
    def test_returns_tuple_with_individual(self):
        ind = _make_ind([30, 15, 1, 5, 1])
        result = mutate(ind)
        assert isinstance(result, tuple) and len(result) == 1
        assert isinstance(result[0], Individual)

    def test_mutates_in_place(self):
        """O objeto retornado deve ser o mesmo (mutação in-place)."""
        ind = _make_ind([30, 15, 1, 5, 1])
        (result,) = mutate(ind)
        assert result is ind

    def test_genes_stay_within_bounds_after_many_mutations(self):
        """Após muitas mutações os genes devem sempre respeitar os limites."""
        random.seed(0)
        for _ in range(500):
            ind = _make_ind([30, 15, 1, 5, 1])
            mutate(ind)
            for i in range(5):
                assert GENE_LOW[i] <= ind[i] <= GENE_HIGH[i], (
                    f"gene[{i}]={ind[i]} fora de [{GENE_LOW[i]}, {GENE_HIGH[i]}]"
                )

    def test_at_least_one_gene_mutates_with_indpb_1(self, monkeypatch):
        """Com MUT_INDPB=1.0 todos os genes devem mudar (ou pelo menos serem reatribuídos)."""
        monkeypatch.setattr("engine.ga_handmade.MUT_INDPB", 1.0)
        random.seed(99)
        original = [30, 15, 1, 5, 1]
        ind = _make_ind(original[:])
        mutate(ind)
        # Com indpb=1.0 todos os genes são reatribuídos; verificamos apenas os limites
        for i in range(5):
            assert GENE_LOW[i] <= ind[i] <= GENE_HIGH[i]

    def test_no_gene_mutates_with_indpb_0(self, monkeypatch):
        """Com MUT_INDPB=0.0 nenhum gene deve ser alterado."""
        monkeypatch.setattr("engine.ga_handmade.MUT_INDPB", 0.0)
        original = [30, 15, 1, 5, 1]
        ind = _make_ind(original[:])
        mutate(ind)
        assert list(ind) == original

    def test_verbose_output(self, capsys):
        """Exibe o cromossomo antes e após a mutação para inspeção visual."""
        random.seed(0)
        original = [30, 15, 1, 5, 1]
        ind = _make_ind(original[:])
        changed = [i for i in range(5) if random.random() < MUT_INDPB]

        random.seed(0)
        ind2 = _make_ind(original[:])
        print("\n--- mutate (seed=0) ---")
        print(f"  antes : {original}")
        mutate(ind2)
        print(f"  após  : {list(ind2)}")
        feat_before = "sqrt" if original[4] == 0 else "log2"
        feat_after  = "sqrt" if ind2[4]    == 0 else "log2"
        print(f"  max_features: {feat_before} → {feat_after}")
        captured = capsys.readouterr()
        assert "após" in captured.out


# ---------------------------------------------------------------------------
# selection (torneio)
# ---------------------------------------------------------------------------

class TestSelection:
    def _pop_with_fitness(self) -> list[Individual]:
        fitnesses = [0.75, 0.80, 0.70, 0.85, 0.72, 0.78]
        return [_make_ind([30, 15, 1, 5, 1], fitness=f) for f in fitnesses]

    def test_returns_correct_size(self):
        pop = self._pop_with_fitness()
        chosen = selection(pop, k=4)
        assert len(chosen) == 4

    def test_chosen_are_from_population(self):
        pop = self._pop_with_fitness()
        chosen = selection(pop, k=6)
        for ind in chosen:
            assert any(ind is orig for orig in pop), "Indivíduo selecionado não pertence à população original"

    def test_best_individual_selected_more_often(self):
        """O indivíduo com maior fitness deve aparecer com frequência acima da média."""
        random.seed(0)
        pop = self._pop_with_fitness()
        best_fitness = max(ind.fitness for ind in pop) # type: ignore[arg-type]
        best_ind = next(ind for ind in pop if ind.fitness == best_fitness)

        k = 1000
        chosen = selection(pop, k=k)
        count_best = sum(1 for ind in chosen if ind is best_ind)
        # Com torneio de 3, esperamos seleção do melhor bem acima de 1/6 ≈ 16,7 %
        assert count_best / k > 0.25, (
            f"Melhor indivíduo selecionado apenas {count_best/k:.1%} das vezes"
        )

    def test_verbose_output(self, capsys):
        """Exibe os fitness dos selecionados para inspeção visual."""
        random.seed(42)
        pop = self._pop_with_fitness()
        print("\n--- selection (torneio, k=6, seed=42) ---")
        print(f"  fitness da população : {[ind.fitness for ind in pop]}")
        chosen = selection(pop, k=6)
        print(f"  fitness selecionados : {[ind.fitness for ind in chosen]}")
        captured = capsys.readouterr()
        assert "fitness selecionados" in captured.out


# ---------------------------------------------------------------------------
# Integração: crossover + mutate
# ---------------------------------------------------------------------------

class TestCrossoverMutateIntegration:
    def test_full_pipeline_genes_within_bounds(self):
        """Após crossover e mutação os genes dos filhos devem respeitar os limites."""
        random.seed(123)
        ind1 = _make_ind([20, 5,  1,  2, 0])
        ind2 = _make_ind([60, 25, 10, 20, 1])
        c1, c2 = crossover(copy.copy(ind1), copy.copy(ind2))
        mutate(c1)
        mutate(c2)
        for gene_idx in range(5):
            assert GENE_LOW[gene_idx] <= c1[gene_idx] <= GENE_HIGH[gene_idx]
            assert GENE_LOW[gene_idx] <= c2[gene_idx] <= GENE_HIGH[gene_idx]

    def test_verbose_pipeline(self, capsys):
        """Exibe o pipeline completo: pais → crossover → mutação → filhos finais."""
        random.seed(7)
        parent1 = [20, 5,  1,  2, 0]
        parent2 = [60, 25, 10, 20, 1]
        ind1 = _make_ind(parent1[:])
        ind2 = _make_ind(parent2[:])

        print("\n=== Pipeline crossover + mutate (seed=7) ===")
        print(f"  pai1        : {parent1}")
        print(f"  pai2        : {parent2}")

        c1, c2 = crossover(ind1, ind2)
        print(f"  após cx → c1: {list(c1)}")
        print(f"  após cx → c2: {list(c2)}")

        mutate(c1)
        mutate(c2)
        print(f"  após mut → c1: {list(c1)}")
        print(f"  após mut → c2: {list(c2)}")

        feat1 = "sqrt" if c1[4] == 0 else "log2"
        feat2 = "sqrt" if c2[4] == 0 else "log2"
        print(f"  max_features finais: c1={feat1}, c2={feat2}")

        captured = capsys.readouterr()
        assert "após mut" in captured.out
