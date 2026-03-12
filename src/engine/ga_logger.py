"""Módulo de logging para o Algoritmo Genético.

Registra parâmetros de execução, operadores genéticos (seleção, cruzamento,
mutação) e estatísticas por geração em arquivo de log persistente.

Arquivo criado em: logs/ga_{algoritmo}_{YYYYMMDD_HHMMSS}.log

Níveis de log utilizados:
  INFO  — eventos relevantes: início/fim de execução, seleção, cruzamentos e
          mutações que ocorreram de fato, estatísticas por geração.
  DEBUG — eventos detalhados: indivíduos da população, slots selecionados,
          pares que NÃO realizaram cruzamento/mutação.
"""
import logging
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()  # carrega .env a partir da raiz do projeto (ou qualquer diretório pai)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")

_LOG_LEVEL_MAP = {"DEBUG": logging.DEBUG, "INFO": logging.INFO}
_FILE_LOG_LEVEL = _LOG_LEVEL_MAP.get(
    os.getenv("GA_LOG_LEVEL", "INFO").upper(), logging.INFO
)

# Nomes dos genes na ordem do cromossomo
GENE_NAMES = [
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "max_features",
]

_MAX_FEAT_NAME = {0: "sqrt", 1: "log2"}


def _decode_genes(genes) -> str:
    feat = _MAX_FEAT_NAME.get(int(genes[4]), str(genes[4]))
    return (
        f"n_est={genes[0]} max_depth={genes[1]} min_leaf={genes[2]} "
        f"min_split={genes[3]} max_feat={feat}"
    )


def _get_fitness(ind) -> float | None:
    """Retorna o fitness independente da implementação (handmade ou DEAP)."""
    f = getattr(ind, "fitness", None)
    if f is None:
        return None
    if hasattr(f, "values"):
        return f.values[0] if f.values else None
    try:
        return float(f)
    except (TypeError, ValueError):
        return None


class GALogger:
    """Logger por execução do AG. Cria logs/ga_{algorithm}_{timestamp}.log."""

    def __init__(self, algorithm: str):
        os.makedirs(_LOGS_DIR, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(_LOGS_DIR, f"ga_{algorithm}_{run_id}.log")
        self._algorithm = algorithm

        logger_name = f"ga.{algorithm}.{run_id}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._logger.handlers.clear()

        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(_FILE_LOG_LEVEL)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

    # ------------------------------------------------------------------
    # Início / fim de execução
    # ------------------------------------------------------------------

    def log_run_start(
        self,
        n_pop: int,
        cx_pb: float,
        mut_pb: float,
        mut_indpb: float,
        target_improvement: float,
        target_cv: float,
        max_gen: int | None = None,
    ):
        max_gen_str = str(max_gen) if max_gen is not None else "ilimitado"
        self._logger.info(
            f"[RUN_START  ] algoritmo={self._algorithm} | n_pop={n_pop} | "
            f"max_gen={max_gen_str} | cx_pb={cx_pb:.2f} | mut_pb={mut_pb:.2f} | "
            f"mut_indpb={mut_indpb:.2f} | "
            f"target_improvement={target_improvement * 100:.2f}% | "
            f"target_cv={target_cv:.4f}"
        )

    def log_run_end(
        self,
        gen: int,
        best_fitness: float,
        elapsed: float,
        meta_atingida: bool,
        best_ind=None,
    ):
        genes_str = (
            f" | genes=[{_decode_genes(best_ind)}]" if best_ind is not None else ""
        )
        self._logger.info(
            f"[RUN_END    ] gen={gen} | best_fitness={best_fitness:.4f} | "
            f"elapsed={elapsed:.1f}s | "
            f"meta_atingida={'SIM' if meta_atingida else 'NAO'}{genes_str}"
        )
        for h in self._logger.handlers[:]:
            h.close()
            self._logger.removeHandler(h)

    # ------------------------------------------------------------------
    # Eventos por geração
    # ------------------------------------------------------------------

    def log_generation_start(self, gen: int):
        label = " (população inicial)" if gen == 0 else ""
        self._logger.info(f"[GEN_START  ] ── geração={gen}{label} ──")

    def log_generation_stats(
        self,
        gen: int,
        best_fitness: float,
        mean_fitness: float,
        target_cv: float,
        best_ind=None,
    ):
        genes_str = (
            f" | melhor=[{_decode_genes(best_ind)}]" if best_ind is not None else ""
        )
        self._logger.info(
            f"[GEN_STATS  ] gen={gen} | best={best_fitness:.4f} | "
            f"mean={mean_fitness:.4f} | target={target_cv:.4f}{genes_str}"
        )

    def log_population(self, gen: int, population):
        ranked = sorted(
            population, key=lambda x: _get_fitness(x) or 0.0, reverse=True
        )
        for rank, ind in enumerate(ranked, 1):
            fit = _get_fitness(ind)
            fit_str = f"{fit:.4f}" if fit is not None else "N/A"
            self._logger.debug(
                f"[INDIVIDUAL ] gen={gen} | rank={rank:>2}/{len(population)} | "
                f"fitness={fit_str} | {_decode_genes(ind)}"
            )

    # ------------------------------------------------------------------
    # Operadores genéticos
    # ------------------------------------------------------------------

    def log_selection(self, gen: int, n_selected: int, tournsize: int = 3):
        self._logger.info(
            f"[SELECTION  ] gen={gen} | método=torneio | tournsize={tournsize} | "
            f"selecionados={n_selected}"
        )

    def log_selected_individuals(self, gen: int, selected):
        for slot, ind in enumerate(selected):
            fit = _get_fitness(ind)
            fit_str = f"{fit:.4f}" if fit is not None else "N/A"
            self._logger.debug(
                f"[SELECTED   ] gen={gen} | slot={slot:>2} | fitness={fit_str} | "
                f"{_decode_genes(ind)}"
            )

    def log_crossover(
        self,
        gen: int,
        pair_idx: tuple,
        ind1_before: list,
        ind2_before: list,
        ind1_after: list,
        ind2_after: list,
        happened: bool,
    ):
        i, j = pair_idx
        if happened:
            changed = sorted(
                {k for k in range(len(ind1_before)) if ind1_before[k] != ind1_after[k]}
                | {k for k in range(len(ind2_before)) if ind2_before[k] != ind2_after[k]}
            )
            self._logger.info(
                f"[CROSSOVER  ] gen={gen} | par=({i},{j}) | CX=SIM | "
                f"genes_trocados={[GENE_NAMES[k] for k in changed]} | "
                f"antes={ind1_before}+{ind2_before} | "
                f"depois={ind1_after}+{ind2_after}"
            )
        else:
            self._logger.debug(
                f"[CROSSOVER  ] gen={gen} | par=({i},{j}) | CX=NAO"
            )

    def log_mutation(
        self,
        gen: int,
        ind_idx: int,
        ind_before: list,
        ind_after: list,
        happened: bool,
    ):
        if happened:
            changed = [
                k for k in range(len(ind_before)) if ind_before[k] != ind_after[k]
            ]
            self._logger.info(
                f"[MUTATION   ] gen={gen} | ind={ind_idx:>2} | MUT=SIM | "
                f"genes_mutados={[GENE_NAMES[k] for k in changed]} | "
                f"antes={ind_before} | depois={ind_after}"
            )
        else:
            self._logger.debug(
                f"[MUTATION   ] gen={gen} | ind={ind_idx:>2} | MUT=NAO"
            )
