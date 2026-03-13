"""Módulo de logging para o Algoritmo Genético.

Registra parâmetros de execução, operadores genéticos (seleção, cruzamento,
mutação) e estatísticas por geração em arquivo de log persistente.

Arquivo criado em: logs/{algoritmo}_{YYYYMMDD_HHMMSS}.log

Níveis de log utilizados:
  INFO  — eventos relevantes: início/fim de execução, seleção, cruzamentos e
          mutações que ocorreram de fato, estatísticas por geração.
  DEBUG — eventos detalhados: indivíduos da população, slots selecionados,
          pares que NÃO realizaram cruzamento/mutação.
"""
import logging
import os
import time
from contextlib import contextmanager
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
    """Logger por execução do AG. Cria logs/{algorithm}_{timestamp}.log."""

    def __init__(self, algorithm: str):
        os.makedirs(_LOGS_DIR, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = run_id
        self.log_file = os.path.join(_LOGS_DIR, f"{algorithm}_{run_id}.log")
        self._algorithm = algorithm

        logger_name = f"{algorithm}.{run_id}"
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

        # Acumuladores de profiling: tempo total e número de chamadas por fase
        self._phase_times: dict[str, float] = {}
        self._phase_counts: dict[str, int]  = {}

    # ------------------------------------------------------------------
    # Início / fim de execução
    # ------------------------------------------------------------------

    @contextmanager
    def timer(self, phase: str):
        """Context manager que mede o tempo de uma fase e acumula no profiling.

        Uso::

            with logger.timer("avaliacao"):
                evaluate(ind, X, y)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._phase_times[phase]  = self._phase_times.get(phase, 0.0)  + elapsed
            self._phase_counts[phase] = self._phase_counts.get(phase, 0)   + 1

    def _log_profiling(self) -> None:
        """Emite um bloco PROFILING no log com o tempo acumulado por fase."""
        if not self._phase_times:
            return
        total = sum(self._phase_times.values())
        self._logger.info("[PROFILING  ] ── resumo de tempo por fase ──")
        for phase, t in sorted(self._phase_times.items(), key=lambda x: x[1], reverse=True):
            count = self._phase_counts.get(phase, 1)
            pct   = (t / total * 100) if total > 0 else 0.0
            self._logger.info(
                f"[PROFILING  ] fase={phase:<16} | total={t:.3f}s | "
                f"chamadas={count:>5} | media={(t/count)*1000:.2f}ms | {pct:.1f}%"
            )
        self._logger.info(f"[PROFILING  ] total_rastreado={total:.3f}s")

    # ------------------------------------------------------------------
    # Início / fim de execução
    # ------------------------------------------------------------------

    def log_run_start(
        self,
        n_pop: int,
        mut_pb: float,
        mut_indpb: float,
        target_improvement: float,
        target_cv: float,
        max_gen: int | None = None,
    ):
        max_gen_str = str(max_gen) if max_gen is not None else "ilimitado"
        self._logger.info(
            f"[RUN_START  ] algoritmo={self._algorithm} | n_pop={n_pop} | "
            f"max_gen={max_gen_str} | mut_pb={mut_pb:.2f} | "
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
        self.last_gen = gen  # persiste para uso posterior (ex.: log_comparison_summary)
        genes_str = (
            f" | genes=[{_decode_genes(best_ind)}]" if best_ind is not None else ""
        )
        self._log_profiling()
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

    # ------------------------------------------------------------------
    # Resumo final de comparação
    # ------------------------------------------------------------------

    def log_comparison_summary(
        self,
        best_ind,
        cv_fitness: float,
        params: dict,
        model_path: str,
        acc_optimized: float,
        report_optimized_str: str,
        acc_original: float | None,
        report_original_str: str | None,
        gen: int,
        elapsed: float,
        meta_atingida: bool,
        n_pop: int | None = None,
        max_gen: int | None = None,
        mut_pb: float | None = None,
        mut_indpb: float | None = None,
        target_improvement: float | None = None,
        target_cv: float | None = None,
    ) -> str:
        """Grava um arquivo de resumo legível comparando o modelo original
        (Fase 1) com o modelo otimizado pelo AG.

        Usa I/O direto (``open``), independente dos handlers do módulo
        ``logging`` (que podem já ter sido fechados por ``log_run_end``).

        Returns:
            Caminho absoluto do arquivo de resumo criado.
        """
        summary_path = os.path.join(
            _LOGS_DIR, f"summary_{self._algorithm}_{self._run_id}.txt"
        )
        SEP  = "=" * 80
        SEP2 = "-" * 80
        max_gen_str = str(max_gen) if max_gen is not None else "ilimitado"
        lines = [
            SEP,
            "  RELATÓRIO FINAL — Comparação entre modelo original (Fase 1) e modelo AG",
            SEP,
            f"  Algoritmo        : {self._algorithm}",
            f"  Timestamp        : {self._run_id}",
            f"  Duração          : {elapsed:.1f}s",
            f"  Gerações         : {gen}",
            f"  Meta atingida    : {'SIM' if meta_atingida else 'NAO'}",
            "",
            SEP2,
            "  Parâmetros da evolução genética",
            SEP2,
            f"  {'n_pop':<24}: {n_pop if n_pop is not None else 'N/D'}",
            f"  {'max_gen':<24}: {max_gen_str}",
            f"  {'mut_pb':<24}: {mut_pb:.2f}" if mut_pb is not None else f"  {'mut_pb':<24}: N/D",
            f"  {'mut_indpb':<24}: {mut_indpb:.2f}" if mut_indpb is not None else f"  {'mut_indpb':<24}: N/D",
            f"  {'target_improvement':<24}: {target_improvement * 100:.2f}%" if target_improvement is not None else f"  {'target_improvement':<24}: N/D",
            f"  {'target_cv':<24}: {target_cv:.4f}" if target_cv is not None else f"  {'target_cv':<24}: N/D",
            "",
            SEP2,
            "  Melhor indivíduo — hiperparâmetros otimizados",
            SEP2,
        ]
        for k, v in params.items():
            lines.append(f"  {k:<24}: {v}")
        lines += [
            f"  {'CV accuracy (treino, 5-fold)':<24}: {cv_fitness:.4f}",
            "",
            SEP2,
            "  Avaliação no conjunto de teste",
            SEP2,
            f"  {'Modelo':<35} {'Acurácia':>10}",
            f"  {'-'*47}",
        ]
        if acc_original is not None:
            delta = acc_optimized - acc_original
            lines.append(f"  {'Original (Fase 1)':<35} {acc_original:>10.4f}")
            lines.append(f"  {'Otimizado (AG)':<35} {acc_optimized:>10.4f}   ({delta:+.4f})")
        else:
            lines.append(f"  {'Original (Fase 1)':<35} {'N/D':>10}")
            lines.append(f"  {'Otimizado (AG)':<35} {acc_optimized:>10.4f}")
        lines += [
            "",
            SEP2,
            "  Relatório de classificação — Modelo original (Fase 1)",
            SEP2,
        ]
        if report_original_str is not None:
            for ln in report_original_str.splitlines():
                lines.append(f"  {ln}")
        else:
            lines.append("  (modelo original não disponível)")
        lines += [
            "",
            SEP2,
            "  Relatório de classificação — Modelo otimizado (AG)",
            SEP2,
        ]
        for ln in report_optimized_str.splitlines():
            lines.append(f"  {ln}")
        lines += [
            "",
            SEP2,
            "  Arquivos gerados",
            SEP2,
            f"  Modelo exportado : {os.path.normpath(model_path)}",
            f"  Log de execução  : {os.path.normpath(self.log_file)}",
            f"  Este resumo      : {os.path.normpath(summary_path)}",
        ]
        if self._phase_times:
            total_t = sum(self._phase_times.values())
            lines += [
                "",
                SEP2,
                "  Profiling — tempo por fase",
                SEP2,
                f"  {'Fase':<18} {'Total (s)':>10} {'Chamadas':>10} {'Média (ms)':>12} {'%':>6}",
                f"  {'-'*58}",
            ]
            for phase, t in sorted(self._phase_times.items(), key=lambda x: x[1], reverse=True):
                count = self._phase_counts.get(phase, 1)
                pct   = (t / total_t * 100) if total_t > 0 else 0.0
                lines.append(
                    f"  {phase:<18} {t:>10.3f} {count:>10} {(t/count)*1000:>12.2f} {pct:>6.1f}%"
                )
            lines.append(f"  {'TOTAL':<18} {total_t:>10.3f}")
        lines += [
            SEP,
            "",
        ]
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return summary_path
