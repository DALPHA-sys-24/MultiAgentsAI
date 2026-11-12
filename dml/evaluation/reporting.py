"""
Résumés sur les graines (médiane et quantiles 5 % / 95 %) et tableaux affichés en fin de run.
Utilisé par les scripts de experiments/
"""
import json
import logging
import os
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)

QUANTILES = (0.05, 0.95)


def summarize(values) -> dict:
    """Médiane et quantiles 5 % / 95 % sur les graines (moyenne et écart-type gardés pour info)."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    lo, hi = np.quantile(values, QUANTILES)
    return {"median": float(np.median(values)), "q05": float(lo), "q95": float(hi),
            "mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}


def fmt_summary(m) -> str:
    return f"{m['median']:.4f} [{m['q05']:.4f};{m['q95']:.4f}]" if m else "-"


def write_json(path: str, payload: dict) -> None:
    """Écriture atomique : un results.json n'est jamais à moitié écrit si le job est tué."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


# =============================================================================
# COLLABORATION VS GRAND MODÈLE
# =============================================================================

def summary_collab(runs: list) -> dict:
    """Médiane et quantiles 5 % / 95 % sur les graines des métriques à comparer."""
    rows = {
        "Grand agent seul": ([r["big_agent"]["accuracy"] for r in runs],
                             [r["big_agent"]["nll"] for r in runs],
                             [r["big_agent"]["ece"] for r in runs]),
        "Consensus des petits agents": ([r["small_agents"]["accuracy_consensus"] for r in runs],
                                        [r["small_agents"]["nll_consensus"] for r in runs],
                                        [r["small_agents"]["ece_consensus"] for r in runs]),
        "Moyenne simple des petits": ([r["small_agents"]["accuracy_mean_ensemble"] for r in runs], [], []),
        "Petit agent (médiane)": ([r["small_agents"]["accuracy_agent_median"] for r in runs],
                                  [float(np.median(r["small_agents"]["nll_agents"])) for r in runs],
                                  [float(np.median(r["small_agents"]["ece_agents"])) for r in runs]),
        "Meilleur petit agent": ([r["small_agents"]["accuracy_agent_best"] for r in runs], [], []),
    }
    return {name: {"accuracy": summarize(acc),
                   "nll": summarize(nll) if nll else None,
                   "ece": summarize(ece) if ece else None}
            for name, (acc, nll, ece) in rows.items()}


def print_collab_table(summary: dict, config, sizes: dict, n_seeds: int) -> None:
    width = 104
    lines = [
        "=" * width,
        f"{config.num_agents} x {config.small_arch} ({sizes['small_params_total']:,} param.) "
        f"vs grand agent width={sizes['big_width']} ({sizes['big_params']:,} param.) | {n_seeds} graine(s)",
        "médiane [quantile 5 % ; quantile 95 %] sur les graines",
        "-" * width,
        f"{'':30s} {'accuracy':>24s} {'NLL':>24s} {'ECE':>24s}",
    ]
    for name, m in summary.items():
        lines.append(f"{name:30s} {fmt_summary(m['accuracy']):>24s} {fmt_summary(m['nll']):>24s} "
                     f"{fmt_summary(m['ece']):>24s}")
    lines.append("=" * width)
    log.info("\n" + "\n".join(lines))


# =============================================================================
# BALAYAGE DE ALPHA
# =============================================================================

SWEEP_METRICS = {
    "Consensus": "accuracy_consensus",
    "Moyenne simple": "accuracy_mean_ensemble",
    "Agent (médiane)": "accuracy_agent_median",
    "Meilleur agent": "accuracy_agent_best",
    "Désaccord": "disagreement",
}
METRICS = SWEEP_METRICS  # ancien nom


def summary_by_alpha(runs: List[dict]) -> Dict[str, dict]:
    """Médiane et quantiles 5 % / 95 % sur les graines, pour chaque alpha."""
    out = {}
    for alpha in sorted({r["alpha"] for r in runs}):
        rs = [r for r in runs if r["alpha"] == alpha]
        out[f"{alpha:g}"] = {
            "alpha": alpha,
            "strategy": rs[0]["strategy"],
            "n_seeds": len(rs),
            **{key: summarize([r[key] for r in rs]) for key in SWEEP_METRICS.values()},
            "nll_agent_median": summarize([float(np.median(r["nll_agents"])) for r in rs]),
            "ece_agent_median": summarize([float(np.median(r["ece_agents"])) for r in rs]),
        }
    return out


def print_sweep_table(summary: dict, num_agents: int, arch: str, loss_type: str) -> None:
    width = 18 + 25 * len(SWEEP_METRICS)
    lines = [
        "=" * width,
        f"{num_agents} x {arch} | {loss_type} | médiane [quantile 5 % ; quantile 95 %] sur les graines",
        "-" * width,
        f"{'alpha (strat.)':18s}" + "".join(f"{name:>25s}" for name in SWEEP_METRICS),
    ]
    for s in summary.values():
        label = f"{s['alpha']:g} ({s['strategy'][:4]})"
        lines.append(f"{label:18s}" + "".join(f"{fmt_summary(s[key]):>25s}" for key in SWEEP_METRICS.values()))
    lines.append("=" * width)
    log.info("\n" + "\n".join(lines))
