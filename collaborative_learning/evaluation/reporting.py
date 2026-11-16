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
    """
    Écriture atomique : un fichier n'est jamais à moitié écrit si le job est tué, et plusieurs
    tâches du job array peuvent écrire le même fichier en même temps (fichier temporaire par processus).
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


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
