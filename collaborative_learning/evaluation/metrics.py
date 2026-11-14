"""
Métriques d'évaluation : accuracy, NLL, ECE, par agent, du consensus et de la moyenne simple.
"""
from typing import Any, Dict

import numpy as np


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error sur n_bins intervalles de confiance égaux."""
    conf = probs.max(axis=-1)
    correct = (probs.argmax(axis=-1) == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def prob_metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p_true = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return {
        "accuracy": float((probs.argmax(axis=-1) == y).mean()),
        "nll": float(-np.log(p_true).mean()),
        "ece": expected_calibration_error(probs, y),
    }


def evaluate_detailed(model, x: np.ndarray, y: np.ndarray,
                      batch_size: int = 256) -> Dict[str, Any]:
    """
    Accuracy, NLL et ECE de chaque agent, du consensus configuré et de la
    moyenne simple des agents.

    Returns:
        dict avec 'agents' (liste de dicts), 'consensus' (dict), 'mean_ensemble' (dict),
        et 'agents_probabilities' (N, K, C) pour construire des ensembles a posteriori.
    """
    res = model.predict(x, batch_size=batch_size, verbose=0,
                        return_agents=True, return_probabilities=True)
    agent_probs = np.asarray(res["agents_probabilities"], dtype=np.float64)
    out = {
        "agents": [prob_metrics(agent_probs[:, k], y) for k in range(agent_probs.shape[1])],
        "mean_ensemble": prob_metrics(agent_probs.mean(axis=1), y),
        "agents_probabilities": agent_probs.astype(np.float32),
    }
    if "consensus_probabilities" in res:
        out["consensus"] = prob_metrics(np.asarray(res["consensus_probabilities"], np.float64), y)
    else:  # stratégies à labels durs (vote, majority, k_fold)
        out["consensus"] = {"accuracy": float((np.asarray(res["consensus"]) == y).mean())}
    return out


def disagreement(probs: np.ndarray) -> float:
    """Proportion moyenne d'exemples où deux agents prédisent une classe différente. probs : (N, K, C)."""
    preds = probs.argmax(-1)
    k = preds.shape[1]
    pairs = [(preds[:, i] != preds[:, j]).mean() for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(pairs)) if pairs else 0.0
