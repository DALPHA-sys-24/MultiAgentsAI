"""
Fonctions d'expérience et de benchmark pour le système multi-agents
avec des agents ResNet / MobileNet.

Le benchmark compare, pour chaque architecture :
    - une référence : l'agent entraîné seul (MultiAgentModel à 1 agent, donc
      exactement la même boucle, le même optimiseur et la même augmentation) ;
    - le système multi-agents (loss_type / strategy au choix) ;
    - un ensemble de K références indépendantes (même coût que K agents, sans interaction).
"""
import gc
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from data import make_eval_dataset, make_train_dataset
from models import AgentNetwork
from multiagent import MultiAgentModel
from optimizers import build_agent_optimizers


# =============================================================================
# TAILLE DES MODÈLES
# =============================================================================

def count_trainable_params(arch: str, input_shape, num_classes: int, width: float = 1.0) -> int:
    """Nombre de paramètres entraînables d'un agent (backbone + tête)."""
    agent = AgentNetwork(arch=arch, num_classes=num_classes, width=width)
    agent.build((None,) + tuple(input_shape))
    agent(tf.zeros((1,) + tuple(input_shape)))  # construit aussi la tête
    n = int(sum(np.prod(w.shape) for w in agent.trainable_weights))
    del agent
    tf.keras.backend.clear_session()
    return n


def match_width(arch: str, target_params: int, input_shape, num_classes: int,
                max_width: float = 16.0, iters: int = 12) -> Dict[str, float]:
    """
    Cherche la largeur w telle que AgentNetwork(arch, width=w) ait target_params
    paramètres entraînables (recherche dichotomique ; le nombre de paramètres croît avec w).

    Returns:
        dict avec 'width', 'params' et 'relative_gap' (écart relatif à la cible).
    """
    cache: Dict[float, int] = {}

    def count(w):
        if w not in cache:
            cache[w] = count_trainable_params(arch, input_shape, num_classes, w)
        return cache[w]

    lo, hi = 1.0, 2.0
    while count(hi) < target_params:
        lo, hi = hi, hi * 2
        if hi > max_width:
            raise ValueError(f"Largeur > {max_width} nécessaire pour atteindre {target_params} paramètres.")
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if count(mid) < target_params else (lo, mid)
    best = min((lo, hi), key=lambda w: abs(count(w) - target_params))
    params = count(best)
    return {"width": round(best, 4), "params": params,
            "relative_gap": (params - target_params) / target_params}


# =============================================================================
# ÉVALUATION DÉTAILLÉE
# =============================================================================

def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=-1)
    correct = (probs.argmax(axis=-1) == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def _prob_metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p_true = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return {
        "accuracy": float((probs.argmax(axis=-1) == y).mean()),
        "nll": float(-np.log(p_true).mean()),
        "ece": expected_calibration_error(probs, y),
    }


def evaluate_detailed(model: MultiAgentModel, x: np.ndarray, y: np.ndarray,
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
        "agents": [_prob_metrics(agent_probs[:, k], y) for k in range(agent_probs.shape[1])],
        "mean_ensemble": _prob_metrics(agent_probs.mean(axis=1), y),
        "agents_probabilities": agent_probs.astype(np.float32),
    }
    if "consensus_probabilities" in res:
        out["consensus"] = _prob_metrics(np.asarray(res["consensus_probabilities"], np.float64), y)
    else:  # stratégies à labels durs (vote, majority, k_fold)
        out["consensus"] = {"accuracy": float((np.asarray(res["consensus"]) == y).mean())}
    return out


# =============================================================================
# EXPÉRIENCES UNITAIRES
# =============================================================================

def run_multi_experiment(
    data: Dict[str, Any],
    *,
    archs: Union[str, List[str]],
    num_agents: Optional[int] = None,
    widths: Union[float, List[float]] = 1.0,
    loss_type: str = "global_interaction",
    strategy: str = "simplicial",
    alpha: float = 1.0,
    temperature: float = 1.0,
    epochs: int = 200,
    batch_size: int = 128,
    optimizer: Union[str, List[str]] = "sgd",
    lr: Union[float, List[float]] = 0.1,
    weight_decay: Union[float, List[float]] = 5e-4,
    warmup_epochs: int = 5,
    optimizer_kwargs: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
    augment: bool = True,
    per_agent_views: bool = False,
    dropout_rate: float = 0.0,
    seed: Optional[int] = None,
    jit_compile: Union[bool, str] = "auto",
    callbacks: Optional[list] = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Entraîne et évalue un MultiAgentModel sur un jeu de données de data.load_dataset.

    Args:
        archs: str (K agents identiques, K = num_agents) ou liste de str (un par agent).
        num_agents: requis si archs est un str ; déduit de la liste sinon.
        widths: Multiplicateur de largeur des agents (float pour tous, ou liste).
        optimizer: Nom d'optimiseur (voir optimizers.available_optimizers()), le même
            pour tous les agents, ou liste d'un nom par agent. lr, weight_decay et
            optimizer_kwargs acceptent aussi une valeur commune ou une liste par agent.
        per_agent_views: True = chaque agent voit une augmentation différente du même
            exemple ; False = même image pour tous (DML classique).
        Les autres arguments sont ceux de MultiAgentModel.compile.

    Returns:
        dict avec 'accuracy_consensus', 'accuracy_agents' (liste), 'accuracy_agents_median',
        'accuracy_mean_ensemble', les NLL/ECE, 'history' (dict Keras) et 'test_probabilities'.
    """
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    if isinstance(archs, str):
        if num_agents is None:
            raise ValueError("num_agents est requis quand archs est un str.")
    else:
        num_agents = len(archs)

    train_ds = make_train_dataset(
        data["x_train"], data["y_train"], batch_size=batch_size, augment=augment,
        per_agent_views=num_agents if per_agent_views else None, seed=seed,
    )
    val_ds = make_eval_dataset(data["x_val"], data["y_val"]) if len(data["x_val"]) else None
    steps_per_epoch = len(data["x_train"]) // batch_size

    model = MultiAgentModel(
        num_agents=num_agents,
        archs=archs,
        widths=widths,
        num_classes=data["num_classes"],
        dropout_rate=dropout_rate,
        name="MultiAgent",
    )
    model.compile(
        optimizers_config=build_agent_optimizers(
            optimizer, num_agents, lr=lr, weight_decay=weight_decay, epochs=epochs,
            steps_per_epoch=steps_per_epoch, warmup_epochs=warmup_epochs,
            optimizer_kwargs=optimizer_kwargs,
        ),
        loss_type=loss_type,
        strategy=strategy,
        alpha=alpha,
        temperature=temperature,
        jit_compile=jit_compile,
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks, verbose=verbose)

    ev = evaluate_detailed(model, data["x_test"], data["y_test"])
    agent_accs = [a["accuracy"] for a in ev["agents"]]
    result = {
        "archs": model.archs,
        "num_agents": num_agents,
        "accuracy_consensus": ev["consensus"]["accuracy"],
        "accuracy_agents": agent_accs,
        "accuracy_agents_median": float(np.median(agent_accs)),
        "accuracy_mean_ensemble": ev["mean_ensemble"]["accuracy"],
        "nll_agents": [a["nll"] for a in ev["agents"]],
        "ece_agents": [a["ece"] for a in ev["agents"]],
        "nll_consensus": ev["consensus"].get("nll"),
        "ece_consensus": ev["consensus"].get("ece"),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "test_probabilities": ev["agents_probabilities"],
    }
    return result


def run_single_reference(data: Dict[str, Any], *, arch: str, **kwargs) -> Dict[str, Any]:
    """
    Référence : un seul agent entraîné avec la cross-entropy classique.

    Toutes les losses du MultiAgentModel se réduisent à la CE pour K = 1, donc la
    référence partage exactement la boucle d'entraînement du système multi-agents.
    """
    kwargs.pop("num_agents", None)
    res = run_multi_experiment(data, archs=[arch], **kwargs)
    return {
        "arch": arch,
        "accuracy_reference": res["accuracy_agents"][0],
        "nll_reference": res["nll_agents"][0],
        "ece_reference": res["ece_agents"][0],
        "history": res["history"],
        "test_probabilities": res["test_probabilities"][:, 0],
    }


# =============================================================================
# BENCHMARK
# =============================================================================

def _cleanup():
    tf.keras.backend.clear_session()
    gc.collect()


def benchmark_architectures(
    data: Dict[str, Any],
    *,
    configs: Sequence[Union[str, List[str]]],
    num_agents: int = 2,
    n_runs: int = 3,
    include_reference: bool = True,
    **train_kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Benchmark référence vs multi-agents sur une liste d'architectures.

    Args:
        configs: Liste de configurations. Un str = num_agents agents identiques
            (ex. "resnet20") ; une liste = agents hétérogènes (ex. ["resnet32", "mobilenetv2"]).
        num_agents: K pour les configurations homogènes.
        n_runs: Nombre de graines par configuration.
        include_reference: Entraîne aussi chaque architecture seule (n_runs graines).
            Si n_runs >= K, l'ensemble des K premières références est aussi évalué.
        train_kwargs: Transmis à run_multi_experiment (epochs, loss_type, strategy, alpha, ...).

    Returns:
        dict avec :
        - 'multi'     : une ligne par run multi-agents ;
        - 'reference' : une ligne par run de référence ;
        - 'summary'   : par configuration, médianes et gain moyen des agents sur leur référence.
    """
    configs = [[c] * num_agents if isinstance(c, str) else list(c) for c in configs]
    ref_archs = sorted({a for c in configs for a in c}) if include_reference else []
    y_test = data["y_test"]

    pbar = tqdm(total=len(ref_archs) * n_runs + len(configs) * n_runs)
    ref_records, ref_probs = [], {}
    for arch in ref_archs:
        for run in range(n_runs):
            pbar.set_description(f"ref {arch} run={run+1}")
            res = run_single_reference(data, arch=arch, seed=run, **train_kwargs)
            ref_probs.setdefault(arch, []).append(res.pop("test_probabilities"))
            res.pop("history")
            ref_records.append({"run": run, **res})
            pbar.update(1)
            _cleanup()

    multi_records = []
    for archs in configs:
        name = "+".join(archs) if len(set(archs)) > 1 else f"{archs[0]}x{len(archs)}"
        for run in range(n_runs):
            pbar.set_description(f"multi {name} run={run+1}")
            res = run_multi_experiment(data, archs=archs, seed=run, **train_kwargs)
            res.pop("test_probabilities")
            res.pop("history")
            multi_records.append({"config": name, "run": run, **res})
            pbar.update(1)
            _cleanup()
    pbar.close()

    df_ref = pd.DataFrame(ref_records)
    df_multi = pd.DataFrame(multi_records)

    ref_median = df_ref.groupby("arch")["accuracy_reference"].median().to_dict() if len(df_ref) else {}
    rows = []
    for name, grp in df_multi.groupby("config", sort=False):
        archs = grp["archs"].iloc[0]
        row = {
            "config": name,
            "acc_consensus_median": grp["accuracy_consensus"].median(),
            "acc_consensus_q05": grp["accuracy_consensus"].quantile(0.05),
            "acc_consensus_q95": grp["accuracy_consensus"].quantile(0.95),
            "acc_agents_median": grp["accuracy_agents_median"].median(),
            "acc_mean_ensemble_median": grp["accuracy_mean_ensemble"].median(),
        }
        if ref_median:
            gains = [np.mean([accs[k] - ref_median[a] for k, a in enumerate(archs)])
                     for accs in grp["accuracy_agents"]]
            row["gain_agents_vs_reference"] = float(np.median(gains))
            # Ensemble de K références indépendantes (même coût, sans interaction)
            pools = [ref_probs[a] for a in archs]
            if all(len(p) > k for k, p in enumerate(pools)) and len(set(archs)) == 1:
                ens = np.mean([pools[0][k] for k in range(len(archs))], axis=0)
                row["acc_independent_ensemble"] = float((ens.argmax(-1) == y_test).mean())
            elif all(len(p) >= 1 for p in pools) and len(set(archs)) > 1:
                ens = np.mean([pools[k][0] for k in range(len(archs))], axis=0)
                row["acc_independent_ensemble"] = float((ens.argmax(-1) == y_test).mean())
        rows.append(row)

    return {"multi": df_multi, "reference": df_ref, "summary": pd.DataFrame(rows)}
