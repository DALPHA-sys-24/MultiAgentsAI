"""
Fonctions d'expérience pour le système multi-agents (agents ResNet / MobileNet).

- run_multi_experiment : entraîne et évalue un MultiAgentModel (reprise sur checkpoint possible) ;
- run_single_reference : un agent seul, même boucle d'entraînement ;
- count_trainable_params / match_width : taille des agents ;
- benchmark_architectures : référence vs multi-agents sur plusieurs architectures.

Le benchmark compare, pour chaque architecture :
    - une référence : l'agent entraîné seul (MultiAgentModel à 1 agent, donc
      exactement la même boucle, le même optimiseur et la même augmentation) ;
    - le système multi-agents (loss_type / strategy au choix) ;
    - un ensemble de K références indépendantes (même coût que K agents, sans interaction).
"""
import gc
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from dml.training.checkpoint import TrainingCheckpoint
from dml.data import make_eval_dataset, make_train_dataset
from dml.evaluation.metrics import evaluate_detailed
from dml.models import AgentNetwork
from dml.core.multiagent import MultiAgentModel
from dml.training.optimizers import DEFAULT_WD_EXCLUDE, build_agent_optimizers

log = logging.getLogger(__name__)

EVAL_ON = ("final", "best")


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
    form: str = "multiplicative",
    single_agent_loss: str = "ce",
    epochs: int = 200,
    batch_size: int = 128,
    optimizer: Union[str, List[str]] = "sgd",
    lr: Union[float, List[float]] = 0.1,
    weight_decay: Union[float, List[float]] = 5e-4,
    wd_exclude: Optional[Sequence[str]] = DEFAULT_WD_EXCLUDE,
    warmup_epochs: int = 5,
    optimizer_kwargs: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
    augment: bool = True,
    per_agent_views: bool = False,
    dropout_rate: float = 0.0,
    seed: Optional[int] = None,
    jit_compile: Union[bool, str] = "auto",
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 1,
    eval_on: str = "final",
    csv_log: Optional[str] = None,
    callbacks: Optional[list] = None,
    evaluate_test: bool = True,
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
        wd_exclude: motifs de noms de variables sans weight decay (défaut BatchNorm et biais).
        per_agent_views: True = chaque agent voit une augmentation différente du même
            exemple ; False = même image pour tous (DML classique).
        checkpoint_dir: dossier de sauvegarde (voir checkpoint.py). S'il contient un état,
            l'entraînement reprend là où il s'était arrêté. None = pas de sauvegarde.
        eval_on: "final" (poids de la dernière époque) ou "best" (meilleure val_accuracy,
            nécessite checkpoint_dir et un ensemble de validation).
        csv_log: fichier CSV de l'historique par époque (CSVLogger Keras).
        evaluate_test: False = ne touche pas au test (recherche d'hyperparamètres) ; renvoie
            alors seulement l'historique et les infos de validation.
        Les autres arguments sont ceux de MultiAgentModel.compile.

    Returns:
        dict avec 'accuracy_consensus', 'accuracy_agents' (liste), 'accuracy_agents_median',
        'accuracy_mean_ensemble', les NLL/ECE, 'history' (dict), 'best_epoch',
        'evaluated_on' et 'test_probabilities'.
    """
    if eval_on not in EVAL_ON:
        raise ValueError(f"eval_on inconnu : {eval_on}. Choix : {EVAL_ON}")
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
            optimizer_kwargs=optimizer_kwargs, wd_exclude=wd_exclude,
        ),
        loss_type=loss_type,
        strategy=strategy,
        alpha=alpha,
        temperature=temperature,
        form=form,
        single_agent_loss=single_agent_loss,
        eval_seed=seed or 0,
        jit_compile=jit_compile,
    )
    # Construit tous les poids (backbones puis têtes) avant une éventuelle reprise
    image_shape = tuple(data["x_train"].shape[1:])
    model(tf.zeros((1, num_agents) + image_shape if per_agent_views else (1,) + image_shape), training=False)

    callbacks = list(callbacks or [])
    ckpt, initial_epoch = None, 0
    if checkpoint_dir is not None:
        ckpt = TrainingCheckpoint(checkpoint_dir, monitor="val_accuracy", every=checkpoint_every)
        initial_epoch = ckpt.restore(model)
        callbacks.append(ckpt)
    if csv_log is not None:
        os.makedirs(os.path.dirname(os.path.abspath(csv_log)), exist_ok=True)
        callbacks.append(tf.keras.callbacks.CSVLogger(csv_log, append=initial_epoch > 0))
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())

    history = {}
    if initial_epoch < epochs:
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, initial_epoch=initial_epoch,
                      callbacks=callbacks, verbose=verbose)
        history = h.history
    if ckpt is not None:
        history = ckpt.history  # historique complet, y compris les époques d'avant une reprise
        ckpt.save_final(model)

    evaluated_on = "final"
    if eval_on == "best":
        if ckpt is not None and ckpt.load_best(model):
            evaluated_on = "best"
        else:
            log.warning("eval_on='best' sans checkpoint_dir ni validation : évaluation sur les poids finaux.")

    if not evaluate_test:
        return {
            "archs": model.archs,
            "num_agents": num_agents,
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
            "best_epoch": ckpt.best_epoch if ckpt is not None else None,
            "best_val_accuracy": ckpt.best if ckpt is not None else None,
        }

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
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        "evaluated_on": evaluated_on,
        "best_epoch": ckpt.best_epoch if ckpt is not None else None,
        "best_val_accuracy": ckpt.best if ckpt is not None else None,
        "test_probabilities": ev["agents_probabilities"],
    }
    return result


def run_single_reference(data: Dict[str, Any], *, arch: str, **kwargs) -> Dict[str, Any]:
    """
    Référence : un seul agent, entraîné avec la même boucle que le système multi-agents.

    Sa loss est la cross-entropy (single_agent_loss="ce", défaut) ou la même forme que
    celle des agents avec I_k = 0 (single_agent_loss="form", soit L_k² en multiplicatif).
    """
    kwargs.pop("num_agents", None)
    res = run_multi_experiment(data, archs=[arch], **kwargs)
    return {
        "arch": arch,
        "accuracy_reference": res["accuracy_agents"][0],
        "nll_reference": res["nll_agents"][0],
        "ece_reference": res["ece_agents"][0],
        "history": res["history"],
        "evaluated_on": res["evaluated_on"],
        "best_epoch": res["best_epoch"],
        "test_probabilities": res["test_probabilities"][:, 0],
    }


# =============================================================================
# BENCHMARK
# =============================================================================

def cleanup():
    """Libère les modèles TensorFlow entre deux runs (sinon la RAM grossit à chaque run)."""
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
            cleanup()

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
            cleanup()
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
