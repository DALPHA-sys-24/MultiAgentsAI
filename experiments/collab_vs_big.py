"""
Expérience « collaboration vs grand modèle ».

Compare, à nombre de paramètres entraînables égal :
    - K petits agents (même architecture) entraînés ensemble avec la loss multi-agents
      (par défaut global_interaction + simplicial) ;
    - un grand agent seul, de la même architecture élargie (width) pour avoir autant
      de paramètres que les K petits réunis. Sa loss est la cross-entropy
      (reference_loss = "ce") ou la même forme que les petits agents avec I_k = 0
      (reference_loss = "same_as_agents", soit L_k² en multiplicatif).

Les réglages sont dans dml/utils/config.py (CollabVsBigConfig) et le fichier JSON ; chaque option
de la ligne de commande le surcharge (python -m experiments.collab_vs_big --help pour la liste).

Exemples :
    python -m experiments.collab_vs_big --config configs/collab_vs_big.json
    python -m experiments.collab_vs_big --config configs/collab_vs_big.json --dataset fashion_mnist --epochs 50
    python -m experiments.collab_vs_big --cpu --train-subset 2000 --epochs 1 --seeds 0
    python -m experiments.collab_vs_big --run-dir results/collab_cifar10       # dossier fixe (cluster)
    python -m experiments.collab_vs_big --resume results/20261112-101500_collab_vs_big

Chaque lancement crée results/<date>_<nom>/ (ou le dossier --run-dir) contenant :
    config.json     config utilisée, tailles des modèles, environnement (commit Git, versions)
    results.json    métriques par graine (petits agents, consensus, grand agent) et résumé
    probs/          probabilités de test (.npy) par graine
    history/        historique par époque (.csv) de chaque entraînement
    checkpoints/    état de reprise et poids (meilleure époque, finaux) de chaque entraînement
    log.txt         journal du run

Si le job s'arrête, relancer la même commande avec --run-dir (ou --resume) : les graines
terminées sont gardées et l'entraînement en cours reprend à sa dernière époque.
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from dml.utils.config import (CollabVsBigConfig, add_config_arguments, config_from_dict, environment_info, load_config,
                    overrides_from_args, save_config)
from dml.data import load_dataset
from dml.utils.device import configure_device, default_device
from dml.training.runner import cleanup, count_trainable_params, match_width, run_multi_experiment, run_single_reference
from dml.utils.log import setup_logging
from dml.evaluation.reporting import print_collab_table, summary_collab, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "configs", "collab_vs_big.json")

log = setup_logging()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Fichier de config JSON (défaut : configs/collab_vs_big.json "
                                     "à côté du script s'il existe, sinon valeurs par défaut).")
    p.add_argument("--run-dir", metavar="DOSSIER",
                   help="Dossier de sortie fixe au lieu de results/<date>_<nom> ; un run interrompu dedans reprend.")
    p.add_argument("--resume", metavar="DOSSIER",
                   help="Reprend un run interrompu dans ce dossier, avec la config enregistrée.")
    p.add_argument("--cpu", action="store_true", help="Force le CPU même si un GPU est disponible.")
    p.add_argument("--verbose", type=int, default=2)
    add_config_arguments(p, CollabVsBigConfig)
    return p.parse_args()


def read_config(args) -> CollabVsBigConfig:
    if args.resume:
        path = os.path.join(args.resume, "config.json")
        if not os.path.exists(path):
            raise SystemExit(f"Pas de config.json dans {os.path.abspath(args.resume)}")
        with open(path) as f:
            log.info("[reprise] %s", os.path.abspath(args.resume))
            return config_from_dict(CollabVsBigConfig, json.load(f), strict=False).validate()
    path = args.config or (DEFAULT_CONFIG_PATH if os.path.exists(DEFAULT_CONFIG_PATH) else None)
    log.info("[config] %s", os.path.abspath(path) if path else "aucun fichier, valeurs par défaut")
    return load_config(CollabVsBigConfig, path, overrides_from_args(args, CollabVsBigConfig))


# =============================================================================
# EXPÉRIENCE
# =============================================================================

def _cached(path: str, probs_path: str, train_fn):
    """Résultat d'un entraînement déjà terminé (reprise), sinon l'entraîne et l'enregistre."""
    if os.path.exists(path) and os.path.exists(probs_path):
        log.info("[reprise] déjà terminé : %s", os.path.basename(path))
        with open(path) as f:
            return json.load(f)
    res = train_fn()
    np.save(probs_path, res.pop("test_probabilities"))
    write_json(path, res)
    return res


def run_seed(data: dict, cfg: CollabVsBigConfig, big: dict, seed: int, verbose: int, run_dir: str) -> dict:
    """Entraîne les K petits agents puis le grand agent pour une graine."""
    common = cfg.train_kwargs(verbose)
    k = cfg.num_agents
    probs_dir, parts_dir = os.path.join(run_dir, "probs"), os.path.join(run_dir, "checkpoints")

    def paths(tag):
        return dict(checkpoint_dir=os.path.join(parts_dir, f"{tag}_s{seed}"),
                    csv_log=os.path.join(run_dir, "history", f"{tag}_s{seed}.csv"))

    log.info("=== Graine %d : %d x %s (%s + %s, alpha=%g) ===", seed, k, cfg.small_arch,
             cfg.loss_type, cfg.strategy, cfg.alpha)
    collab = _cached(
        os.path.join(parts_dir, f"small_s{seed}.json"), os.path.join(probs_dir, f"small_agents_s{seed}.npy"),
        lambda: run_multi_experiment(
            data, archs=cfg.small_arch, num_agents=k, seed=seed,
            loss_type=cfg.loss_type, strategy=cfg.strategy, alpha=cfg.alpha, temperature=cfg.temperature,
            per_agent_views=cfg.per_agent_views,
            optimizer=cfg.optimizer, lr=cfg.lr, optimizer_kwargs=cfg.optimizer_kwargs,
            **paths("small"), **common,
        ))
    cleanup()  # libère les modèles avant le run suivant (sinon la RAM grossit à chaque run)

    big_opt = cfg.big_agent_optimizer()
    single_loss = "ce" if cfg.reference_loss == "ce" else "form"
    log.info("=== Graine %d : grand agent %s width=%s (%s, %s) ===", seed, cfg.small_arch, big["width"],
             "cross-entropy" if single_loss == "ce" else f"{cfg.loss_form}, I_k=0", big_opt["optimizer"])
    single = _cached(
        os.path.join(parts_dir, f"big_s{seed}.json"), os.path.join(probs_dir, f"big_agent_s{seed}.npy"),
        lambda: run_single_reference(
            data, arch=cfg.small_arch, widths=big["width"], seed=seed, single_agent_loss=single_loss,
            **big_opt, **paths("big"), **common,
        ))
    cleanup()

    accs = collab["accuracy_agents"]
    return {
        "seed": seed,
        "small_agents": {
            "accuracy_consensus": collab["accuracy_consensus"],
            "accuracy_mean_ensemble": collab["accuracy_mean_ensemble"],
            "accuracy_agents": accs,
            "accuracy_agent_median": float(np.median(accs)),
            "accuracy_agent_best": float(np.max(accs)),
            "nll_consensus": collab["nll_consensus"],
            "ece_consensus": collab["ece_consensus"],
            "nll_agents": collab["nll_agents"],
            "ece_agents": collab["ece_agents"],
            "evaluated_on": collab.get("evaluated_on"),
            "best_epoch": collab.get("best_epoch"),
            "history": collab["history"],
        },
        "big_agent": {
            "accuracy": single["accuracy_reference"],
            "nll": single["nll_reference"],
            "ece": single["ece_reference"],
            "evaluated_on": single.get("evaluated_on"),
            "best_epoch": single.get("best_epoch"),
            "history": single["history"],
        },
    }


def main():
    args = parse_args()
    cfg = read_config(args)
    device = configure_device(force_cpu=args.cpu, precision=cfg.precision)

    # Données : séparation train / validation fixe pour toutes les graines
    data = load_dataset(cfg.dataset, val_size=cfg.val_size, seed=0)
    if cfg.train_subset:
        data["x_train"] = data["x_train"][:cfg.train_subset]
        data["y_train"] = data["y_train"][:cfg.train_subset]

    # Largeur du grand agent : autant de paramètres que les K petits réunis
    shape, num_classes = data["input_shape"], data["num_classes"]
    small_params = count_trainable_params(cfg.small_arch, shape, num_classes)
    target = cfg.num_agents * small_params
    big = match_width(cfg.small_arch, target, shape, num_classes)
    sizes = {
        "small_params_each": small_params,
        "small_params_total": target,
        "big_width": big["width"],
        "big_params": big["params"],
        "params_relative_gap": big["relative_gap"],
    }
    log.info("[taille] petit agent : %s param. | %d agents : %s | grand agent width=%s : %s (%+.2f%%)",
             f"{small_params:,}", cfg.num_agents, f"{target:,}", big["width"], f"{big['params']:,}",
             100 * big["relative_gap"])

    run_dir = (args.resume or args.run_dir
               or os.path.join(cfg.out_dir, f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.name}"))
    os.makedirs(os.path.join(run_dir, "probs"), exist_ok=True)
    setup_logging(run_dir)
    if not args.resume:
        save_config(run_dir, cfg, device=device, model_sizes=sizes, environment=environment_info(device))

    results_path = os.path.join(run_dir, "results.json")
    runs, elapsed = [], 0.0
    if os.path.exists(results_path):
        with open(results_path) as f:
            previous = json.load(f)
        runs, elapsed = previous["runs"], previous.get("train_time_s", 0.0)
        log.info("[reprise] %d graine(s) déjà terminée(s)", len(runs))
    done = {r["seed"] for r in runs}

    start = time.time() - elapsed
    for seed in cfg.seeds:
        if seed in done:
            continue
        runs.append(run_seed(data, cfg, big, seed, args.verbose, run_dir))
        # Réécrit après chaque graine : les résultats partiels sont gardés si le job s'arrête
        write_json(results_path, {"summary": summary_collab(runs), "runs": runs, "model_sizes": sizes,
                                  "train_time_s": time.time() - start})

    print_collab_table(summary_collab(runs), cfg, sizes, len(runs))
    log.info("Résultats : %s", run_dir)


if __name__ == "__main__":
    # Tout le code principal tourne sur le GPU s'il est visible, sinon sur le CPU
    with tf.device(default_device()):
        main()
