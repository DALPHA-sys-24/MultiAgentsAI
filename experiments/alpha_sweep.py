"""
Expérience « impact de la collaboration » : balayage du poids alpha.

Les K agents (même architecture) sont entraînés avec la loss multi-agents
L_k * (L_k + alpha * I_k) pour chaque valeur de alpha :
    - alpha = 0 : pas d'interaction, chaque agent est entraîné seul (loss L_k²) ;
      le consensus est un vote majoritaire (strategy = "majority") ;
    - alpha > 0 : collaboration de plus en plus forte ; le consensus utilise la
      stratégie du fichier de config (par défaut "simplicial").

Les réglages sont dans dml/utils/config.py (AlphaSweepConfig) et le fichier JSON ; chaque option
de la ligne de commande le surcharge (python -m experiments.alpha_sweep --help pour la liste).

Exemples :
    python -m experiments.alpha_sweep
    python -m experiments.alpha_sweep --config configs/alpha_sweep.json --dataset fashion_mnist --epochs 50
    python -m experiments.alpha_sweep --alphas 0 1 10 --seeds 0 --cpu --train-subset 2000 --epochs 2 --warmup-epochs 1
    python -m experiments.alpha_sweep --resume results/20261019-114342_alpha_sweep   # reprend un sweep interrompu
    python -m experiments.alpha_sweep --alphas 10 --seeds 2 --run-dir results/campagne/jobs/a10_s2   # un job du cluster

Chaque lancement crée results/<date>_<nom>/ (ou le dossier donné par --run-dir) contenant :
    config.json     config utilisée, nombre de paramètres, environnement (commit Git, versions)
    results.json    métriques par (alpha, graine) et résumé par alpha (médiane, quantiles 5 % / 95 %)
    probs/          probabilités de test des agents (.npy), une par (alpha, graine)
    history/        historique par époque (.csv) de chaque entraînement
    checkpoints/    état de reprise et poids (meilleure époque, finaux) de chaque entraînement
    log.txt         journal du run

Un job relancé dans le même dossier saute les couples (alpha, graine) terminés et reprend
l'entraînement en cours à sa dernière époque.
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from dml.utils.config import (STRATEGY_NO_INTERACTION, AlphaSweepConfig, add_config_arguments, config_from_dict,  # noqa: F401
                    environment_info, load_config, overrides_from_args, save_config)
from dml.data import load_dataset
from dml.utils.device import configure_device, default_device
from dml.training.runner import cleanup, count_trainable_params, run_multi_experiment
from dml.utils.log import setup_logging
from dml.evaluation.metrics import disagreement
from dml.evaluation.reporting import METRICS, print_sweep_table, summary_by_alpha, write_json  # noqa: F401  (compatibilité)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "configs", "alpha_sweep.json")

log = setup_logging()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Fichier de config JSON (défaut : configs/alpha_sweep.json "
                                     "à côté du script s'il existe, sinon valeurs par défaut).")
    p.add_argument("--resume", metavar="DOSSIER",
                   help="Reprend un sweep interrompu dans ce dossier de résultats, avec sa config ; "
                        "les couples (alpha, graine) déjà faits sont sautés.")
    p.add_argument("--run-dir", metavar="DOSSIER",
                   help="Dossier de sortie fixe au lieu de results/<date>_<nom> (jobs du cluster) ; "
                        "les couples (alpha, graine) déjà présents dedans sont sautés.")
    p.add_argument("--cpu", action="store_true", help="Force le CPU même si un GPU est disponible.")
    p.add_argument("--verbose", type=int, default=2)
    add_config_arguments(p, AlphaSweepConfig)
    return p.parse_args()


def read_config(args) -> AlphaSweepConfig:
    if args.resume:
        # Reprise : la config enregistrée du sweep fait foi, sans autres surcharges
        path = os.path.join(args.resume, "config.json")
        if not os.path.exists(path):
            raise SystemExit(f"Pas de config.json dans {os.path.abspath(args.resume)}")
        with open(path) as f:
            log.info("[reprise] %s", os.path.abspath(args.resume))
            return config_from_dict(AlphaSweepConfig, json.load(f), strict=False).validate()
    path = args.config or (DEFAULT_CONFIG_PATH if os.path.exists(DEFAULT_CONFIG_PATH) else None)
    log.info("[config] %s", os.path.abspath(path) if path else "aucun fichier, valeurs par défaut")
    return load_config(AlphaSweepConfig, path, overrides_from_args(args, AlphaSweepConfig))


def strategy_for(alpha: float, config) -> str:
    """Compatibilité : voir AlphaSweepConfig.strategy_for."""
    return config.strategy_for(alpha)


# =============================================================================
# EXPÉRIENCE
# =============================================================================

def run_one(data: dict, cfg: AlphaSweepConfig, alpha: float, seed: int, verbose: int, run_dir: str) -> dict:
    """Entraîne les K agents pour un couple (alpha, graine)."""
    strategy = cfg.strategy_for(alpha)
    tag = f"a{alpha:g}_s{seed}"
    log.info("=== Graine %d | alpha = %g | %d x %s (%s + %s) ===", seed, alpha, cfg.num_agents, cfg.arch,
             cfg.loss_type, strategy)
    res = run_multi_experiment(
        data, archs=cfg.arch, num_agents=cfg.num_agents, seed=seed,
        loss_type=cfg.loss_type, strategy=strategy,
        alpha=alpha, temperature=cfg.temperature,
        per_agent_views=cfg.per_agent_views,
        optimizer=cfg.optimizer, lr=cfg.lr, optimizer_kwargs=cfg.optimizer_kwargs,
        checkpoint_dir=os.path.join(run_dir, "checkpoints", tag),
        csv_log=os.path.join(run_dir, "history", f"{tag}.csv"),
        **cfg.train_kwargs(verbose),
    )
    probs = res.pop("test_probabilities")
    np.save(os.path.join(run_dir, "probs", f"agents_alpha{alpha:g}_s{seed}.npy"), probs)

    accs = res["accuracy_agents"]
    return {
        "alpha": alpha,
        "strategy": strategy,
        "seed": seed,
        "accuracy_consensus": res["accuracy_consensus"],
        "accuracy_mean_ensemble": res["accuracy_mean_ensemble"],
        "accuracy_agents": accs,
        "accuracy_agent_median": float(np.median(accs)),
        "accuracy_agent_best": float(np.max(accs)),
        "nll_consensus": res["nll_consensus"],     # None pour un consensus à labels durs (majority)
        "ece_consensus": res["ece_consensus"],
        "nll_agents": res["nll_agents"],
        "ece_agents": res["ece_agents"],
        "disagreement": disagreement(probs),
        "evaluated_on": res["evaluated_on"],
        "best_epoch": res["best_epoch"],
        "history": res["history"],
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

    params = count_trainable_params(cfg.arch, data["input_shape"], data["num_classes"])
    log.info("[taille] %d x %s : %s param. chacun", cfg.num_agents, cfg.arch, f"{params:,}")
    log.info("[alphas] %s", ", ".join(f"{a:g} ({cfg.strategy_for(a)})" for a in cfg.alphas))

    # Dossier de sortie : celui d'un sweep à reprendre, un dossier fixe (cluster) ou un nouveau dossier daté
    run_dir = (args.resume or args.run_dir
               or os.path.join(cfg.out_dir, f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.name}"))
    os.makedirs(os.path.join(run_dir, "probs"), exist_ok=True)
    setup_logging(run_dir)
    if not args.resume:
        save_config(run_dir, cfg, device=device, agent_params=params,
                    strategy_alpha0=STRATEGY_NO_INTERACTION, environment=environment_info(device))

    # Runs déjà terminés dans ce dossier (reprise après un arrêt) : on les garde et on les saute
    runs, elapsed = [], 0.0
    results_path = os.path.join(run_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            previous = json.load(f)
        runs, elapsed = previous["runs"], previous.get("train_time_s", 0.0)
        log.info("[reprise] %d run(s) déjà terminé(s) dans %s", len(runs), os.path.abspath(run_dir))

    done = {(float(r["alpha"]), r["seed"]) for r in runs}
    start = time.time() - elapsed
    for seed in cfg.seeds:
        for alpha in cfg.alphas:
            if (float(alpha), seed) in done:
                continue
            runs.append(run_one(data, cfg, float(alpha), seed, args.verbose, run_dir))
            cleanup()  # libère les modèles du run (sinon la RAM grossit jusqu'au « Killed »)
            # Réécrit après chaque run : les résultats partiels sont gardés si le job s'arrête
            write_json(results_path, {"experiment": "alpha_sweep", "summary": summary_by_alpha(runs),
                                      "runs": runs, "agent_params": params,
                                      "train_time_s": time.time() - start})

    print_sweep_table(summary_by_alpha(runs), cfg.num_agents, cfg.arch, cfg.loss_type)
    log.info("Résultats : %s", run_dir)


if __name__ == "__main__":
    # Tout le code principal tourne sur le GPU s'il est visible, sinon sur le CPU
    with tf.device(default_device()):
        main()
