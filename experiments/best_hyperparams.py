"""
Recherche des meilleurs hyperparamètres d'un modèle sur un jeu de données (Optuna).

Chaque essai tire des hyperparamètres dans `search_space`, entraîne les K agents et
mesure la métrique de validation (par défaut l'accuracy du consensus) à chaque époque.
Les essais mauvais sont arrêtés tôt (pruning) ; le test n'est jamais utilisé.

Réglages : configs/best_hyperparams.json (et HyperparamSearchConfig dans dml/utils/config.py) ;
chaque option de la ligne de commande le surcharge (--help pour la liste).

Exemples :
    python -m experiments.best_hyperparams --config configs/best_hyperparams.json
    python -m experiments.best_hyperparams --dataset cifar10 --arch resnet32 --num-agents 3 --n-trials 50
    python -m experiments.best_hyperparams --cpu --train-subset 2000 --epochs 3 --n-trials 4   # test rapide
    python -m experiments.best_hyperparams --run-dir results/hp_fmnist     # reprend / complète une recherche

Plusieurs processus (ou jobs SLURM, voir slurm/soumission_best_hyperparams.sh) lancés avec le
même --run-dir travaillent sur la même recherche : leurs essais s'additionnent.

Sorties dans results/<date>_<nom>/ (ou --run-dir) :
    config.json        config de la recherche et environnement (commit Git, versions)
    study.journal      l'étude Optuna (relancer avec le même --run-dir la complète)
    trials.csv         un essai par ligne : valeur, état (COMPLETE / PRUNED / FAIL), paramètres
    best.json          meilleur essai, ses paramètres, importance de chaque hyperparamètre
    best_config.json   config prête pour experiments.alpha_sweep avec les meilleurs paramètres
    trials/            historique par époque (.csv) de chaque essai
    log.txt            journal
"""
import argparse
import dataclasses
import json
import os
from datetime import datetime

import tensorflow as tf

from dml.data import load_dataset
from dml.training.runner import cleanup, count_trainable_params, run_multi_experiment
from dml.tuning.pruning import OptunaPruningCallback
from dml.tuning.space import decode_params, suggest
from dml.tuning.study import create_study, export_results
from dml.utils.config import (AlphaSweepConfig, HyperparamSearchConfig, add_config_arguments, environment_info,
                              load_config, overrides_from_args, save_config)
from dml.utils.device import configure_device, default_device
from dml.utils.log import setup_logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "configs", "best_hyperparams.json")

log = setup_logging()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Fichier de config JSON (défaut : configs/best_hyperparams.json).")
    p.add_argument("--run-dir", metavar="DOSSIER",
                   help="Dossier de la recherche ; s'il existe déjà, ses essais sont gardés et complétés.")
    p.add_argument("--cpu", action="store_true", help="Force le CPU même si un GPU est disponible.")
    p.add_argument("--verbose", type=int, default=0, help="Verbosité de Keras pendant les essais.")
    add_config_arguments(p, HyperparamSearchConfig)
    return p.parse_args()


def read_config(args) -> HyperparamSearchConfig:
    path = args.config or (DEFAULT_CONFIG_PATH if os.path.exists(DEFAULT_CONFIG_PATH) else None)
    log.info("[config] %s", os.path.abspath(path) if path else "aucun fichier, valeurs par défaut")
    cfg = load_config(HyperparamSearchConfig, path, overrides_from_args(args, HyperparamSearchConfig))
    if len(cfg.seeds) > 1:
        log.warning("seeds = %s : seule la graine %d est utilisée pendant la recherche.", cfg.seeds, cfg.seeds[0])
    return cfg


def make_objective(data: dict, cfg: HyperparamSearchConfig, run_dir: str, verbose: int):
    """Fonction objectif d'Optuna : un essai = un entraînement des K agents, sans le test."""
    import optuna

    def objective(trial):
        params = suggest(trial, cfg.search_space)
        tc = dataclasses.replace(cfg, **params).validate()
        log.info("[essai %d] %s", trial.number, json.dumps(params))
        pruning = OptunaPruningCallback(trial, cfg.metric, cfg.direction)
        try:
            run_multi_experiment(
                data, archs=tc.arch, num_agents=tc.num_agents, seed=tc.seeds[0],
                loss_type=tc.loss_type, strategy=tc.strategy, alpha=tc.alpha, temperature=tc.temperature,
                per_agent_views=tc.per_agent_views,
                optimizer=tc.optimizer, lr=tc.lr, optimizer_kwargs=tc.optimizer_kwargs,
                callbacks=[pruning], evaluate_test=False,
                csv_log=os.path.join(run_dir, "trials", f"trial_{trial.number:04d}.csv"),
                **tc.train_kwargs(verbose),
            )
        finally:
            cleanup()
        trial.set_user_attr("best_epoch", pruning.best_epoch)
        if pruning.pruned_at is not None:
            trial.set_user_attr("pruned_at_epoch", pruning.pruned_at)
            log.info("[essai %d] arrêté à l'époque %d (%s = %s)", trial.number, pruning.pruned_at,
                     cfg.metric, pruning.best)
            raise optuna.TrialPruned()
        log.info("[essai %d] %s = %.4f (époque %s)", trial.number, cfg.metric, pruning.best, pruning.best_epoch)
        return pruning.best

    return objective


def best_alpha_sweep_config(cfg: HyperparamSearchConfig, best_params: dict) -> dict:
    """Config experiments.alpha_sweep avec les meilleurs paramètres (alphas = [meilleur alpha])."""
    best = dataclasses.replace(cfg, **best_params)
    fields = {f.name for f in dataclasses.fields(AlphaSweepConfig)}
    out = {k: v for k, v in best.to_dict().items() if k in fields}
    out.update(name=f"{cfg.name}_best", alphas=[best.alpha], seeds=[0, 1, 2],
               _note="Meilleurs paramètres trouvés par experiments.best_hyperparams. "
                     "Penser à remettre le nombre d'époques final (epochs).")
    return out


def print_top(study, cfg: HyperparamSearchConfig, n: int = 10):
    import optuna
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    done.sort(key=lambda t: t.value, reverse=cfg.direction == "maximize")
    lines = [f"Meilleurs essais ({cfg.metric}, {len(done)} terminés / {len(study.trials)} lancés) :"]
    for t in done[:n]:
        lines.append(f"  #{t.number:<4d} {t.value:.4f}  " + "  ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in t.params.items()))
    log.info("\n".join(lines))


def main():
    import optuna  # noqa: F401  (erreur claire tout de suite si Optuna n'est pas installé)

    args = parse_args()
    cfg = read_config(args)
    device = configure_device(force_cpu=args.cpu, precision=cfg.precision)

    data = load_dataset(cfg.dataset, val_size=cfg.val_size, seed=0)
    if cfg.train_subset:
        data["x_train"] = data["x_train"][:cfg.train_subset]
        data["y_train"] = data["y_train"][:cfg.train_subset]
    params = count_trainable_params(cfg.arch, data["input_shape"], data["num_classes"])
    log.info("[taille] %d x %s : %s param. chacun | %s | critère : %s (%s)", cfg.num_agents, cfg.arch,
             f"{params:,}", cfg.dataset, cfg.metric, cfg.direction)
    log.info("[espace] %s", json.dumps(cfg.search_space))

    run_dir = args.run_dir or os.path.join(cfg.out_dir, f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.name}")
    os.makedirs(os.path.join(run_dir, "trials"), exist_ok=True)
    setup_logging(run_dir)
    if not os.path.exists(os.path.join(run_dir, "config.json")):   # le 1er job de la recherche l'écrit
        save_config(run_dir, cfg, device=device, agent_params=params, environment=environment_info(device))

    study = create_study(run_dir, cfg.name, cfg.direction, cfg.sampler, cfg.pruner, cfg.study_seed,
                         cfg.pruner_warmup_epochs, cfg.epochs)
    if study.trials:
        log.info("[reprise] %d essai(s) déjà dans %s", len(study.trials), os.path.abspath(run_dir))

    def after_each_trial(study_, trial_):   # résultats à jour même si le job est coupé
        export_results(study_, run_dir, cfg.to_dict(), decode_params)

    study.optimize(
        make_objective(data, cfg, run_dir, args.verbose),
        n_trials=cfg.n_trials,
        timeout=cfg.timeout_hours * 3600 if cfg.timeout_hours else None,
        callbacks=[after_each_trial],
        catch=(tf.errors.ResourceExhaustedError,),   # manque de mémoire GPU : essai FAIL, on continue
    )

    summary = export_results(study, run_dir, cfg.to_dict(), decode_params)
    print_top(study, cfg)
    if "best_params" in summary:
        best_cfg = best_alpha_sweep_config(cfg, summary["best_params"])
        with open(os.path.join(run_dir, "best_config.json"), "w") as f:
            json.dump(best_cfg, f, indent=2)
        log.info("Meilleur essai #%d : %s = %.4f avec %s", summary["best_trial"], cfg.metric,
                 summary["best_value"], json.dumps(summary["best_params"]))
        if "param_importances" in summary:
            log.info("Importance des hyperparamètres : %s", ", ".join(
                f"{k} {v:.2f}" for k, v in summary["param_importances"].items()))
        log.info("Pour l'évaluer sur le test avec plusieurs graines :\n"
                 "  python -m experiments.alpha_sweep --config %s --epochs <final>",
                 os.path.join(run_dir, "best_config.json"))
    log.info("Résultats : %s", run_dir)


if __name__ == "__main__":
    with tf.device(default_device()):
        main()
