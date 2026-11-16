"""
Recherche des meilleurs hyperparamètres d'UN modèle (une architecture) sur un jeu de données (Optuna).

Chaque essai tire des hyperparamètres dans `search_space` (lr, weight decay, optimiseur...),
entraîne un seul modèle `arch` et mesure son accuracy de validation à chaque époque. Les essais
mauvais sont arrêtés tôt (pruning) ; le test n'est jamais utilisé. Pas d'agents ni d'alpha ici.

Avec les meilleurs paramètres, le script écrit tuned_params.yaml (optimiseur, lr, weight decay,
batch...), repris automatiquement par les campagnes qui ont « use_tuned_params: true »
(agents et grand modèle), ou à la main :
    python -m experiments.collaboration --config results/hyperparameter_search/<dataset>/<arch>/tuned_params.yaml

Une tâche = un worker qui lance `n_trials` essais. Les `n_workers` workers partagent la même
étude (study.journal) : sur le cluster, ils tournent en parallèle dans un job array, et
chaque relance ajoute des essais à la recherche existante.

Exemples (depuis la racine du dépôt) :
    python -m experiments.hyperparameter_search                        # configs/hyperparameter_search.yaml
    python -m experiments.hyperparameter_search --dataset cifar10 --arch resnet32 --n-trials 50
    python -m experiments.hyperparameter_search --cpu --train-subset 2000 --epochs 3 --n-trials 2 --n-workers 1
    python -m experiments.campaign hyperparameter_search --list          # cluster : voir configs/campaigns/

Sorties : results/hyperparameter_search/<dataset>/<arch>[_<tag>]/
    config.json              config de la recherche et environnement (commit Git, versions)
    study.journal            l'étude Optuna (partagée par les workers)
    trials.csv               un essai par ligne : valeur, état (COMPLETE / PRUNED / FAIL), paramètres
    best.json                meilleur essai, ses paramètres, importance de chaque hyperparamètre
    tuned_params.yaml        meilleurs paramètres d'entraînement (repris par les campagnes)
    trials/                  historique par époque (.csv) de chaque essai
    logs/                    journaux (sorties SLURM dans slurm_logs/ à la racine)
"""
import dataclasses
import json
import os

import tensorflow as tf

from collaborative_learning.training.runner import cleanup, count_trainable_params, run_single_reference
from collaborative_learning.tuning.pruning import OptunaPruningCallback
from collaborative_learning.tuning.space import decode_params, suggest
from collaborative_learning.tuning.study import create_study, export_results
from collaborative_learning.utils.config import CollaborationConfig, HyperparamSearchConfig
from collaborative_learning.utils.config_files import write_yaml
from experiments.common import Experiment, Task, log

# Réglages d'entraînement repris par les K agents et le grand modèle (les époques finales restent
# celles de la campagne : les essais sont courts)
TRANSFERRED = ("batch_size", "optimizer", "lr", "weight_decay", "wd_exclude_norm_bias",
               "optimizer_kwargs", "warmup_epochs", "augment", "val_size", "precision")
TUNED_FILE = "tuned_params.yaml"
LEGACY_TUNED_FILE = "tuned_collaboration.json"     # avant le 2026-11-15 (config K agents complète)


def make_objective(data: dict, cfg: HyperparamSearchConfig, trials_dir: str, verbose: int):
    """Fonction objectif d'Optuna : un essai = un entraînement d'un seul modèle, sans le test."""
    import optuna

    def objective(trial):
        params = suggest(trial, cfg.search_space)
        tc = dataclasses.replace(cfg, **params).validate()
        log.info("[essai %d] %s", trial.number, json.dumps(params))
        pruning = OptunaPruningCallback(trial, cfg.metric, cfg.direction)
        try:
            run_single_reference(
                data, arch=tc.arch, seed=0, single_agent_loss="ce" if tc.loss == "ce" else "form",
                optimizer=tc.optimizer, lr=tc.lr, optimizer_kwargs=tc.optimizer_kwargs,
                callbacks=[pruning], evaluate_test=False,
                csv_log=os.path.join(trials_dir, f"trial_{trial.number:04d}.csv"),
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


def tuned_params(cfg: HyperparamSearchConfig, best_params: dict) -> dict:
    """Réglages d'entraînement du meilleur essai : ceux de TRANSFERRED et ceux qui ont été cherchés."""
    best = dataclasses.replace(cfg, **best_params)
    fields = {f.name for f in dataclasses.fields(CollaborationConfig)}
    keys = [k for k in TRANSFERRED] + [k for k in best_params if k in fields and k not in TRANSFERRED]
    return {k: getattr(best, k) for k in keys}


class HyperparameterSearch(Experiment):
    NAME = "hyperparameter_search"
    CONFIG_CLS = HyperparamSearchConfig
    DESCRIPTION = __doc__
    WRITES_TASK_DIR = False

    def tasks(self, cfg):
        return [Task(f"worker{i}", {"worker": i}) for i in range(cfg.n_workers)]

    def is_done(self, task_dir):
        return False   # un worker relancé ajoute des essais à l'étude

    def prepare(self, cfg, data):
        params = count_trainable_params(cfg.arch, data["input_shape"], data["num_classes"])
        log.info("[taille] 1 x %s : %s param. | %s | loss %s | critère : %s (%s)", cfg.arch,
                 f"{params:,}", cfg.dataset, cfg.loss, cfg.metric, cfg.direction)
        log.info("[espace] %s", json.dumps(cfg.search_space))
        return {"model_params": params}

    def study(self, cfg, run_dir, seed):
        return create_study(run_dir, self.NAME, cfg.direction, cfg.sampler, cfg.pruner, seed,
                            cfg.pruner_warmup_epochs, cfg.epochs)

    def run_task(self, task, cfg, data, ctx, task_dir, verbose):
        run_dir = self.run_dir
        trials_dir = os.path.join(run_dir, "trials")
        os.makedirs(trials_dir, exist_ok=True)
        # chaque worker a sa propre graine de sampler, sinon ils proposeraient les mêmes essais
        study = self.study(cfg, run_dir, cfg.study_seed + task.params["worker"])
        if study.trials:
            log.info("[reprise] %d essai(s) déjà dans l'étude", len(study.trials))
        study.optimize(
            make_objective(data, cfg, trials_dir, verbose),
            n_trials=cfg.n_trials,
            timeout=cfg.timeout_hours * 3600 if cfg.timeout_hours else None,
            callbacks=[lambda s, t: self._merge(cfg, run_dir)],   # résultats à jour après chaque essai
            catch=(tf.errors.ResourceExhaustedError,),            # manque de mémoire GPU : essai FAIL, on continue
        )
        return None    # rien dans tasks/ : tout est dans l'étude

    def merge(self, cfg, run_dir, tasks):
        import optuna
        if not os.path.exists(os.path.join(run_dir, "study.journal")):
            log.info("Pas encore d'étude dans %s.", run_dir)
            return
        study = self.study(cfg, run_dir, cfg.study_seed)
        summary = export_results(study, run_dir, cfg.to_dict(), decode_params)
        done = sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
                      key=lambda t: t.value, reverse=cfg.direction == "maximize")
        lines = [f"Meilleurs essais ({cfg.metric}, {len(done)} terminés / {len(study.trials)} lancés) :"]
        for t in done[:10]:
            lines.append(f"  #{t.number:<4d} {t.value:.4f}  " + "  ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in t.params.items()))
        log.info("\n".join(lines))
        if "best_params" not in summary:
            return
        path = os.path.join(run_dir, TUNED_FILE)
        write_yaml(path, tuned_params(cfg, summary["best_params"]),
                   header=f"Meilleurs paramètres d'un {cfg.arch} seul sur {cfg.dataset} (essai #{summary['best_trial']}, "
                          f"{cfg.metric} = {summary['best_value']:.4f}).\n"
                          "Repris par les campagnes avec use_tuned_params: true (agents et grand modèle).")
        log.info("Meilleur essai #%d : %s = %.4f avec %s", summary["best_trial"], cfg.metric,
                 summary["best_value"], json.dumps(summary["best_params"]))
        if "param_importances" in summary:
            log.info("Importance des hyperparamètres : %s", ", ".join(
                f"{k} {v:.2f}" for k, v in summary["param_importances"].items()))
        log.info("Paramètres écrits dans %s (repris par les campagnes avec use_tuned_params: true)", path)


if __name__ == "__main__":
    HyperparameterSearch().main()
