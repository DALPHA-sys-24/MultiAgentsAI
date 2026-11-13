"""
Recherche des meilleurs hyperparamètres d'UN modèle (une architecture) sur un jeu de données (Optuna).

Chaque essai tire des hyperparamètres dans `search_space` (lr, weight decay, optimiseur...),
entraîne un seul modèle `arch` et mesure son accuracy de validation à chaque époque. Les essais
mauvais sont arrêtés tôt (pruning) ; le test n'est jamais utilisé. Pas d'agents ni d'alpha ici.

Avec les meilleurs paramètres, le script écrit des configs prêtes pour construire ensuite
K agents de cette architecture (best_collab_vs_big.json, best_alpha_sweep.json) :
    python -m experiments.collab_vs_big --config runs/best_hyperparams_<arch>_<dataset>/best_collab_vs_big.json --num-agents 3

Une tâche = un worker qui lance `n_trials` essais. Les `n_workers` workers partagent la même
étude (study.journal) : sur le cluster, ils tournent en parallèle dans un job array, et
chaque relance ajoute des essais à la recherche existante.

Exemples (depuis la racine du dépôt) :
    python -m experiments.best_hyperparams                                  # configs/best_hyperparams.json
    python -m experiments.best_hyperparams --dataset cifar10 --arch resnet32 --n-trials 50
    python -m experiments.best_hyperparams --cpu --train-subset 2000 --epochs 3 --n-trials 2 --n-workers 1
    sbatch slurm/best_hyperparams.sh     # cluster (réglages en tête du script)

Sorties : runs/best_hyperparams_<arch>_<dataset>[_<tag>]/
    config.json              config de la recherche et environnement (commit Git, versions)
    study.journal            l'étude Optuna (partagée par les workers)
    trials.csv               un essai par ligne : valeur, état (COMPLETE / PRUNED / FAIL), paramètres
    best.json                meilleur essai, ses paramètres, importance de chaque hyperparamètre
    best_collab_vs_big.json  config collab_vs_big avec ces paramètres (K agents : --num-agents)
    best_alpha_sweep.json    config alpha_sweep avec ces paramètres
    trials/                  historique par époque (.csv) de chaque essai
    logs/                    journaux (sorties SLURM dans Logs/ à la racine)
"""
import dataclasses
import json
import os

import tensorflow as tf

from dml.evaluation.reporting import write_json
from dml.training.runner import cleanup, count_trainable_params, run_single_reference
from dml.tuning.pruning import OptunaPruningCallback
from dml.tuning.space import decode_params, suggest
from dml.tuning.study import create_study, export_results
from dml.utils.config import AlphaSweepConfig, CollabVsBigConfig, HyperparamSearchConfig, load_config
from experiments.common import ROOT, Experiment, Task, log

# Réglages du modèle repris dans les configs K agents (les époques finales restent celles du fichier
# de l'expérience : les essais sont courts)
TRANSFERRED = ("dataset", "batch_size", "optimizer", "lr", "weight_decay", "wd_exclude_norm_bias",
               "optimizer_kwargs", "warmup_epochs", "augment", "val_size", "precision")


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


def multi_agent_config(cls, cfg: HyperparamSearchConfig, best_params: dict) -> dict:
    """
    Config d'une expérience à K agents (cls) : celle de configs/<expérience>.json, avec
    l'architecture, le jeu de données et les meilleurs paramètres du modèle seul.
    """
    best = dataclasses.replace(cfg, **best_params)
    path = os.path.join(ROOT, "configs", f"{cls.EXPERIMENT}.json")
    base = load_config(cls, path if os.path.exists(path) else None)
    arch_key = "small_arch" if cls is CollabVsBigConfig else "arch"
    keys = set(TRANSFERRED) | (set(best_params) & {f.name for f in dataclasses.fields(cls)})
    values = {k: getattr(best, k) for k in keys}
    out = {"_note": f"Paramètres du meilleur {cfg.arch} seul sur {cfg.dataset} (experiments.best_hyperparams). "
                    "Choisir K avec --num-agents ; vérifier epochs.",
           **dataclasses.asdict(dataclasses.replace(base, **values, **{arch_key: cfg.arch}, tag="best"))}
    return out


class BestHyperparams(Experiment):
    NAME = "best_hyperparams"
    CONFIG_CLS = HyperparamSearchConfig
    DESCRIPTION = __doc__
    USES_PARTS = False

    def tasks(self, cfg):
        return [Task(f"w{i}", {"worker": i}) for i in range(cfg.n_workers)]

    def is_done(self, part_dir):
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

    def run_task(self, task, cfg, data, ctx, part_dir, verbose):
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
        return None    # rien dans parts/ : tout est dans l'étude

    def merge(self, cfg, run_dir, parts):
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
        paths = {}
        for exp, cls in (("collab_vs_big", CollabVsBigConfig), ("alpha_sweep", AlphaSweepConfig)):
            paths[exp] = os.path.join(run_dir, f"best_{exp}.json")
            write_json(paths[exp], multi_agent_config(cls, cfg, summary["best_params"]))
        log.info("Meilleur essai #%d : %s = %.4f avec %s", summary["best_trial"], cfg.metric,
                 summary["best_value"], json.dumps(summary["best_params"]))
        if "param_importances" in summary:
            log.info("Importance des hyperparamètres : %s", ", ".join(
                f"{k} {v:.2f}" for k, v in summary["param_importances"].items()))
        log.info("Construire K agents avec ces paramètres :\n  python -m experiments.collab_vs_big --config %s "
                 "--num-agents <K>\n  python -m experiments.alpha_sweep --config %s --num-agents <K>",
                 paths["collab_vs_big"], paths["alpha_sweep"])


if __name__ == "__main__":
    BestHyperparams().main()
