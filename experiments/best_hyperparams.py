"""
Recherche des meilleurs hyperparamètres d'un modèle sur un jeu de données (Optuna).

Chaque essai tire des hyperparamètres dans `search_space`, entraîne les K agents et
mesure la métrique de validation (par défaut l'accuracy du consensus) à chaque époque.
Les essais mauvais sont arrêtés tôt (pruning) ; le test n'est jamais utilisé.

Une tâche = un worker qui lance `n_trials` essais. Les `n_workers` workers partagent la même
étude (study.journal) : sur le cluster, ils tournent en parallèle dans un job array, et
chaque relance ajoute des essais à la recherche existante.

Exemples (depuis la racine du dépôt) :
    python -m experiments.best_hyperparams                                  # configs/best_hyperparams.json
    python -m experiments.best_hyperparams --dataset cifar10 --arch resnet32 --num-agents 3 --n-trials 50
    python -m experiments.best_hyperparams --cpu --train-subset 2000 --epochs 3 --n-trials 2 --n-workers 1
    sbatch slurm/best_hyperparams.sh     # cluster (réglages en tête du script)

Sorties : runs/best_hyperparams_<arch>_<dataset>_x<K>[_<tag>]/
    config.json        config de la recherche et environnement (commit Git, versions)
    study.journal      l'étude Optuna (partagée par les workers)
    trials.csv         un essai par ligne : valeur, état (COMPLETE / PRUNED / FAIL), paramètres
    best.json          meilleur essai, ses paramètres, importance de chaque hyperparamètre
    best_config.json   config prête pour experiments.alpha_sweep avec les meilleurs paramètres
    trials/            historique par époque (.csv) de chaque essai
    logs/              journaux (sorties SLURM dans Logs/ à la racine)
"""
import dataclasses
import json
import os

import tensorflow as tf

from dml.evaluation.reporting import write_json
from dml.training.runner import cleanup, count_trainable_params, run_multi_experiment
from dml.tuning.pruning import OptunaPruningCallback
from dml.tuning.space import decode_params, suggest
from dml.tuning.study import create_study, export_results
from dml.utils.config import AlphaSweepConfig, HyperparamSearchConfig
from experiments.common import Experiment, Task, log


def make_objective(data: dict, cfg: HyperparamSearchConfig, trials_dir: str, verbose: int):
    """Fonction objectif d'Optuna : un essai = un entraînement des K agents, sans le test."""
    import optuna

    def objective(trial):
        params = suggest(trial, cfg.search_space)
        tc = dataclasses.replace(cfg, **params).validate()
        log.info("[essai %d] %s", trial.number, json.dumps(params))
        pruning = OptunaPruningCallback(trial, cfg.metric, cfg.direction)
        try:
            run_multi_experiment(
                data, archs=tc.arch, num_agents=tc.num_agents, seed=0,
                loss_type=tc.loss_type, strategy=tc.strategy, alpha=tc.alpha, temperature=tc.temperature,
                per_agent_views=tc.per_agent_views,
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


def best_alpha_sweep_config(cfg: HyperparamSearchConfig, best_params: dict) -> dict:
    """Config experiments.alpha_sweep avec les meilleurs paramètres (alphas = [meilleur alpha])."""
    best = dataclasses.replace(cfg, **best_params)
    fields = {f.name for f in dataclasses.fields(AlphaSweepConfig)}
    out = {k: v for k, v in dataclasses.asdict(best).items() if k in fields}
    out.update(alphas=[best.alpha], n_runs=3, tag="best",
               _note="Meilleurs paramètres trouvés par experiments.best_hyperparams. "
                     "Penser à remettre le nombre d'époques final (epochs).")
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
        log.info("[taille] %d x %s : %s param. chacun | %s | critère : %s (%s)", cfg.num_agents, cfg.arch,
                 f"{params:,}", cfg.dataset, cfg.metric, cfg.direction)
        log.info("[espace] %s", json.dumps(cfg.search_space))
        return {"agent_params": params}

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
        best_path = os.path.join(run_dir, "best_config.json")
        write_json(best_path, best_alpha_sweep_config(cfg, summary["best_params"]))
        log.info("Meilleur essai #%d : %s = %.4f avec %s", summary["best_trial"], cfg.metric,
                 summary["best_value"], json.dumps(summary["best_params"]))
        if "param_importances" in summary:
            log.info("Importance des hyperparamètres : %s", ", ".join(
                f"{k} {v:.2f}" for k, v in summary["param_importances"].items()))
        log.info("Pour l'évaluer sur le test avec plusieurs graines :\n"
                 "  python -m experiments.alpha_sweep --config %s --epochs <final>", best_path)


if __name__ == "__main__":
    BestHyperparams().main()
