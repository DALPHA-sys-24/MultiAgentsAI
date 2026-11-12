"""Création de l'étude Optuna et export des résultats."""
import logging
import os
from typing import Any, Dict

log = logging.getLogger(__name__)

SAMPLERS = ("tpe", "random", "cmaes")
PRUNERS = ("median", "hyperband", "none")


def make_sampler(name: str, seed: int):
    import optuna
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed, multivariate=True)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    raise ValueError(f"Sampler inconnu : {name}. Choix : {SAMPLERS}")


def make_pruner(name: str, warmup_epochs: int, max_epochs: int):
    import optuna
    if name == "median":
        # un essai est arrêté s'il est sous la médiane des essais précédents à la même époque
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=warmup_epochs)
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=max(1, warmup_epochs), max_resource=max_epochs)
    if name == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(f"Pruner inconnu : {name}. Choix : {PRUNERS}")


def make_storage(run_dir: str):
    """
    Stockage dans un fichier journal (run_dir/study.journal) : plusieurs jobs SLURM peuvent
    travailler en même temps sur la même étude, même sur un disque réseau (contrairement à SQLite).
    """
    from optuna.storages import JournalStorage
    path = os.path.join(run_dir, "study.journal")
    try:
        from optuna.storages.journal import JournalFileBackend          # Optuna >= 4.0
        return JournalStorage(JournalFileBackend(path))
    except ImportError:
        from optuna.storages import JournalFileStorage                  # Optuna 3.x
        return JournalStorage(JournalFileStorage(path))


def create_study(run_dir: str, name: str, direction: str, sampler: str, pruner: str,
                 seed: int, pruner_warmup: int, epochs: int):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=name, storage=make_storage(run_dir), load_if_exists=True, direction=direction,
        sampler=make_sampler(sampler, seed), pruner=make_pruner(pruner, pruner_warmup, epochs),
    )


def export_results(study, run_dir: str, base_config: Dict[str, Any], decode) -> Dict[str, Any]:
    """Écrit trials.csv (tous les essais) et best.json (meilleur essai, config complète, importances)."""
    import optuna
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    tmp = os.path.join(run_dir, f"trials.csv.{os.getpid()}.tmp")   # plusieurs workers écrivent en même temps
    df.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(run_dir, "trials.csv"))

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    states = {s.name: sum(t.state == s for t in study.trials) for s in optuna.trial.TrialState}
    summary: Dict[str, Any] = {"n_trials": len(study.trials), "states": {k: v for k, v in states.items() if v}}
    if not complete:
        log.warning("Aucun essai terminé pour l'instant.")
        return summary

    best = study.best_trial
    best_params = decode(best.params)
    summary.update({
        "best_value": best.value,
        "best_trial": best.number,
        "best_params": best_params,
        "best_epoch": best.user_attrs.get("best_epoch"),
        "best_config": {**base_config, **best_params},
    })
    if len(complete) >= 4:
        try:
            summary["param_importances"] = {k: float(v) for k, v in
                                            optuna.importance.get_param_importances(study).items()}
        except Exception as exc:  # importance impossible (un seul paramètre varié, valeurs identiques...)
            log.info("Importances non calculées : %s", exc)
    from dml.evaluation.reporting import write_json
    write_json(os.path.join(run_dir, "best.json"), summary)
    return summary
