"""
Scripts d'expériences, à lancer depuis la racine du dépôt :

    python -m experiments.collab_vs_big      [options]     # configs/collab_vs_big.json
    python -m experiments.alpha_sweep        [options]     # configs/alpha_sweep.json
    python -m experiments.best_hyperparams   [options]     # configs/best_hyperparams.json
    sbatch slurm/<expérience>.sh                           # cluster : job array + fusion

Ajouter une expérience : une sous-classe de Experiment (experiments/common.py) qui découpe le
travail en tâches indépendantes (tasks), exécute une tâche (run_task) et rassemble les
résultats (merge) ; une dataclass de config dans dml/utils/config.py ; un JSON dans configs/.
"""
