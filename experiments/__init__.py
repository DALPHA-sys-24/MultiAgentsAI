"""
Scripts d'expériences, à lancer depuis la racine du dépôt :

    python -m experiments.collaboration           [options]   # configs/collaboration.json
    python -m experiments.hyperparameter_search   [options]   # configs/hyperparameter_search.json
    python -m experiments.plot_results            [options]   # figures PNG + PDF de results/
    sbatch slurm/<expérience>.sh                           # cluster : job array + fusion

Ajouter une expérience : une sous-classe de Experiment (experiments/common.py) qui découpe le
travail en tâches indépendantes (tasks), exécute une tâche (run_task) et rassemble les
résultats (merge) ; une dataclass de config dans collaborative_learning/utils/config.py ; un JSON dans configs/.
"""
