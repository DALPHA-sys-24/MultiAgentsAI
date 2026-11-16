"""
Scripts d'expériences, à lancer depuis la racine du dépôt :

    python -m experiments.campaign <campagne> [--list|--task I|--merge]   # configs/campaigns/<campagne>.yaml
    python -m experiments.collaboration           [options]   # K agents, configs/collaboration.yaml
    python -m experiments.reference               [options]   # grand modèle seul, configs/reference.yaml
    python -m experiments.hyperparameter_search   [options]   # configs/hyperparameter_search.yaml
    python -m experiments.plot_results <campagne>             # figures PNG + PDF d'une campagne
    sbatch --array=0-<N-1> slurm/campaign.sh <campagne>       # cluster : job array + fusion

Ajouter une expérience : une sous-classe de Experiment (experiments/common.py) qui découpe le
travail en tâches indépendantes (tasks), exécute une tâche (run_task) et rassemble les
résultats (merge) ; une dataclass de config dans collaborative_learning/utils/config.py ; un YAML
dans configs/ ; son nom dans experiment_class (experiments/campaign.py) pour les campagnes.
"""
