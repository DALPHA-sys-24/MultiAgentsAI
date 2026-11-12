"""
Recherche d'hyperparamètres avec Optuna.

    space.py      espace de recherche décrit en JSON -> trial.suggest_*
    pruning.py    callback Keras qui arrête tôt les essais mauvais (pruning)
    study.py      création de l'étude (sampler, pruner, stockage partagé entre jobs) et résumés

Script : experiments/best_hyperparams.py, config : configs/best_hyperparams.json.
Optuna n'est importé que par ce package : le reste du code tourne sans lui.
"""
