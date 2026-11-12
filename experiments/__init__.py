"""
Scripts d'expériences, à lancer depuis la racine du dépôt :

    python -m experiments.collab_vs_big --config configs/collab_vs_big.json
    python -m experiments.alpha_sweep   --config configs/alpha_sweep.json
    python -m experiments.merge_sweep   results/alpha_sweep

Ajouter une expérience : un module ici (lecture de la config, boucle sur les graines,
écriture des résultats), une dataclass de config dans dml/utils/config.py et un JSON
dans configs/.
"""
