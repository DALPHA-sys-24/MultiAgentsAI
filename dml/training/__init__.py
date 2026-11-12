"""
Entraînement.

    optimizers.py  optimiseurs par nom (@register_optimizer), schedules, weight decay
    checkpoint.py  sauvegarde à chaque époque, reprise, meilleurs poids
    runner.py      run_multi_experiment, run_single_reference, match_width, benchmark

Ce fichier n'importe rien : core/multiagent.py importe training/optimizers.py, et
runner.py importe core/multiagent.py (pas d'import circulaire).
"""
