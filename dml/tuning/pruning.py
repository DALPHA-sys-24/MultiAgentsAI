"""Callback Keras qui transmet la métrique de validation à Optuna et arrête les essais mauvais."""
from typing import Callable, Dict, Optional

import numpy as np
import tensorflow as tf

# Métrique suivie -> fonction des logs Keras de fin d'époque
METRICS: Dict[str, Callable[[dict], Optional[float]]] = {
    # accuracy sur la validation (consensus si K > 1, le modèle lui-même si K = 1)
    "val_accuracy": lambda logs: logs.get("val_accuracy"),
    # loss de validation (à minimiser)
    "val_loss": lambda logs: logs.get("val_loss"),
}


class OptunaPruningCallback(tf.keras.callbacks.Callback):
    """
    À chaque fin d'époque : envoie la métrique à Optuna (trial.report), garde la meilleure
    valeur, et arrête l'entraînement si le pruner juge l'essai moins bon que les autres.
    """

    def __init__(self, trial, metric: str = "val_accuracy", direction: str = "maximize"):
        super().__init__()
        if metric not in METRICS:
            raise ValueError(f"Métrique inconnue : {metric}. Choix : {list(METRICS)}")
        self.trial = trial
        self.metric = metric
        self.direction = direction
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.pruned_at: Optional[int] = None

    def _better(self, value: float) -> bool:
        if self.best is None:
            return True
        return value > self.best if self.direction == "maximize" else value < self.best

    def on_epoch_end(self, epoch, logs=None):
        value = METRICS[self.metric](logs or {})
        if value is None:
            raise KeyError(f"'{self.metric}' absent des logs : il faut un ensemble de validation (val_size > 0).")
        value = float(value)
        if not np.isfinite(value):
            self.model.stop_training = True
            self.pruned_at = epoch + 1
            return
        if self._better(value):
            self.best, self.best_epoch = value, epoch + 1
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            self.pruned_at = epoch + 1
            self.model.stop_training = True   # l'essai est marqué « pruned » après fit()
