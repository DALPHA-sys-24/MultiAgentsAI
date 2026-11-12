"""
Sauvegarde et reprise de l'entraînement d'un MultiAgentModel.

    ckpt = TrainingCheckpoint("results/run/checkpoints/a10_s2")
    initial_epoch = ckpt.restore(model)          # 0 si rien à reprendre
    model.fit(..., initial_epoch=initial_epoch, callbacks=[ckpt])
    ckpt.load_best(model)                        # poids de la meilleure époque (validation)

Le dossier contient :
    state.npz / state.json   dernier état complet (poids, BatchNorm, optimiseurs, époque,
                             historique), réécrit à chaque époque : un job SLURM coupé
                             reprend à l'époque suivante ;
    best_weights.npz         poids des agents à la meilleure valeur de `monitor` ;
    final_weights.npz        poids à la fin de l'entraînement (save_final).

Le format est un simple .npz (listes de tableaux numpy), lisible sous Keras 2 et Keras 3.
À la reprise, l'ordre de mélange des données recommence au début : la suite de
l'entraînement n'est pas identique au bit près à un run sans interruption.
"""
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)


# =============================================================================
# POIDS ET ÉTAT DES OPTIMISEURS
# =============================================================================

def _optimizer_variables(opt) -> list:
    variables = opt.variables
    return list(variables() if callable(variables) else variables)


def _build_optimizers(model, arrays):
    """Crée les variables des optimiseurs (moments...) si elles n'existent pas encore."""
    for k, (agent, opt) in enumerate(zip(model.agents, model.optimizers)):
        n_saved = sum(1 for key in arrays.files if key.startswith(f"opt{k}_"))
        if len(_optimizer_variables(opt)) < n_saved:
            opt.build(agent.trainable_variables)
        n_now = len(_optimizer_variables(opt))
        if n_now != n_saved:
            raise ValueError(f"Optimiseur de l'agent {k + 1} : {n_now} variables, {n_saved} dans la "
                             "sauvegarde. La config a changé depuis le début du run ?")


def _atomic_savez(path: str, arrays: Dict[str, np.ndarray]):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)  # jamais de fichier à moitié écrit si le job est tué pendant l'écriture


def agents_weights(model) -> Dict[str, np.ndarray]:
    return {f"agent{k}_{i}": w for k, agent in enumerate(model.agents) for i, w in enumerate(agent.get_weights())}


def set_agents_weights(model, arrays) -> None:
    for k, agent in enumerate(model.agents):
        n = len(agent.get_weights())
        agent.set_weights([arrays[f"agent{k}_{i}"] for i in range(n)])


def save_weights(model, path: str) -> None:
    """Poids (entraînables et BatchNorm) de tous les agents dans un .npz."""
    _atomic_savez(path, agents_weights(model))


def load_weights(model, path: str) -> None:
    with np.load(path) as arrays:
        set_agents_weights(model, arrays)


# =============================================================================
# CALLBACK
# =============================================================================

class TrainingCheckpoint(tf.keras.callbacks.Callback):
    """
    Sauvegarde l'état complet à chaque fin d'époque et garde les poids de la meilleure époque.

    Args:
        directory: dossier des fichiers de ce run.
        monitor: métrique de sélection du meilleur modèle (ex. "val_accuracy"). Si elle est
            absente des logs (pas de validation), la meilleure époque n'est pas suivie.
        mode: "max" ou "min".
        every: sauvegarde de l'état complet toutes les `every` époques.
    """

    def __init__(self, directory: str, monitor: str = "val_accuracy", mode: str = "max", every: int = 1):
        super().__init__()
        self.directory = directory
        self.monitor = monitor
        self.mode = mode
        self.every = max(1, int(every))
        self.history: Dict[str, List[float]] = {}
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        os.makedirs(directory, exist_ok=True)

    # --- chemins -------------------------------------------------------------
    def _path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    @property
    def best_path(self) -> str:
        return self._path("best_weights.npz")

    @property
    def final_path(self) -> str:
        return self._path("final_weights.npz")

    # --- reprise -------------------------------------------------------------
    def restore(self, model) -> int:
        """Recharge le dernier état s'il existe. Renvoie l'époque à laquelle reprendre (0 sinon)."""
        state_json, state_npz = self._path("state.json"), self._path("state.npz")
        if not (os.path.exists(state_json) and os.path.exists(state_npz)):
            return 0
        with open(state_json) as f:
            meta = json.load(f)
        with np.load(state_npz) as arrays:
            _build_optimizers(model, arrays)
            set_agents_weights(model, arrays)
            for k, opt in enumerate(model.optimizers):
                for i, var in enumerate(_optimizer_variables(opt)):
                    var.assign(arrays[f"opt{k}_{i}"])
        self.history = meta.get("history", {})
        self.best, self.best_epoch = meta.get("best"), meta.get("best_epoch")
        log.info("[reprise] %s : reprise à l'époque %d", self.directory, meta["epoch"])
        return int(meta["epoch"])

    # --- sauvegardes ---------------------------------------------------------
    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        return value > self.best if self.mode == "max" else value < self.best

    def _save_state(self, epoch_done: int):
        arrays = agents_weights(self.model)
        for k, opt in enumerate(self.model.optimizers):
            for i, var in enumerate(_optimizer_variables(opt)):
                arrays[f"opt{k}_{i}"] = np.asarray(var.numpy())
        _atomic_savez(self._path("state.npz"), arrays)
        tmp = self._path("state.json.tmp")
        with open(tmp, "w") as f:
            json.dump({"epoch": epoch_done, "history": self.history,
                       "best": self.best, "best_epoch": self.best_epoch, "monitor": self.monitor}, f)
        os.replace(tmp, self._path("state.json"))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))
        value = logs.get(self.monitor)
        if value is not None and self._improved(float(value)):
            self.best, self.best_epoch = float(value), epoch + 1
            save_weights(self.model, self.best_path)
        if (epoch + 1) % self.every == 0 or epoch + 1 == self.params.get("epochs"):
            self._save_state(epoch + 1)

    def save_final(self, model, keep_state: bool = False):
        """Poids finaux ; supprime l'état de reprise (gros fichier) sauf si keep_state."""
        save_weights(model, self.final_path)
        if not keep_state:
            for name in ("state.npz", "state.json"):
                if os.path.exists(self._path(name)):
                    os.remove(self._path(name))

    def load_best(self, model) -> bool:
        """Charge les poids de la meilleure époque. False s'il n'y en a pas (pas de validation)."""
        if not os.path.exists(self.best_path):
            return False
        load_weights(model, self.best_path)
        return True
