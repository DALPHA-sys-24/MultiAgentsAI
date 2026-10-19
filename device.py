"""
Choix du matériel : GPU si disponible, sinon CPU.

À appeler une seule fois, au début du script, avant de créer un modèle :
    from device import configure_device
    configure_device()             # GPU si visible par TensorFlow, sinon CPU
    configure_device(force_cpu=True)
"""
import os
from typing import Optional

import tensorflow as tf


def configure_device(force_cpu: bool = False, precision: Optional[str] = None, verbose: bool = True) -> str:
    """
    Configure TensorFlow pour utiliser le GPU s'il est disponible, sinon le CPU.

    Args:
        force_cpu: Ignore les GPU même s'ils sont présents.
        precision: None / "float32" (défaut), ou "mixed_bfloat16" pour accélérer
            sur GPU récent (RTX 30xx/40xx, A100, H100). Ignoré sur CPU.
        verbose: Affiche le matériel utilisé.

    Returns:
        "gpu" ou "cpu".
    """
    gpus = [] if force_cpu else tf.config.list_physical_devices("GPU")
    if force_cpu:
        tf.config.set_visible_devices([], "GPU")

    if gpus:
        # Allouer la mémoire au besoin plutôt que tout le GPU d'un coup
        # (le GPU d'un portable sert aussi à l'affichage).
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # GPU déjà initialisé : réglage impossible, sans conséquence
        device = "gpu"
        if precision and precision != "float32":
            tf.keras.mixed_precision.set_global_policy(precision)
    else:
        device = "cpu"

    if verbose:
        if device == "gpu":
            names = [tf.config.experimental.get_device_details(g).get("device_name", g.name) for g in gpus]
            print(f"[device] GPU : {', '.join(names)} | précision : {tf.keras.mixed_precision.global_policy().name}")
        else:
            reason = "forcé" if force_cpu else "aucun GPU visible par TensorFlow"
            print(f"[device] CPU ({reason}, {os.cpu_count()} cœurs)")
            if not force_cpu and os.name == "nt":
                print("[device] Sous Windows, TensorFlow >= 2.11 n'utilise pas le GPU nativement : passer par WSL2.")
    return device
