"""
Chargement des jeux de données, enregistrés par leur nom.

    data = load_dataset("cifar10", val_size=5000)

Ajouter un jeu de données :

    @register_dataset("mon_jeu")
    def mon_jeu():
        return (x_train, y_train), (x_test, y_test)   # images (N, H, W[, C]) uint8, labels entiers
"""
from typing import Callable, Dict, List

import numpy as np
import tensorflow as tf

_LOADERS: Dict[str, Callable] = {}


def register_dataset(name: str):
    """Décorateur : enregistre une fonction qui renvoie (x_train, y_train), (x_test, y_test)."""
    def decorator(fn):
        if name in _LOADERS:
            raise ValueError(f"Jeu de données déjà enregistré : {name}")
        _LOADERS[name] = fn
        return fn
    return decorator


def available_datasets() -> List[str]:
    return list(_LOADERS)


register_dataset("fashion_mnist")(tf.keras.datasets.fashion_mnist.load_data)
register_dataset("cifar10")(tf.keras.datasets.cifar10.load_data)
register_dataset("cifar100")(tf.keras.datasets.cifar100.load_data)


def load_dataset(name: str, val_size: int = 5000, seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Charge un jeu de données, normalise par canal (statistiques du train) et
    sépare un ensemble de validation tiré du train.

    Returns:
        dict avec x_train, y_train, x_val, y_val, x_test, y_test, num_classes, input_shape.
        Images en float32 (N, H, W, C), labels en int32 (N,).
        Fashion-MNIST reste en 28x28x1 : les modèles acceptent n'importe quel nombre de canaux.
    """
    if name not in _LOADERS:
        raise ValueError(f"Jeu de données inconnu : {name}. Choix : {list(_LOADERS)}")
    (x_train, y_train), (x_test, y_test) = _LOADERS[name]()

    if x_train.ndim == 3:
        x_train, x_test = x_train[..., None], x_test[..., None]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.reshape(-1).astype("int32")
    y_test = y_test.reshape(-1).astype("int32")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_train))
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std = x_train.std(axis=(0, 1, 2), keepdims=True)
    x_train, x_val, x_test = [(x - mean) / std for x in (x_train, x_val, x_test)]

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "num_classes": int(y_test.max() + 1),
        "input_shape": x_train.shape[1:],
    }
