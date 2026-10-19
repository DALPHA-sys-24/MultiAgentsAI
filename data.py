"""
Chargement des données et pipelines tf.data avec augmentation.

Usage:
    data = load_dataset("cifar10", val_size=5000)
    train_ds = make_train_dataset(data["x_train"], data["y_train"], batch_size=128)
    test_ds  = make_eval_dataset(data["x_test"], data["y_test"], batch_size=256)
    model.fit(train_ds, validation_data=make_eval_dataset(data["x_val"], data["y_val"]), ...)
"""
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

_LOADERS = {
    "fashion_mnist": tf.keras.datasets.fashion_mnist,
    "cifar10": tf.keras.datasets.cifar10,
    "cifar100": tf.keras.datasets.cifar100,
}


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
    (x_train, y_train), (x_test, y_test) = _LOADERS[name].load_data()

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


def _augment(image: tf.Tensor, pad: int, flip: bool) -> tf.Tensor:
    """Random crop avec padding nul + flip horizontal (standard CIFAR)."""
    h, w, c = image.shape
    if pad > 0:
        image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]])
        image = tf.image.random_crop(image, [h, w, c])
    if flip:
        image = tf.image.random_flip_left_right(image)
    return image


def make_train_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    augment: bool = True,
    pad: int = 4,
    flip: bool = True,
    per_agent_views: Optional[int] = None,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Pipeline d'entraînement mélangé et augmenté.

    Args:
        per_agent_views: None = une seule image par exemple, partagée par tous les
            agents (DML classique, sortie (B, H, W, C)). Un entier K = K augmentations
            indépendantes du même exemple, une par agent (sortie (B, K, H, W, C)).
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)

    if augment:
        if per_agent_views:
            k = per_agent_views
            ds = ds.map(lambda img, lbl: (tf.stack([_augment(img, pad, flip) for _ in range(k)]), lbl),
                        num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda img, lbl: (_augment(img, pad, flip), lbl),
                        num_parallel_calls=tf.data.AUTOTUNE)
    elif per_agent_views:
        k = per_agent_views
        ds = ds.map(lambda img, lbl: (tf.stack([img] * k), lbl), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def make_eval_dataset(x: np.ndarray, y: np.ndarray, batch_size: int = 256) -> tf.data.Dataset:
    """Pipeline d'évaluation (sans augmentation ; l'image est partagée par tous les agents)."""
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
