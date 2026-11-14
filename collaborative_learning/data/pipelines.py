"""
Pipelines tf.data : mélange, augmentation (random crop + flip) et batchs.

    train_ds = make_train_dataset(data["x_train"], data["y_train"], batch_size=128)
    test_ds  = make_eval_dataset(data["x_test"], data["y_test"], batch_size=256)
"""
from typing import Optional

import numpy as np
import tensorflow as tf


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
