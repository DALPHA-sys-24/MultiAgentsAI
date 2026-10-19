"""
ResNet (He et al. 2016) pour petites images (28x28, 32x32).

- resnet20/32/44/56/110 : version CIFAR, 3 étages de 16, 32, 64 filtres.
- resnet18/34 : version ImageNet avec un stem 3x3 de stride 1 et sans max-pool.
"""
from typing import List

import tensorflow as tf

from .layers import conv_bn
from .registry import register_model

L = tf.keras.layers


def basic_block(x, filters, stride):
    """Bloc résiduel basique : deux conv 3x3, projection 1x1 si la forme change."""
    shortcut = x
    out = conv_bn(x, filters, 3, stride)
    out = conv_bn(out, filters, 3, 1, activation=None)
    if stride != 1 or x.shape[-1] != filters:
        shortcut = conv_bn(x, filters, 1, stride, activation=None)
    return L.ReLU()(L.Add()([out, shortcut]))


def resnet(inputs, stem_filters: int, stage_filters: List[int], blocks_per_stage: List[int],
           width: float = 1.0):
    """
    ResNet générique : stem, puis un étage par entrée de stage_filters (stride 2 dès le 2e).
    width multiplie le nombre de filtres de chaque couche (ResNet « large »).
    """
    def scaled(filters):
        return max(1, int(round(filters * width)))

    x = conv_bn(inputs, scaled(stem_filters), 3, 1)
    for stage, (filters, n_blocks) in enumerate(zip(map(scaled, stage_filters), blocks_per_stage)):
        for block in range(n_blocks):
            stride = 2 if stage > 0 and block == 0 else 1
            x = basic_block(x, filters, stride)
    return x


def resnet_cifar(inputs, depth: int, width: float = 1.0):
    """ResNet CIFAR de profondeur 6n + 2."""
    n = (depth - 2) // 6
    return resnet(inputs, 16, [16, 32, 64], [n, n, n], width)


@register_model("resnet20")
def resnet20(inputs, width=1.0):
    return resnet_cifar(inputs, 20, width)


@register_model("resnet32")
def resnet32(inputs, width=1.0):
    return resnet_cifar(inputs, 32, width)


@register_model("resnet44")
def resnet44(inputs, width=1.0):
    return resnet_cifar(inputs, 44, width)


@register_model("resnet56")
def resnet56(inputs, width=1.0):
    return resnet_cifar(inputs, 56, width)


@register_model("resnet110")
def resnet110(inputs, width=1.0):
    return resnet_cifar(inputs, 110, width)


@register_model("resnet18")
def resnet18(inputs, width=1.0):
    return resnet(inputs, 64, [64, 128, 256, 512], [2, 2, 2, 2], width)


@register_model("resnet34")
def resnet34(inputs, width=1.0):
    return resnet(inputs, 64, [64, 128, 256, 512], [3, 4, 6, 3], width)
