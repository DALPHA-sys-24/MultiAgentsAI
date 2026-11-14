"""Blocs de couches partagés par les architectures."""
import tensorflow as tf

L = tf.keras.layers


def conv_bn(x, filters, kernel=3, stride=1, activation="relu"):
    """Conv2D sans biais + BatchNorm + activation optionnelle ("relu", "relu6" ou None)."""
    x = L.Conv2D(filters, kernel, strides=stride, padding="same",
                 use_bias=False, kernel_initializer="he_normal")(x)
    x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    if activation == "relu":
        x = L.ReLU()(x)
    elif activation == "relu6":
        x = L.ReLU(max_value=6.0)(x)
    return x


def depthwise_bn(x, stride=1, activation="relu"):
    """DepthwiseConv2D 3x3 + BatchNorm + activation ("relu" ou "relu6")."""
    x = L.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    return L.ReLU(max_value=6.0 if activation == "relu6" else None)(x)
