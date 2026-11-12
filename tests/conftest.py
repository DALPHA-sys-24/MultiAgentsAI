"""Configuration commune des tests : le dossier du code est importable, TensorFlow silencieux."""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # tests sur CPU : rapides et reproductibles
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def tiny_data():
    """Petit jeu de données aléatoire au format de data.load_dataset (8x8x3, 10 classes)."""
    rng = np.random.default_rng(0)

    def split(n):
        return rng.normal(size=(n, 8, 8, 3)).astype("float32"), rng.integers(0, 10, n).astype("int32")

    x_train, y_train = split(64)
    x_val, y_val = split(32)
    x_test, y_test = split(32)
    return {"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val,
            "x_test": x_test, "y_test": y_test, "num_classes": 10, "input_shape": (8, 8, 3)}


TINY = dict(archs="resnet20", widths=0.25)   # petit ResNet pour des tests rapides
