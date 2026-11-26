"""Poids de confiance w_k à chaque époque, au total et par classe (ConfidenceWeightsCallback)."""
import numpy as np
import pytest
import tensorflow as tf

from collaborative_learning.core.callbacks import ConfidenceWeightsCallback
from collaborative_learning.core.multiagent import MultiAgentModel
from collaborative_learning.training.runner import run_multi_experiment
from conftest import TINY

COMMON = dict(num_agents=3, batch_size=16, optimizer="sgd", lr=0.05, warmup_epochs=0,
              augment=False, seed=0, jit_compile=False, evaluate_test=False, **TINY)


def test_weights_are_exact_means_over_all_images_and_classes(tiny_data):
    x, y = tiny_data["x_val"], tiny_data["y_val"]
    model = MultiAgentModel(3, num_classes=10, **TINY)
    model.compile("sgd", temperature=0.5, jit_compile=False)
    cb = ConfidenceWeightsCallback({"val": (x, y)}, num_classes=10, batch_size=7)   # dernier batch incomplet
    cb.set_model(model)
    res = cb.compute(x, y)

    w = model.compute_confidence_weights(model(x, training=False)).numpy()     # (N, K), calcul direct
    np.testing.assert_allclose(res["overall"], w.mean(0), rtol=1e-5)
    np.testing.assert_allclose(res["w_max"], w.max(1).mean(), rtol=1e-5)
    for c in range(10):
        if (y == c).any():
            np.testing.assert_allclose(res["per_class"][c], w[y == c].mean(0), rtol=1e-5)
            np.testing.assert_allclose(res["per_class_w_max"][c], w[y == c].max(1).mean(), rtol=1e-5)
    np.testing.assert_array_equal(res["class_counts"], np.bincount(y, minlength=10))
    correct = (model(x, training=False).numpy().argmax(-1) == y[:, None])
    np.testing.assert_allclose(res["w_on_correct"], (w * correct).sum(1).mean(), rtol=1e-5)


def test_option_does_not_change_training_and_saves_every_epoch(tmp_path, tiny_data):
    off = run_multi_experiment(tiny_data, epochs=2, **COMMON)
    path = str(tmp_path / "confidence_weights.npz")
    on = run_multi_experiment(tiny_data, epochs=2, confidence_weights=True, confidence_on_train=True,
                              confidence_file=path, **COMMON)
    for key, values in off["history"].items():
        np.testing.assert_array_equal(values, on["history"][key])        # même entraînement
    for prefix in ("val", "train"):
        assert len(on["history"][f"{prefix}_w_max"]) == 2
        assert all(len(on["history"][f"{prefix}_w_agent{k}"]) == 2 for k in (1, 2, 3))
    with np.load(path) as f:
        assert list(f["epochs"]) == [1, 2]
        assert f["val_per_class"].shape == (2, 10, 3) and f["train_per_class"].shape == (2, 10, 3)
        assert f["val_overall"].shape == (2, 3)
        np.testing.assert_allclose(f["val_overall"].sum(1), 1, rtol=1e-5)
        np.testing.assert_allclose(f["val_overall"][:, 0], on["history"]["val_w_agent1"], rtol=1e-6)
        assert f["val_class_counts"].sum() == len(tiny_data["y_val"])


class _Interrupt(Exception):
    pass


class _StopAtEpoch(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 1:
            raise _Interrupt()       # « job tué » au début de la 2e époque


def test_interrupted_run_keeps_earlier_epochs(tmp_path, tiny_data):
    path = str(tmp_path / "confidence_weights.npz")
    kw = dict(epochs=2, confidence_weights=True, confidence_file=path, checkpoint_dir=str(tmp_path / "ckpt"), **COMMON)
    with pytest.raises(_Interrupt):
        run_multi_experiment(tiny_data, callbacks=[_StopAtEpoch()], **kw)
    with np.load(path) as f:
        assert list(f["epochs"]) == [1]
        first = f["val_per_class"][0].copy()
    res = run_multi_experiment(tiny_data, **kw)                          # reprise à la 2e époque
    with np.load(path) as f:
        assert list(f["epochs"]) == [1, 2]
        np.testing.assert_array_equal(f["val_per_class"][0], first)
    assert len(res["history"]["val_w_max"]) == 2                         # historique complet


def test_single_agent_has_no_confidence_weights(tiny_data):
    res = run_multi_experiment(tiny_data, epochs=1, confidence_weights=True,
                               **{**COMMON, "num_agents": 1})
    assert "val_w_max" not in res["history"]
    tf.keras.backend.clear_session()
