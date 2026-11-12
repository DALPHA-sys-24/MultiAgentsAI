"""Pas d'entraînement, évaluation reproductible, weight decay, checkpoints et reprise."""
import os

import numpy as np
import pytest
import tensorflow as tf

from dml.training.checkpoint import TrainingCheckpoint
from conftest import TINY
from dml.training.runner import run_multi_experiment
from dml.core.multiagent import MultiAgentModel
from dml.training.optimizers import build_agent_optimizers


def _weights(model):
    return [w for agent in model.agents for w in agent.get_weights()]


def test_vote_strategy_is_reproducible_at_evaluation():
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 8, 8, 3)).astype("float32")
    y = rng.integers(0, 10, 64).astype("int32")
    model = MultiAgentModel(3, num_classes=10, **TINY)
    model.compile("sgd", loss_type="global_interaction", strategy="vote", jit_compile=False)
    first = model.evaluate(x, y, verbose=0)["accuracy"]
    second = model.evaluate(x, y, verbose=0)["accuracy"]
    assert first == second
    p1 = model.predict(x, verbose=0)["consensus"]
    p2 = model.predict(x, verbose=0)["consensus"]
    np.testing.assert_array_equal(p1, p2)


def test_weight_decay_skips_batchnorm_and_bias():
    model = MultiAgentModel(1, num_classes=10, **TINY)
    opts = build_agent_optimizers("sgd", 1, lr=0.1, weight_decay=5e-4, epochs=1, steps_per_epoch=1, warmup_epochs=0)
    model.compile(opts)
    model(tf.zeros((1, 8, 8, 3)))
    opt = opts[0]
    names = {v.name.split("/")[-1].split(":")[0]: opt._use_weight_decay(v) for v in model.agents[0].trainable_variables}
    assert names["kernel"] is True
    assert names["gamma"] is False and names["beta"] is False and names["bias"] is False
    old = build_agent_optimizers("sgd", 1, lr=0.1, weight_decay=5e-4, epochs=1, steps_per_epoch=1,
                                 warmup_epochs=0, wd_exclude=None)[0]
    assert all(old._use_weight_decay(v) for v in model.agents[0].trainable_variables)


def test_checkpoint_restores_weights_and_optimizer(tmp_path, tiny_data):
    common = dict(num_agents=2, epochs=1, batch_size=16, optimizer="sgd", lr=0.05, warmup_epochs=0,
                  augment=False, seed=0, jit_compile=False, **TINY)
    run_multi_experiment(tiny_data, checkpoint_dir=str(tmp_path / "a"), **common)
    ckpt_dir = tmp_path / "a"
    assert (ckpt_dir / "final_weights.npz").exists() and (ckpt_dir / "best_weights.npz").exists()
    assert not (ckpt_dir / "state.npz").exists()          # état de reprise supprimé à la fin


class _Interrupt(Exception):
    pass


class _StopAtEpoch(tf.keras.callbacks.Callback):
    def __init__(self, epoch):
        super().__init__()
        self.epoch = epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epoch:
            raise _Interrupt()


def test_interrupted_run_resumes_at_next_epoch(tmp_path, tiny_data):
    common = dict(num_agents=2, epochs=3, batch_size=16, optimizer="sgd", lr=0.05, warmup_epochs=0,
                  augment=False, seed=0, jit_compile=False, **TINY)
    ckpt_dir = str(tmp_path / "run")
    with pytest.raises(_Interrupt):   # « job tué » au début de la 3e époque
        run_multi_experiment(tiny_data, checkpoint_dir=ckpt_dir, callbacks=[_StopAtEpoch(2)], **common)
    assert os.path.exists(os.path.join(ckpt_dir, "state.npz"))

    # L'état sauvegardé se recharge à l'identique dans un modèle neuf
    tf.keras.utils.set_random_seed(0)
    model = MultiAgentModel(2, num_classes=10, **TINY)
    opts = build_agent_optimizers("sgd", 2, lr=0.05, epochs=3, steps_per_epoch=4, warmup_epochs=0)
    model.compile(opts)
    model(tf.zeros((1, 8, 8, 3)))
    ckpt = TrainingCheckpoint(ckpt_dir)
    assert ckpt.restore(model) == 2
    assert int(opts[0].iterations.numpy()) == 2 * 4        # 2 époques de 4 pas
    assert len(ckpt.history["loss"]) == 2

    res = run_multi_experiment(tiny_data, checkpoint_dir=ckpt_dir, csv_log=str(tmp_path / "h.csv"), **common)
    assert len(res["history"]["loss"]) == 3                 # historique complet malgré la coupure
    assert res["best_epoch"] in (1, 2, 3)


def test_eval_on_best(tmp_path, tiny_data):
    res = run_multi_experiment(tiny_data, num_agents=2, epochs=2, batch_size=16, optimizer="sgd", lr=0.05,
                               warmup_epochs=0, augment=False, seed=0, jit_compile=False,
                               checkpoint_dir=str(tmp_path / "b"), eval_on="best", **TINY)
    assert res["evaluated_on"] == "best"
