"""Losses : réduction à la CE pour un agent, différence finie, registre."""
import numpy as np
import pytest
import tensorflow as tf

from dml.core import losses as loss_lib
from conftest import TINY
from dml.core.multiagent import LOSS_FUNCTIONS, MultiAgentModel

ALL_LOSSES = list(LOSS_FUNCTIONS)


def make_model(k=3, loss_type="global_interaction", alpha=1.0, **compile_kwargs):
    tf.keras.utils.set_random_seed(0)
    model = MultiAgentModel(k, num_classes=10, **TINY)
    model.compile("sgd", loss_type=loss_type, alpha=alpha, **compile_kwargs)
    x = tf.random.normal((16, 8, 8, 3), seed=1)
    y = tf.constant(np.random.default_rng(0).integers(0, 10, 16), tf.int32)
    model(x)
    return model, x, y


def test_five_losses_registered():
    assert set(ALL_LOSSES) >= {"leave_one_out", "global_interaction", "vote_random",
                               "majority_hard", "majority_loo"}


def test_unknown_loss_is_refused():
    model = MultiAgentModel(2, num_classes=10, **TINY)
    with pytest.raises(ValueError):
        model.compile("sgd", loss_type="pas_une_loss")


def test_single_agent_ce_matches_keras():
    model, x, y = make_model(k=1)
    probs = model(x)[:, 0]
    keras_ce = tf.keras.losses.sparse_categorical_crossentropy(y, probs)
    loss = model.compute_losses(model(x), y)[0]
    np.testing.assert_allclose(float(loss), float(tf.reduce_mean(keras_ce)), rtol=1e-5)


def test_single_agent_form_is_squared_ce():
    model, x, y = make_model(k=1, single_agent_loss="form")
    probs = model(x)[:, 0]
    ce = tf.keras.losses.sparse_categorical_crossentropy(y, probs)
    loss = model.compute_losses(model(x), y)[0]
    np.testing.assert_allclose(float(loss), float(tf.reduce_mean(ce ** 2)), rtol=1e-5)


def test_additive_form_alpha0_is_ce():
    model, x, y = make_model(k=3, alpha=0.0, form="additive")
    out = model(x)
    ce = tf.keras.losses.sparse_categorical_crossentropy(tf.tile(y[:, None], [1, 3]), out)
    expected = tf.reduce_mean(ce, axis=0)
    np.testing.assert_allclose([float(v) for v in model.compute_losses(out, y)], expected.numpy(), rtol=1e-5)


@pytest.mark.parametrize("loss_type", ["leave_one_out", "global_interaction", "majority_hard"])
def test_gradient_matches_finite_difference(loss_type):
    """Gradient TensorFlow vs différence finie, par rapport aux logits."""
    rng = np.random.default_rng(1)
    logits = tf.Variable(rng.normal(size=(8, 3, 10)).astype("float32"))
    y = tf.constant(rng.integers(0, 10, 8), tf.int32)
    ctx = loss_lib.LossContext(num_agents=3, num_classes=10, alpha=2.0)

    def total():
        lo = tf.nn.log_softmax(logits, axis=-1)
        return tf.reduce_sum(loss_lib.agent_losses(loss_type, ctx, tf.exp(lo), y, lo))

    with tf.GradientTape() as tape:
        value = total()
    grad = tape.gradient(value, logits).numpy()

    eps, base = 1e-2, logits.numpy()
    for idx in [(0, 0, 0), (3, 1, 4), (7, 2, 9)]:
        plus, minus = base.copy(), base.copy()
        plus[idx] += eps
        minus[idx] -= eps
        logits.assign(plus)
        f_plus = float(total())
        logits.assign(minus)
        f_minus = float(total())
        logits.assign(base)
        fd = (f_plus - f_minus) / (2 * eps)
        assert abs(fd - grad[idx]) <= 2e-2 * max(1.0, abs(fd)), (idx, fd, grad[idx])


def test_custom_loss_can_be_registered():
    name = "test_zero_interaction"
    if name not in LOSS_FUNCTIONS:
        @loss_lib.register_loss(name)
        def _zero(ctx, outputs, log_p, raw_log_p, y):
            return tf.zeros(tf.shape(outputs)[:2])
    model, x, y = make_model(loss_type=name, alpha=5.0)
    out = model(x)
    ce = tf.keras.losses.sparse_categorical_crossentropy(tf.tile(y[:, None], [1, 3]), out)
    np.testing.assert_allclose([float(v) for v in model.compute_losses(out, y)],
                               tf.reduce_mean(ce ** 2, axis=0).numpy(), rtol=1e-5)


def _kl(p, q):
    """KL(p || q), log q borné à log(1e-7) comme toutes les CE du projet."""
    p_log_p = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return float(np.sum(p_log_p - p * np.log(np.clip(q, 1e-7, 1.0))))


@pytest.mark.parametrize("form", ["additive", "multiplicative"])
def test_dml_matches_zhang_formula_for_two_agents(form):
    alpha = 0.7
    model, x, y = make_model(k=2, loss_type="dml", alpha=alpha, form=form)
    out = model(x).numpy().astype("float64")                      # (batch, 2, C)
    losses = [float(l) for l in model.compute_losses(model(x), y)]
    yi = y.numpy()
    for k, j in ((0, 1), (1, 0)):
        ce = -np.log(np.clip(out[np.arange(len(yi)), k, yi], 1e-7, 1.0))
        kl = np.array([_kl(out[b, j], out[b, k]) for b in range(len(yi))])
        expected = ce + alpha * kl if form == "additive" else ce * (ce + alpha * kl)
        np.testing.assert_allclose(losses[k], expected.mean(), rtol=1e-4)


def test_dml_gradient_of_agent_does_not_reach_other_agents():
    model, x, y = make_model(k=3, loss_type="dml", alpha=1.0, form="additive")
    with tf.GradientTape() as tape:
        loss_0 = model.compute_losses(model(x, training=True), y)[0]
    others = [v for a in model._agents[1:] for v in a.trainable_variables]
    grads = tape.gradient(loss_0, others)
    assert all(g is None or float(tf.reduce_max(tf.abs(g))) == 0.0 for g in grads)
