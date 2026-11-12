"""
Losses multi-agents, enregistrées par leur nom.

Chaque loss définit un terme d'interaction I_k (batch, K). La loss de l'agent k est
ensuite assemblée selon la forme choisie (`form`) :
    - "multiplicative" (défaut, méthode du projet) : L^k = mean( L_k * (L_k + alpha * I_k) )
    - "additive"                                    : L^k = mean( L_k + alpha * I_k )
où L_k = CE(y, y^k). Avec un seul agent, I_k n'existe pas (voir single_agent_loss).

Ajouter une loss :

    @register_loss("ma_loss")
    def ma_loss(ctx, outputs, log_p, raw_log_p, y):
        # outputs : probabilités (batch, K, C) ; log_p : log-probabilités bornées à log(1e-7)
        # raw_log_p : log-probabilités non bornées, ou None si seules les probabilités sont connues
    # y : labels (batch,)
        return interaction  # (batch, K)

Les cross-entropies sont calculées depuis les log-probabilités (log_softmax des logits),
bornées à log(1e-7) comme le fait Keras sur les probabilités. Cette borne est nécessaire :
sans elle, la forme multiplicative donne des gradients énormes sur les exemples très mal
classés et l'entraînement diverge.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf

from dml.core.consensus import (
    ConsensusContext,
    confidence_weights,
    leave_one_out_consensus,
    log_confidence_weights,
    majority_labels,
    pick_agent_labels,
    simplicial_proba,
)

EPS = 1e-7  # même seuil que Keras
LOG_EPS = float(np.log(EPS))

FORMS = ("multiplicative", "additive")
# Loss d'un agent seul (K = 1) : "ce" = cross-entropy ; "form" = la forme ci-dessus avec I_k = 0
# (L_k² en multiplicatif, CE en additif), pour comparer à loss de base identique.
SINGLE_AGENT_LOSSES = ("ce", "form")


@dataclass
class LossContext(ConsensusContext):
    """Réglages de la loss, partagés par tous les agents d'un même pas."""
    alpha: float = 1.0
    form: str = "multiplicative"
    # Agent tiré au sort par exemple (vote_random). Tiré une fois par pas, pour que la
    # loss de chaque agent voie le même tirage.
    vote_idx: Optional[tf.Tensor] = None


_LOSSES: Dict[str, Callable] = {}


def register_loss(name: str):
    """Décorateur : enregistre une fonction d'interaction sous `name`."""
    def decorator(fn):
        if name in _LOSSES:
            raise ValueError(f"Loss déjà enregistrée : {name}")
        _LOSSES[name] = fn
        return fn
    return decorator


def get_loss(name: str) -> Callable:
    if name not in _LOSSES:
        raise ValueError(f"Loss inconnue : {name}. Choix : {available_losses()}")
    return _LOSSES[name]


def available_losses() -> List[str]:
    return list(_LOSSES)


# =============================================================================
# BRIQUES
# =============================================================================

def bounded_log_probs(outputs, log_outputs=None):
    """Log-probabilités bornées à log(1e-7) (identique à la CE Keras sur probabilités tronquées)."""
    if log_outputs is None:
        return tf.math.log(tf.clip_by_value(tf.cast(outputs, tf.float32), EPS, 1.0))
    return tf.maximum(tf.cast(log_outputs, tf.float32), LOG_EPS)


def ce_labels(labels, log_p, num_classes: int):
    """CE(label, y^k) pour chaque agent. labels : (batch,) ou (batch, K). Renvoie (batch, K)."""
    one_hot = tf.one_hot(labels, depth=num_classes)
    if len(one_hot.shape) == 2:
        one_hot = one_hot[:, None, :]
    return -tf.reduce_sum(one_hot * log_p, axis=-1)


def ce_soft(targets, log_p):
    """CE(cible douce, y^k) pour chaque agent. targets : (batch, C) ou (batch, K, C). Renvoie (batch, K)."""
    if len(targets.shape) == 2:
        targets = targets[:, None, :]
    return -tf.reduce_sum(targets * log_p, axis=-1)


def combine(l_k, interaction, alpha: float, form: str):
    """Assemble L_k et I_k selon la forme, moyenne sur le batch. Renvoie (K,)."""
    if form == "multiplicative":
        per_example = l_k * (l_k + alpha * interaction)
    elif form == "additive":
        per_example = l_k + alpha * interaction
    else:
        raise ValueError(f"Forme inconnue : {form}. Choix : {FORMS}")
    return tf.reduce_mean(per_example, axis=0)


# =============================================================================
# TERMES D'INTERACTION I_k
# =============================================================================

@register_loss("leave_one_out")
def leave_one_out(ctx, outputs, log_p, raw_log_p, y):
    """I_k = (1 - w_k) * CE(y^cons_{-k}, y^k)"""
    w_k = confidence_weights(outputs, ctx.temperature, raw_log_p)
    loo = leave_one_out_consensus(outputs, w_k)
    return (1.0 - w_k) * ce_soft(loo, log_p)


@register_loss("global_interaction")
def global_interaction(ctx, outputs, log_p, raw_log_p, y):
    """I_k = CE(y^cons, y^k) + CE(y, y^cons)"""
    log_w = log_confidence_weights(outputs, ctx.temperature, raw_log_p)
    cons = simplicial_proba(outputs, tf.exp(log_w))
    # log y^cons = logsumexp_k(log w_k + log y^k), borné comme les autres CE
    raw = log_p if raw_log_p is None else raw_log_p
    log_cons = tf.maximum(tf.reduce_logsumexp(log_w[..., None] + raw, axis=1), LOG_EPS)
    l_glob = ce_labels(y, log_cons[:, None, :], ctx.num_classes)   # (batch, 1)
    l_loc = ce_soft(cons, log_p)                                     # (batch, K)
    return l_loc + l_glob


@register_loss("vote_random")
def vote_random(ctx, outputs, log_p, raw_log_p, y):
    """I_k = (1 - w_k) * CE(y^vote, y^k), y^vote = classe prédite par un agent tiré au sort."""
    w_k = confidence_weights(outputs, ctx.temperature, raw_log_p)
    idx = ctx.vote_idx if ctx.vote_idx is not None else ctx.random_agent(tf.shape(outputs)[0])
    vote = pick_agent_labels(outputs, idx)
    return (1.0 - w_k) * ce_labels(vote, log_p, ctx.num_classes)


@register_loss("majority_hard")
def majority_hard(ctx, outputs, log_p, raw_log_p, y):
    """I_k = (1 - w_k) * CE(y^majority, y^k)"""
    w_k = confidence_weights(outputs, ctx.temperature, raw_log_p)
    majority = majority_labels(outputs, ctx.num_classes)
    return (1.0 - w_k) * ce_labels(majority, log_p, ctx.num_classes)


@register_loss("majority_loo")
def majority_loo(ctx, outputs, log_p, raw_log_p, y):
    """I_k = (1 - w_k) * CE(y^majority_{-k}, y^k), vote majoritaire des K-1 autres agents."""
    w_k = confidence_weights(outputs, ctx.temperature, raw_log_p)
    preds = tf.argmax(outputs, axis=-1, output_type=tf.int32)          # (batch, K)
    votes = tf.one_hot(preds, depth=ctx.num_classes)                    # (batch, K, C)
    votes_loo = tf.reduce_sum(votes, axis=1, keepdims=True) - votes     # (batch, K, C)
    majority = tf.argmax(votes_loo, axis=-1, output_type=tf.int32)      # (batch, K)
    return (1.0 - w_k) * ce_labels(majority, log_p, ctx.num_classes)


# =============================================================================
# CALCUL DES LOSSES PAR AGENT
# =============================================================================

def agent_losses(loss_type: str, ctx: LossContext, outputs, y, log_outputs=None,
                 single_agent_loss: str = "ce"):
    """
    Loss de chaque agent, (K,).

    Args:
        outputs: probabilités (batch, K, C).
        y: labels (batch,) int32.
        log_outputs: log-probabilités (log_softmax des logits), plus précises si données.
        single_agent_loss: pour K = 1, "ce" ou "form" (voir SINGLE_AGENT_LOSSES).
    """
    log_p = bounded_log_probs(outputs, log_outputs)
    l_k = ce_labels(y, log_p, ctx.num_classes)
    if ctx.num_agents == 1:
        if single_agent_loss == "ce":
            return tf.reduce_mean(l_k, axis=0)
        return combine(l_k, tf.zeros_like(l_k), ctx.alpha, ctx.form)
    raw_log_p = None if log_outputs is None else tf.cast(log_outputs, tf.float32)
    interaction = get_loss(loss_type)(ctx, outputs, log_p, raw_log_p, y)
    return combine(l_k, interaction, ctx.alpha, ctx.form)
