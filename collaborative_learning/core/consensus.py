"""
Poids de confiance et stratégies de consensus, enregistrés par leur nom.

Toutes les fonctions prennent des probabilités `outputs` de forme (batch, K, C).
Une stratégie renvoie soit des probabilités (batch, C), soit des labels durs (batch,).

Ajouter une stratégie :

    @register_strategy("ma_strategie")
    def ma_strategie(outputs, ctx):
        # ctx : ConsensusContext (num_agents, num_classes, temperature, rng)
        return ...  # (batch, C) probabilités ou (batch,) labels int32
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import tensorflow as tf


@dataclass
class ConsensusContext:
    """Ce dont une stratégie a besoin, en plus des sorties des agents."""
    num_agents: int
    num_classes: int
    temperature: float = 1.0
    # Tirage aléatoire d'entiers dans [0, maxval) : fn(shape, maxval) -> tenseur int32.
    # None = tf.random.uniform (aléatoire à chaque appel).
    randint: Optional[Callable] = None

    def random_agent(self, batch_size):
        """Indice d'un agent tiré au hasard pour chaque exemple, (batch,)."""
        if self.randint is not None:
            return self.randint([batch_size], self.num_agents)
        return tf.random.uniform([batch_size], 0, self.num_agents, dtype=tf.int32)


_STRATEGIES: Dict[str, Callable] = {}


def register_strategy(name: str):
    """Décorateur : enregistre une stratégie de consensus sous `name`."""
    def decorator(fn):
        if name in _STRATEGIES:
            raise ValueError(f"Stratégie déjà enregistrée : {name}")
        _STRATEGIES[name] = fn
        return fn
    return decorator


def get_strategy(name: str) -> Callable:
    if name not in _STRATEGIES:
        raise ValueError(f"Stratégie inconnue : {name}. Choix : {available_strategies()}")
    return _STRATEGIES[name]


def available_strategies() -> List[str]:
    return list(_STRATEGIES)


# =============================================================================
# POIDS DE CONFIANCE
# =============================================================================

def log_confidence_weights(outputs, temperature: float, log_outputs=None):
    """log w_k = log_softmax(-H_k / tau), H_k = entropie de y^k. Renvoie (batch, K)."""
    p = tf.cast(outputs, tf.float32)
    if log_outputs is None:
        # xlogy(p, p) = p * log(p), vaut 0 si p = 0 (évite les NaN)
        h_k = -tf.reduce_sum(tf.math.xlogy(p, p), axis=-1)
    else:
        h_k = -tf.reduce_sum(p * tf.cast(log_outputs, tf.float32), axis=-1)
    return tf.nn.log_softmax(-h_k / temperature, axis=-1)


def confidence_weights(outputs, temperature: float, log_outputs=None):
    """w_k = softmax(-H_k / tau), (batch, K)."""
    return tf.exp(log_confidence_weights(outputs, temperature, log_outputs))


def leave_one_out_consensus(outputs, weights):
    """y^cons_{-k} = sum_{j≠k} w_j y^j / sum_{j≠k} w_j. Renvoie (batch, K, C)."""
    w_exp = tf.expand_dims(weights, -1)
    weighted = outputs * w_exp
    total_w = tf.reduce_sum(weighted, axis=1, keepdims=True)
    total_s = tf.reduce_sum(weights, axis=1, keepdims=True)
    return tf.math.divide_no_nan(total_w - weighted, tf.expand_dims(total_s - weights, -1))


def simplicial_proba(outputs, weights):
    """y^cons = sum_k w_k y^k, (batch, C)."""
    return tf.reduce_sum(outputs * tf.expand_dims(weights, -1), axis=1)


def majority_labels(outputs, num_classes: int):
    """Vote majoritaire des K agents, (batch,)."""
    preds = tf.argmax(outputs, axis=-1, output_type=tf.int32)
    votes = tf.one_hot(preds, depth=num_classes)
    return tf.argmax(tf.reduce_sum(votes, axis=1), axis=-1, output_type=tf.int32)


def pick_agent_labels(outputs, idx):
    """Classe prédite par l'agent idx[b] pour chaque exemple b, (batch,)."""
    preds = tf.argmax(outputs, axis=-1, output_type=tf.int32)
    batch_size = tf.shape(preds)[0]
    return tf.gather_nd(preds, tf.stack([tf.range(batch_size), idx], axis=1))


# =============================================================================
# STRATÉGIES
# =============================================================================

@register_strategy("vote")
def vote(outputs, ctx: ConsensusContext):
    """Sélection aléatoire d'un agent pour chaque exemple."""
    return pick_agent_labels(outputs, ctx.random_agent(tf.shape(outputs)[0]))


@register_strategy("majority")
def majority(outputs, ctx: ConsensusContext):
    """Vote majoritaire."""
    return majority_labels(outputs, ctx.num_classes)


@register_strategy("simplicial")
def simplicial(outputs, ctx: ConsensusContext):
    """Moyenne pondérée par la confiance (probabilités)."""
    return simplicial_proba(outputs, confidence_weights(outputs, ctx.temperature))


@register_strategy("k_fold")
def k_fold(outputs, ctx: ConsensusContext):
    """Vote majoritaire sur les K consensus leave-one-out."""
    loo = leave_one_out_consensus(outputs, confidence_weights(outputs, ctx.temperature))
    return majority_labels(loo, ctx.num_classes)


def consensus_to_labels(cons):
    """Probabilités (batch, C) ou labels (batch,) -> labels (batch,)."""
    return tf.argmax(cons, axis=-1, output_type=tf.int32) if len(cons.shape) == 2 else cons
