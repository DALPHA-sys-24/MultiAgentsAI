"""
Choix des optimiseurs par leur nom.

Tous les optimiseurs Keras sont enregistrés ici sous un nom court ; on les crée avec
choix_optimiseur("adam", learning_rate=...). Chaque agent a son propre optimiseur,
et les agents peuvent en avoir des différents :

    build_agent_optimizers(["sgd", "adam", "adamw"], num_agents=3, lr=[0.1, 1e-3, 1e-3], ...)

Ajouter un optimiseur :

    @register_optimizer("mon_sgd")
    def mon_sgd(**kwargs):
        return tf.keras.optimizers.SGD(momentum=0.95, **kwargs)
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import tensorflow as tf

_OPTIMIZERS: Dict[str, Callable[..., tf.keras.optimizers.Optimizer]] = {}


def register_optimizer(name: str):
    """Décorateur : enregistre une fonction qui crée un optimiseur sous `name`."""
    def decorator(fn):
        if name in _OPTIMIZERS:
            raise ValueError(f"Optimiseur déjà enregistré : {name}")
        _OPTIMIZERS[name] = fn
        return fn
    return decorator


def available_optimizers() -> List[str]:
    return sorted(_OPTIMIZERS)


def _register_keras(name: str, class_name: str, **defaults):
    """Enregistre un optimiseur Keras ; ignoré s'il n'existe pas dans la version installée."""
    cls = getattr(tf.keras.optimizers, class_name, None)
    if cls is None:
        return
    register_optimizer(name)(lambda **kwargs: cls(**{**defaults, **kwargs}))


# SGD : momentum de Nesterov 0.9 par défaut (réglage standard ResNet / CIFAR).
# Pour le SGD sans momentum : optimizer_kwargs={"momentum": 0.0, "nesterov": False}.
_register_keras("sgd", "SGD", momentum=0.9, nesterov=True)
_register_keras("adam", "Adam")
_register_keras("adamw", "AdamW")
_register_keras("adamax", "Adamax")
_register_keras("nadam", "Nadam")
_register_keras("rmsprop", "RMSprop")
_register_keras("adagrad", "Adagrad")
_register_keras("adadelta", "Adadelta")
_register_keras("adafactor", "Adafactor")
_register_keras("ftrl", "Ftrl")
_register_keras("lion", "Lion")
_register_keras("lamb", "Lamb")
_register_keras("muon", "Muon")
_register_keras("schedule_free_adamw", "ScheduleFreeAdamW")


def choix_optimiseur(
    name: str,
    learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule, None] = None,
    weight_decay: Optional[float] = None,
    **kwargs,
) -> tf.keras.optimizers.Optimizer:
    """
    Crée un nouvel optimiseur à partir de son nom (voir available_optimizers()).

    learning_rate / weight_decay à None = valeur par défaut de Keras pour cet optimiseur.
    Les autres arguments (beta_1, momentum, clipnorm...) sont transmis à l'optimiseur.
    """
    key = name.lower()
    if key not in _OPTIMIZERS:
        raise ValueError(f"Optimiseur inconnu : {name}. Choix : {available_optimizers()}")
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    if weight_decay is not None:
        kwargs["weight_decay"] = weight_decay
    return _OPTIMIZERS[key](**kwargs)


# =============================================================================
# OPTIMISEURS DES AGENTS
# =============================================================================

def make_lr_schedule(lr: float, epochs: int, steps_per_epoch: int, warmup_epochs: int = 5):
    """Warmup linéaire puis décroissance cosinus jusqu'à 0."""
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0 if warmup_steps else lr,
        decay_steps=max(1, total_steps - warmup_steps),
        warmup_target=lr if warmup_steps else None,
        warmup_steps=warmup_steps,
    )


def _per_agent(value, num_agents: int, label: str) -> list:
    """Une valeur pour tous les agents, ou une liste avec une valeur par agent."""
    if isinstance(value, (list, tuple)):
        if len(value) != num_agents:
            raise ValueError(f"{label} : {len(value)} valeurs pour {num_agents} agents.")
        return list(value)
    return [value] * num_agents


def build_agent_optimizers(
    optimizer: Union[str, Sequence[str]],
    num_agents: int,
    *,
    lr: Union[float, Sequence[float]] = 0.1,
    weight_decay: Union[float, Sequence[float]] = 5e-4,
    epochs: int = 200,
    steps_per_epoch: int = 390,
    warmup_epochs: int = 5,
    optimizer_kwargs: Union[Dict[str, Any], Sequence[Dict[str, Any]], None] = None,
) -> List[tf.keras.optimizers.Optimizer]:
    """
    Un optimiseur neuf par agent, chacun avec son propre schedule warmup + cosinus.

    optimizer, lr, weight_decay et optimizer_kwargs acceptent soit une valeur commune,
    soit une liste (une valeur par agent). Le weight decay est porté par l'optimiseur,
    car le train_step du MultiAgentModel n'ajoute pas les régularisations des couches.
    """
    names = _per_agent(optimizer, num_agents, "optimizer")
    lrs = _per_agent(lr, num_agents, "lr")
    wds = _per_agent(weight_decay, num_agents, "weight_decay")
    extras = _per_agent(optimizer_kwargs or {}, num_agents, "optimizer_kwargs")
    return [
        choix_optimiseur(name, learning_rate=make_lr_schedule(lr_k, epochs, steps_per_epoch, warmup_epochs),
                         weight_decay=wd_k, **extra)
        for name, lr_k, wd_k, extra in zip(names, lrs, wds, extras)
    ]
