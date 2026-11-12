"""
Espace de recherche décrit en JSON, une entrée par hyperparamètre :

    "search_space": {
        "lr":           {"type": "float", "low": 1e-4, "high": 0.3, "log": true},
        "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-2, "log": true},
        "alpha":        {"type": "float", "low": 0.0, "high": 50.0},
        "batch_size":   {"type": "int", "low": 64, "high": 256, "step": 64},
        "optimizer":    {"type": "categorical", "choices": ["sgd", "adam", "adamw"]},
        "optimizer_kwargs": {"type": "categorical", "choices": [null, {"global_clipnorm": 1.0}]}
    }

Chaque clé est un champ de la config de l'expérience (dml/utils/config.py) : la valeur
tirée remplace celle de la config pour cet essai. Un hyperparamètre absent garde la
valeur de la config.
"""
import json
from typing import Any, Dict, Iterable

TYPES = ("float", "int", "categorical")


def validate_space(space: Dict[str, Dict[str, Any]], allowed: Iterable[str]) -> list:
    """Liste des erreurs de l'espace de recherche (vide si tout va bien)."""
    allowed = set(allowed)
    errors = []
    if not space:
        errors.append("search_space est vide : rien à chercher")
    for name, spec in (space or {}).items():
        if name not in allowed:
            errors.append(f"search_space : '{name}' n'est pas un réglage de la config")
            continue
        kind = spec.get("type")
        if kind not in TYPES:
            errors.append(f"search_space['{name}'] : type {kind!r}, choix {TYPES}")
        elif kind in ("float", "int"):
            if "low" not in spec or "high" not in spec:
                errors.append(f"search_space['{name}'] : il faut 'low' et 'high'")
            elif spec["low"] > spec["high"]:
                errors.append(f"search_space['{name}'] : low > high")
            elif spec.get("log") and spec["low"] <= 0:
                errors.append(f"search_space['{name}'] : log=true demande low > 0")
        elif not spec.get("choices"):
            errors.append(f"search_space['{name}'] : il faut une liste 'choices'")
    return errors


def _encode(value):
    """Optuna n'accepte que des choix simples : les listes / dicts passent par du JSON."""
    return json.dumps(value) if isinstance(value, (dict, list)) or value is None else value


def _decode(value):
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return value
        return decoded if isinstance(decoded, (dict, list)) or decoded is None else value
    return value


def suggest(trial, space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Tire une valeur par hyperparamètre pour cet essai."""
    params = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                               log=spec.get("log", False), step=spec.get("step"))
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                             step=spec.get("step", 1), log=spec.get("log", False))
        else:
            choice = trial.suggest_categorical(name, [_encode(c) for c in spec["choices"]])
            params[name] = _decode(choice)
    return params


def decode_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Paramètres d'un essai terminé (study.best_params) -> valeurs de config."""
    return {k: _decode(v) for k, v in params.items()}
