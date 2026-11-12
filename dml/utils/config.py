"""
Configuration des expériences : un seul endroit pour les valeurs par défaut, la lecture
du JSON, la ligne de commande et la vérification.

    cfg = load_config(AlphaSweepConfig, "configs/alpha_sweep.json", overrides={"epochs": 2})

Ordre de priorité : valeurs par défaut de la dataclass < fichier JSON < ligne de commande.
Une clé inconnue dans le JSON (faute de frappe) arrête le script avec un message clair.
Les clés qui commencent par "_" sont des commentaires et sont ignorées.

Ajouter un réglage : ajouter un champ à la dataclass avec opt(...). L'option de ligne de
commande (--mon-reglage) et la vérification des choix sont créées automatiquement.
"""
import argparse
import dataclasses
import difflib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from dml.core.consensus import available_strategies
from dml.core.losses import FORMS, available_losses
from dml.data import available_datasets
from dml.training.optimizers import DEFAULT_WD_EXCLUDE, available_optimizers

CODE_DIR = os.path.dirname(os.path.abspath(__file__))  # dans le dépôt Git (pour le commit)
PRECISIONS = ("float32", "mixed_bfloat16")
EVAL_ON = ("final", "best")
REFERENCE_LOSSES = ("ce", "same_as_agents")

# Stratégie de consensus imposée quand alpha = 0 (agents indépendants) dans le sweep
STRATEGY_NO_INTERACTION = "majority"


def opt(default, help: str = "", choices=None, kind: str = "auto"):
    """
    Champ de config.

    kind : comment lire la valeur sur la ligne de commande
        "auto"       d'après le type de la valeur par défaut (int, float, str, bool)
        "per_agent"  une valeur pour tous, ou une par agent (--lr 0.1 ou --lr 0.1 0.05 ...)
        "list_int" / "list_float"   liste (--seeds 0 1 2)
        "json"       objet JSON (--optimizer-kwargs '{"beta_1": 0.9}')
        "int_or_none" / "str_or_none" / "float_or_none"
        "jit"        auto | true | false
    """
    if isinstance(default, (list, dict)):
        value = default
        return field(default_factory=lambda: json.loads(json.dumps(value)),
                     metadata={"help": help, "choices": choices, "kind": kind})
    return field(default=default, metadata={"help": help, "choices": choices, "kind": kind})


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "oui")


def parse_jit(value):
    """Compilation XLA : True / False, ou "auto" (Keras l'active sur GPU, jamais sur CPU)."""
    if isinstance(value, bool):
        return value
    return "auto" if str(value).lower() == "auto" else _parse_bool(value)


# =============================================================================
# CONFIGS
# =============================================================================

@dataclass
class TrainConfig:
    """Réglages communs à toutes les expériences."""
    dataset: str = opt("cifar10", "Jeu de données.", choices=available_datasets())

    num_agents: int = opt(5, "Nombre d'agents K.")
    loss_type: str = opt("global_interaction", "Loss multi-agents (losses.py).", choices=available_losses())
    loss_form: str = opt("multiplicative", "L_k*(L_k + alpha*I_k) ou L_k + alpha*I_k.", choices=FORMS)
    strategy: str = opt("simplicial", "Stratégie de consensus (consensus.py).", choices=available_strategies())
    temperature: float = opt(1.0, "Température des poids de confiance w_k.")

    epochs: int = opt(200, "Nombre d'époques.")
    batch_size: int = opt(128, "Taille de batch.")
    optimizer: Union[str, List[str]] = opt("sgd", "Un nom pour tous les agents, ou un par agent.",
                                           choices=available_optimizers(), kind="per_agent")
    lr: Union[float, List[float]] = opt(0.1, "Un lr pour tous les agents, ou un par agent.", kind="per_agent")
    weight_decay: float = opt(5e-4, "Weight decay (porté par l'optimiseur).")
    wd_exclude_norm_bias: bool = opt(True, "Pas de weight decay sur BatchNorm et biais (false = ancien comportement).")
    optimizer_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = opt(
        None, "Arguments en plus de l'optimiseur (JSON), ex. {\"global_clipnorm\": 1.0}.", kind="json")
    warmup_epochs: int = opt(5, "Époques de warmup linéaire du lr.")
    augment: bool = opt(True, "Random crop + flip.")
    per_agent_views: bool = opt(False, "Une augmentation différente par agent.")

    val_size: int = opt(5000, "Taille de la validation, tirée du train.")
    train_subset: Optional[int] = opt(None, "N premiers exemples du train (tests rapides).", kind="int_or_none")
    n_runs: int = opt(3, "Nombre de runs (graines 0 à n_runs - 1) pour la médiane et les quantiles.")

    precision: str = opt("float32", "Précision des calculs.", choices=PRECISIONS)
    jit: Union[str, bool] = opt("auto", "Compilation XLA : auto (GPU seulement), true ou false.", kind="jit")
    eval_on: str = opt("final", "Évaluer le test sur les poids finaux ou ceux de la meilleure val_accuracy.",
                       choices=EVAL_ON)
    steps_per_execution: int = opt(8, "Pas d'entraînement par appel au GPU (plus rapide, mêmes calculs).")
    checkpoint_every: int = opt(1, "Sauvegarde de l'état de reprise toutes les N époques.")
    out_dir: str = opt("runs", "Dossier racine des runs.")
    tag: Optional[str] = opt(None, "Suffixe du dossier de run, pour séparer deux variantes "
                                   "(ex. lr001 -> runs/<exp>_<arch>_<dataset>_x<K>_lr001).", kind="str_or_none")

    EXPERIMENT = "experiment"          # nom de l'expérience, début du nom du dossier de run
    # Réglages qui peuvent changer entre deux lancements dans le même dossier de run
    # (ajouter des runs, des alphas...) sans mélanger deux expériences différentes.
    EXTENSIBLE = ("n_runs", "out_dir", "jit", "steps_per_execution", "checkpoint_every")

    # ------------------------------------------------------------------------
    def validate(self):
        """Vérifie les choix et les listes par agent ; SystemExit avec un message clair sinon."""
        errors = []
        for f in dataclasses.fields(self):
            choices, value = f.metadata.get("choices"), getattr(self, f.name)
            if choices is None or value is None:
                continue
            values = value if isinstance(value, list) else [value]
            bad = [v for v in values if (v.lower() if f.name.endswith("optimizer") else v) not in choices]
            if bad:
                errors.append(f"{f.name} = {bad} : choix possibles {list(choices)}")
            if f.metadata.get("kind") == "per_agent" and isinstance(value, list) and len(value) != self.num_agents:
                errors.append(f"{f.name} a {len(value)} valeurs pour {self.num_agents} agents")
        for name in ("lr", "optimizer_kwargs"):
            value = getattr(self, name)
            if isinstance(value, list) and len(value) != self.num_agents:
                errors.append(f"{name} a {len(value)} valeurs pour {self.num_agents} agents")
        errors += self._extra_checks()
        if errors:
            raise SystemExit("Config invalide :\n  - " + "\n  - ".join(errors))
        return self

    def _extra_checks(self) -> List[str]:
        return []

    def to_dict(self) -> Dict[str, Any]:
        return {"experiment": self.EXPERIMENT, **dataclasses.asdict(self)}

    @property
    def seeds(self) -> List[int]:
        """Graine de chaque run : 0, 1, ..., n_runs - 1."""
        return list(range(self.n_runs))

    @property
    def model_arch(self) -> str:
        return getattr(self, "arch", None) or getattr(self, "small_arch")

    def run_name(self) -> str:
        """<expérience>_<arch>_<dataset>_x<K>[_<tag>]"""
        name = f"{self.EXPERIMENT}_{self.model_arch}_{self.dataset}_x{self.num_agents}"
        return f"{name}_{self.tag}" if self.tag else name

    def run_dir(self) -> str:
        return os.path.join(self.out_dir, self.run_name())

    def signature(self) -> Dict[str, Any]:
        """Réglages qui définissent l'expérience : ils doivent être identiques dans un même dossier de run."""
        return {k: v for k, v in self.to_dict().items() if k not in self.EXTENSIBLE}

    def train_kwargs(self, verbose: int = 2) -> Dict[str, Any]:
        """Réglages d'entraînement communs, au format de experiment.run_multi_experiment."""
        return dict(
            epochs=self.epochs,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            wd_exclude=DEFAULT_WD_EXCLUDE if self.wd_exclude_norm_bias else (),
            warmup_epochs=self.warmup_epochs,
            augment=self.augment,
            form=self.loss_form,
            eval_on=self.eval_on,
            checkpoint_every=self.checkpoint_every,
            steps_per_execution=self.steps_per_execution,
            jit_compile=parse_jit(self.jit),
            verbose=verbose,
        )


@dataclass
class CollabVsBigConfig(TrainConfig):
    """K petits agents collaboratifs vs 1 grand agent au même nombre de paramètres."""
    EXPERIMENT = "collab_vs_big"
    small_arch: str = opt("resnet20", "Architecture des petits agents (le grand agent est la même, élargie).")
    alpha: float = opt(1.0, "Poids du terme d'interaction.")
    big_optimizer: Optional[str] = opt(None, "Optimiseur du grand agent (défaut : celui des petits).",
                                       choices=available_optimizers(), kind="str_or_none")
    big_lr: Optional[float] = opt(None, "lr du grand agent (défaut : celui des petits).", kind="float_or_none")
    big_optimizer_kwargs: Optional[Dict[str, Any]] = opt(None, "Arguments en plus pour le grand agent (JSON).",
                                                         kind="json")
    reference_loss: str = opt("ce", "Loss du grand agent : cross-entropy, ou même forme que les petits "
                                    "avec I_k = 0 (L_k² en multiplicatif).", choices=REFERENCE_LOSSES)

    def _extra_checks(self) -> List[str]:
        errors = []
        for key in ("optimizer", "lr", "optimizer_kwargs"):
            if getattr(self, f"big_{key}") is None and isinstance(getattr(self, key), list):
                errors.append(f"'{key}' est une liste (un par petit agent) : précise 'big_{key}' pour le grand agent")
        return errors

    def big_agent_optimizer(self) -> Dict[str, Any]:
        """Optimiseur du grand agent : big_* si donné, sinon celui des petits."""
        return {key: getattr(self, key) if getattr(self, f"big_{key}") is None else getattr(self, f"big_{key}")
                for key in ("optimizer", "lr", "optimizer_kwargs")}


@dataclass
class AlphaSweepConfig(TrainConfig):
    """Balayage du poids alpha de l'interaction."""
    EXPERIMENT = "alpha_sweep"
    EXTENSIBLE = TrainConfig.EXTENSIBLE + ("alphas",)
    arch: str = opt("resnet20", "Architecture des agents.")
    alphas: List[float] = opt([0, 0.5, 1, 10, 50, 100], "Valeurs de alpha à tester.", kind="list_float")
    lr: Union[float, List[float]] = opt(0.01, "Un lr pour tous les agents, ou un par agent "
                                              "(0.1 fait diverger la loss multiplicative).", kind="per_agent")

    def strategy_for(self, alpha: float) -> str:
        """Vote majoritaire sans interaction (alpha = 0), sinon la stratégie de la config."""
        return STRATEGY_NO_INTERACTION if alpha == 0 else self.strategy


DEFAULT_SEARCH_SPACE = {
    "lr": {"type": "float", "low": 1e-4, "high": 0.3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "alpha": {"type": "float", "low": 0.0, "high": 50.0},
}
SEARCH_METRICS = {"val_accuracy": "maximize", "val_agent_accuracy": "maximize", "val_loss": "minimize"}
SEARCH_SAMPLERS = ("tpe", "random", "cmaes")
SEARCH_PRUNERS = ("median", "hyperband", "none")


@dataclass
class HyperparamSearchConfig(TrainConfig):
    """Recherche des meilleurs hyperparamètres (Optuna) pour une architecture et un jeu de données."""
    EXPERIMENT = "best_hyperparams"
    arch: str = opt("resnet20", "Architecture des agents.")
    alpha: float = opt(1.0, "Poids du terme d'interaction (si absent de search_space).")
    n_runs: int = opt(1, "Ignoré : chaque essai utilise la graine 0.")
    n_workers: int = opt(4, "Nombre de processus (tâches du job array) qui travaillent sur la même recherche.")
    epochs: int = opt(20, "Époques par essai.")
    search_space: Dict[str, Any] = opt(DEFAULT_SEARCH_SPACE, "Espace de recherche (JSON, voir dml/tuning/space.py).",
                                       kind="json")
    n_trials: int = opt(30, "Nombre d'essais par processus (au total : n_workers x n_trials).")
    timeout_hours: Optional[float] = opt(None, "Arrête de lancer de nouveaux essais après ce temps.",
                                         kind="float_or_none")
    metric: str = opt("val_accuracy", "Critère : accuracy du consensus, accuracy moyenne des agents, "
                                      "ou loss sur la validation.", choices=tuple(SEARCH_METRICS))
    sampler: str = opt("tpe", "Stratégie de tirage des essais.", choices=SEARCH_SAMPLERS)
    pruner: str = opt("median", "Arrêt précoce des essais mauvais.", choices=SEARCH_PRUNERS)
    pruner_warmup_epochs: int = opt(3, "Époques avant qu'un essai puisse être arrêté.")
    study_seed: int = opt(0, "Graine du sampler.")

    SEARCH_FIELDS = ("search_space", "n_trials", "n_workers", "timeout_hours", "metric", "sampler", "pruner",
                     "pruner_warmup_epochs", "study_seed")
    EXTENSIBLE = TrainConfig.EXTENSIBLE + ("n_trials", "n_workers", "timeout_hours", "study_seed")

    @property
    def direction(self) -> str:
        return SEARCH_METRICS[self.metric]

    def searchable_fields(self) -> List[str]:
        return [f.name for f in dataclasses.fields(self)
                if f.name not in self.SEARCH_FIELDS + ("out_dir", "tag", "n_runs")]

    def _extra_checks(self) -> List[str]:
        from dml.tuning.space import validate_space
        errors = validate_space(self.search_space, self.searchable_fields())
        if self.val_size <= 0:
            errors.append("val_size doit être > 0 : les essais sont comparés sur la validation")
        return errors


# =============================================================================
# LECTURE
# =============================================================================

def _unknown_keys_message(cls, unknown: List[str]) -> str:
    known = [f.name for f in dataclasses.fields(cls)]
    lines = []
    for key in unknown:
        close = difflib.get_close_matches(key, known, n=1)
        lines.append(f"'{key}'" + (f" (voulais-tu dire '{close[0]}' ?)" if close else ""))
    return "Clé(s) inconnue(s) dans la config : " + ", ".join(lines)


def config_from_dict(cls, values: Dict[str, Any], strict: bool = True):
    """Crée une config ; strict=False ignore les clés inconnues (relecture d'un config.json enregistré)."""
    known = {f.name for f in dataclasses.fields(cls)}
    # "_..." = commentaires ; "experiment" = écrit dans config.json (une config enregistrée peut resservir)
    values = {k: v for k, v in values.items() if not k.startswith("_") and k != "experiment"}
    unknown = sorted(set(values) - known)
    if unknown and strict:
        raise SystemExit(_unknown_keys_message(cls, unknown))
    return cls(**{k: v for k, v in values.items() if k in known})


def load_config(cls, path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None, strict: bool = True):
    """Valeurs par défaut, puis fichier JSON, puis surcharges (valeurs None ignorées)."""
    values: Dict[str, Any] = {}
    if path is not None:
        if not os.path.exists(path):
            raise SystemExit(f"Fichier de config introuvable : {os.path.abspath(path)}")
        with open(path) as f:
            values.update(json.load(f))
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return config_from_dict(cls, values, strict=strict).validate()


# =============================================================================
# LIGNE DE COMMANDE
# =============================================================================

_SCALAR_TYPES = {int: int, float: float, str: str}


def add_config_arguments(parser: argparse.ArgumentParser, cls) -> None:
    """Ajoute une option --nom-du-champ par champ de la config (défaut None = valeur du JSON)."""
    group = parser.add_argument_group("réglages (surchargent le fichier de config)")
    for f in dataclasses.fields(cls):
        kind, help_ = f.metadata.get("kind", "auto"), f.metadata.get("help", "")
        choices = f.metadata.get("choices")
        flag = "--" + f.name.replace("_", "-")
        kwargs: Dict[str, Any] = {"dest": f.name, "default": None, "help": help_}
        if kind == "per_agent":
            sample = f.default if f.default is not dataclasses.MISSING else None
            kwargs.update(nargs="+", type=float if isinstance(sample, float) else str)
            if choices and not isinstance(sample, float):
                kwargs["choices"] = list(choices)
        elif kind == "list_int":
            kwargs.update(nargs="+", type=int)
        elif kind == "list_float":
            kwargs.update(nargs="+", type=float)
        elif kind == "json":
            kwargs.update(type=json.loads, metavar="JSON")
        elif kind in ("int_or_none", "float_or_none", "str_or_none"):
            kwargs["type"] = {"int_or_none": int, "float_or_none": float, "str_or_none": str}[kind]
            if choices:
                kwargs["choices"] = list(choices)
        elif kind == "jit":
            kwargs.update(type=parse_jit, metavar="auto|true|false")
        elif isinstance(f.default, bool):
            kwargs.update(type=_parse_bool, metavar="true|false")
        else:
            kwargs["type"] = _SCALAR_TYPES.get(type(f.default), str)
            if choices:
                kwargs["choices"] = list(choices)
        group.add_argument(flag, **kwargs)


def overrides_from_args(args: argparse.Namespace, cls) -> Dict[str, Any]:
    """Options de la ligne de commande -> surcharges ; une seule valeur par agent = commune à tous."""
    out = {}
    for f in dataclasses.fields(cls):
        value = getattr(args, f.name, None)
        if value is None:
            continue
        if f.metadata.get("kind") == "per_agent" and len(value) == 1:
            value = value[0]
        out[f.name] = value
    return out


# =============================================================================
# REPRODUCTIBILITÉ
# =============================================================================

def _git(*args) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", CODE_DIR, *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def environment_info(device: Optional[str] = None) -> Dict[str, Any]:
    """Commit Git, versions et machine : de quoi savoir quel code a produit un résultat."""
    import numpy as np
    import tensorflow as tf
    try:
        import keras
        keras_version = keras.__version__
    except (ImportError, AttributeError):
        keras_version = getattr(tf.keras, "__version__", None)
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),  # True = modifs non commitées
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "keras": keras_version,
        "numpy": np.__version__,
        "device": device,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "command": " ".join(sys.argv),
    }


def save_config(run_dir: str, config, **extra) -> str:
    """Écrit <run_dir>/config.json : les réglages, les infos en plus et l'environnement."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "config.json")
    payload = {**config.to_dict(), **extra}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
