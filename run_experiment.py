"""
Expérience « collaboration vs grand modèle ».

Compare, à nombre de paramètres entraînables égal :
    - K petits agents (même architecture) entraînés ensemble avec la loss multi-agents
      (par défaut global_interaction + simplicial) ;
    - un grand agent seul, de la même architecture élargie (width) pour avoir autant
      de paramètres que les K petits réunis, entraîné en cross-entropy.

Les réglages sont lus dans un fichier de config JSON ; chaque option de la ligne de
commande le surcharge.

Exemples :
    python run_experiment.py --config configs/collab_vs_big.json
    python run_experiment.py --config configs/collab_vs_big.json --dataset fashion_mnist --epochs 50
    python run_experiment.py --config configs/collab_vs_big.json --cpu --train-subset 2000 --epochs 1

Chaque lancement crée results/<date>_<nom>/ contenant :
    config.json     config utilisée + nombres de paramètres et largeur du grand agent
    results.json    métriques par graine (petits agents, consensus, grand agent) et résumé
    probs/          probabilités de test (.npy) par graine
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np

from data import load_dataset
from device import configure_device
from multiagent import LOSS_FUNCTIONS, STRATEGIES
from optimizers import available_optimizers
from tools import _cleanup, count_trainable_params, match_width, run_multi_experiment, run_single_reference

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "collab_vs_big.json")

# Valeurs par défaut, utilisées pour toute clé absente du fichier de config
DEFAULTS = {
    "name": "collab_vs_big",
    "dataset": "cifar10",
    "small_arch": "resnet20",
    "num_agents": 5,
    "loss_type": "global_interaction",
    "strategy": "simplicial",
    "alpha": 1.0,
    "temperature": 1.0,
    "epochs": 200,
    "batch_size": 128,
    "optimizer": "sgd",            # un nom, ou une liste d'un nom par petit agent
    "lr": 0.1,                     # un lr, ou une liste d'un lr par petit agent
    "weight_decay": 5e-4,
    "optimizer_kwargs": None,      # ex. {"beta_1": 0.9} ; ou une liste par petit agent
    "big_optimizer": None,         # None = même optimiseur que les petits (s'il est unique)
    "big_lr": None,
    "big_optimizer_kwargs": None,
    "warmup_epochs": 5,
    "augment": True,
    "per_agent_views": False,
    "val_size": 5000,
    "train_subset": None,
    "seeds": [0, 1, 2],
    "precision": "float32",
    "jit": "auto",                 # XLA : "auto" = activé sur GPU si possible, désactivé sur CPU
    "out_dir": "results",
}


# =============================================================================
# CONFIGURATION
# =============================================================================

def parse_jit(value):
    """Compilation XLA : True / False, ou "auto" (Keras l'active sur GPU, jamais sur CPU)."""
    if isinstance(value, bool):
        return value
    v = str(value).lower()
    if v == "auto":
        return "auto"
    return v in ("1", "true", "yes")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Fichier de config JSON (défaut : configs/collab_vs_big.json "
                                     "à côté du script s'il existe, sinon valeurs par défaut).")
    # Surcharges optionnelles (None = valeur du fichier de config)
    p.add_argument("--name")
    p.add_argument("--dataset", choices=["fashion_mnist", "cifar10", "cifar100"])
    p.add_argument("--small-arch", help="Architecture des petits agents (ex. resnet20).")
    p.add_argument("--num-agents", type=int)
    p.add_argument("--loss-type", choices=list(LOSS_FUNCTIONS))
    p.add_argument("--strategy", choices=list(STRATEGIES))
    p.add_argument("--alpha", type=float)
    p.add_argument("--temperature", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--optimizer", nargs="+", choices=available_optimizers(),
                   help="Un nom pour tous les petits agents, ou un nom par agent.")
    p.add_argument("--lr", type=float, nargs="+", help="Un lr pour tous, ou un par agent.")
    p.add_argument("--big-optimizer", choices=available_optimizers())
    p.add_argument("--big-lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--warmup-epochs", type=int)
    p.add_argument("--augment", type=lambda v: v.lower() in ("1", "true", "yes"), metavar="true|false")
    p.add_argument("--per-agent-views", type=lambda v: v.lower() in ("1", "true", "yes"), metavar="true|false")
    p.add_argument("--val-size", type=int)
    p.add_argument("--train-subset", type=int, help="N premiers exemples du train (tests rapides).")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--precision", choices=["float32", "mixed_bfloat16"])
    p.add_argument("--jit", type=parse_jit, metavar="auto|true|false",
                   help="Compilation XLA : auto (défaut, GPU seulement), true ou false.")
    p.add_argument("--out-dir")
    p.add_argument("--cpu", action="store_true", help="Force le CPU même si un GPU est disponible.")
    p.add_argument("--verbose", type=int, default=2)
    return p.parse_args()


def load_config(args) -> dict:
    """Valeurs par défaut, puis fichier de config, puis surcharges de la ligne de commande."""
    config = dict(DEFAULTS)
    path = args.config or (DEFAULT_CONFIG_PATH if os.path.exists(DEFAULT_CONFIG_PATH) else None)
    if path is not None:
        if not os.path.exists(path):
            raise SystemExit(f"Fichier de config introuvable : {os.path.abspath(path)}")
        with open(path) as f:
            config.update(json.load(f))
        print(f"[config] {os.path.abspath(path)}")
    else:
        print("[config] aucun fichier de config, valeurs par défaut du script")
    for key, value in vars(args).items():
        if key not in ("config", "cpu", "verbose") and value is not None:
            # --optimizer sgd / --lr 0.1 : une seule valeur = commune à tous les agents
            single = key in ("optimizer", "lr") and len(value) == 1
            config[key] = value[0] if single else value
    return config


def check_optimizers(config: dict):
    """Erreur claire avant tout entraînement si un nom d'optimiseur est inconnu."""
    names = config["optimizer"] if isinstance(config["optimizer"], list) else [config["optimizer"]]
    names = names + [big_agent_optimizer(config)["optimizer"]]
    unknown = [n for n in names if n.lower() not in available_optimizers()]
    if unknown:
        raise SystemExit(f"Optimiseur(s) inconnu(s) : {unknown}. Choix : {available_optimizers()}")


def big_agent_optimizer(config: dict) -> dict:
    """Optimiseur du grand agent : big_* si donné, sinon celui des petits s'il est commun."""
    out = {}
    for key in ("optimizer", "lr", "optimizer_kwargs"):
        value = config.get(f"big_{key}")
        if value is None:
            value = config[key]
            if isinstance(value, list):
                raise SystemExit(f"'{key}' est une liste (un par petit agent) : "
                                 f"précise 'big_{key}' pour le grand agent.")
        out[key] = value
    return out


# =============================================================================
# EXPÉRIENCE
# =============================================================================

def train_kwargs(config: dict, verbose: int) -> dict:
    """Réglages d'entraînement communs aux petits agents et au grand agent (hors optimiseur)."""
    return dict(
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        weight_decay=config["weight_decay"],
        warmup_epochs=config["warmup_epochs"],
        augment=config["augment"],
        jit_compile=parse_jit(config["jit"]),
        verbose=verbose,
    )


QUANTILES = (0.05, 0.95)


def summarize(values) -> dict:
    """Médiane et quantiles 5 % / 95 % sur les graines (moyenne et écart-type gardés pour info)."""
    values = [v for v in values if v is not None]
    lo, hi = np.quantile(values, QUANTILES)
    return {"median": float(np.median(values)), "q05": float(lo), "q95": float(hi),
            "mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}


def run_seed(data: dict, config: dict, big: dict, seed: int, verbose: int, probs_dir: str) -> dict:
    """Entraîne les K petits agents puis le grand agent pour une graine."""
    common = train_kwargs(config, verbose)
    k = config["num_agents"]

    print(f"\n=== Graine {seed} : {k} x {config['small_arch']} ({config['loss_type']} + {config['strategy']}) ===")
    collab = run_multi_experiment(
        data, archs=config["small_arch"], num_agents=k, seed=seed,
        loss_type=config["loss_type"], strategy=config["strategy"],
        alpha=config["alpha"], temperature=config["temperature"],
        per_agent_views=config["per_agent_views"],
        optimizer=config["optimizer"], lr=config["lr"], optimizer_kwargs=config["optimizer_kwargs"],
        **common,
    )
    _cleanup()  # libère les modèles avant le run suivant (sinon la RAM grossit à chaque run)

    big_opt = big_agent_optimizer(config)
    print(f"\n=== Graine {seed} : grand agent {config['small_arch']} width={big['width']} "
          f"(cross-entropy, {big_opt['optimizer']}) ===")
    single = run_single_reference(data, arch=config["small_arch"], widths=big["width"], seed=seed,
                                  **big_opt, **common)
    _cleanup()

    np.save(os.path.join(probs_dir, f"small_agents_s{seed}.npy"), collab.pop("test_probabilities"))
    np.save(os.path.join(probs_dir, f"big_agent_s{seed}.npy"), single.pop("test_probabilities"))

    accs = collab["accuracy_agents"]
    return {
        "seed": seed,
        "small_agents": {
            "accuracy_consensus": collab["accuracy_consensus"],
            "accuracy_mean_ensemble": collab["accuracy_mean_ensemble"],
            "accuracy_agents": accs,
            "accuracy_agent_median": float(np.median(accs)),
            "accuracy_agent_best": float(np.max(accs)),
            "nll_consensus": collab["nll_consensus"],
            "ece_consensus": collab["ece_consensus"],
            "nll_agents": collab["nll_agents"],
            "ece_agents": collab["ece_agents"],
            "history": collab["history"],
        },
        "big_agent": {
            "accuracy": single["accuracy_reference"],
            "nll": single["nll_reference"],
            "ece": single["ece_reference"],
            "history": single["history"],
        },
    }


def summary_table(runs: list) -> dict:
    """Médiane et quantiles 5 % / 95 % sur les graines des métriques à comparer."""
    rows = {
        "Grand agent seul": ([r["big_agent"]["accuracy"] for r in runs],
                             [r["big_agent"]["nll"] for r in runs],
                             [r["big_agent"]["ece"] for r in runs]),
        "Consensus des petits agents": ([r["small_agents"]["accuracy_consensus"] for r in runs],
                                        [r["small_agents"]["nll_consensus"] for r in runs],
                                        [r["small_agents"]["ece_consensus"] for r in runs]),
        "Moyenne simple des petits": ([r["small_agents"]["accuracy_mean_ensemble"] for r in runs], [], []),
        "Petit agent (médiane)": ([r["small_agents"]["accuracy_agent_median"] for r in runs],
                                  [float(np.median(r["small_agents"]["nll_agents"])) for r in runs],
                                  [float(np.median(r["small_agents"]["ece_agents"])) for r in runs]),
        "Meilleur petit agent": ([r["small_agents"]["accuracy_agent_best"] for r in runs], [], []),
    }
    return {name: {"accuracy": summarize(acc),
                   "nll": summarize(nll) if nll else None,
                   "ece": summarize(ece) if ece else None}
            for name, (acc, nll, ece) in rows.items()}


def print_table(summary: dict, config: dict, sizes: dict):
    n = len(config["seeds"])
    width = 104
    print("\n" + "=" * width)
    print(f"{config['num_agents']} x {config['small_arch']} ({sizes['small_params_total']:,} param.) "
          f"vs grand agent width={sizes['big_width']} ({sizes['big_params']:,} param.) | {n} graine(s)")
    print("médiane [quantile 5 % ; quantile 95 %] sur les graines")
    print("-" * width)
    print(f"{'':30s} {'accuracy':>24s} {'NLL':>24s} {'ECE':>24s}")
    fmt = lambda m: f"{m['median']:.4f} [{m['q05']:.4f};{m['q95']:.4f}]" if m else "-"
    for name, m in summary.items():
        print(f"{name:30s} {fmt(m['accuracy']):>24s} {fmt(m['nll']):>24s} {fmt(m['ece']):>24s}")
    print("=" * width)


def main():
    args = parse_args()
    config = load_config(args)
    check_optimizers(config)
    device = configure_device(force_cpu=args.cpu, precision=config["precision"])

    # Données : séparation train / validation fixe pour toutes les graines
    data = load_dataset(config["dataset"], val_size=config["val_size"], seed=0)
    if config.get("train_subset"):
        data["x_train"] = data["x_train"][:config["train_subset"]]
        data["y_train"] = data["y_train"][:config["train_subset"]]

    # Largeur du grand agent : autant de paramètres que les K petits réunis
    shape, num_classes = data["input_shape"], data["num_classes"]
    small_params = count_trainable_params(config["small_arch"], shape, num_classes)
    target = config["num_agents"] * small_params
    big = match_width(config["small_arch"], target, shape, num_classes)
    sizes = {
        "small_params_each": small_params,
        "small_params_total": target,
        "big_width": big["width"],
        "big_params": big["params"],
        "params_relative_gap": big["relative_gap"],
    }
    print(f"[taille] petit agent : {small_params:,} param. | {config['num_agents']} agents : {target:,} | "
          f"grand agent width={big['width']} : {big['params']:,} ({big['relative_gap']:+.2%})")

    # Dossier de résultats + config enregistrée avant l'entraînement
    run_dir = os.path.join(config["out_dir"], f"{datetime.now():%Y%m%d-%H%M%S}_{config['name']}")
    probs_dir = os.path.join(run_dir, "probs")
    os.makedirs(probs_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**config, "device": device, "model_sizes": sizes}, f, indent=2)

    runs = []
    start = time.time()
    for seed in config["seeds"]:
        runs.append(run_seed(data, config, big, seed, args.verbose, probs_dir))
        summary = summary_table(runs)
        # Réécrit après chaque graine : les résultats partiels sont gardés si le job s'arrête
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump({"summary": summary, "runs": runs, "model_sizes": sizes,
                       "train_time_s": time.time() - start}, f, indent=2)

    print_table(summary, config, sizes)
    print(f"Résultats : {run_dir}")


if __name__ == "__main__":
    main()
