"""
Expérience « impact de la collaboration » : balayage du poids alpha.

Les K agents (même architecture) sont entraînés avec la loss multi-agents
L_k * (L_k + alpha * I_k) pour chaque valeur de alpha :
    - alpha = 0 : pas d'interaction, chaque agent est entraîné seul (loss L_k²) ;
      le consensus est un vote majoritaire (strategy = "majority") ;
    - alpha > 0 : collaboration de plus en plus forte ; le consensus utilise la
      stratégie du fichier de config (par défaut "simplicial").

Les réglages sont lus dans un fichier de config JSON ; chaque option de la ligne de
commande le surcharge.

Exemples :
    python run_alpha_sweep.py
    python run_alpha_sweep.py --config configs/alpha_sweep.json --dataset fashion_mnist --epochs 50
    python run_alpha_sweep.py --alphas 0 1 10 --seeds 0 --cpu --train-subset 2000 --epochs 2 --warmup-epochs 1
    python run_alpha_sweep.py --resume results/20261019-114342_alpha_sweep   # reprend un sweep interrompu

Chaque lancement crée results/<date>_<nom>/ contenant :
    config.json     config utilisée + nombre de paramètres des agents
    results.json    métriques par (alpha, graine) et résumé par alpha (médiane, quantiles 5 % / 95 %)
    probs/          probabilités de test des agents (.npy), une par (alpha, graine)
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
from run_experiment import parse_jit, summarize, train_kwargs
from tools import _cleanup, count_trainable_params, run_multi_experiment

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "alpha_sweep.json")

# Stratégie de consensus imposée quand alpha = 0 (agents indépendants)
STRATEGY_NO_INTERACTION = "majority"

# Valeurs par défaut, utilisées pour toute clé absente du fichier de config
DEFAULTS = {
    "name": "alpha_sweep",
    "dataset": "cifar10",
    "arch": "resnet20",
    "num_agents": 5,
    "loss_type": "global_interaction",
    "strategy": "simplicial",       # utilisée pour alpha > 0
    "alphas": [0, 0.5, 1, 10, 50, 100],
    "temperature": 1.0,
    "epochs": 200,
    "batch_size": 128,
    "optimizer": "sgd",             # un nom, ou une liste d'un nom par agent
    "lr": 0.01,                     # un lr, ou une liste d'un lr par agent (0.1 fait diverger la loss multiplicative)
    "weight_decay": 5e-4,
    "optimizer_kwargs": None,
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

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Fichier de config JSON (défaut : configs/alpha_sweep.json "
                                     "à côté du script s'il existe, sinon valeurs par défaut).")
    # Surcharges optionnelles (None = valeur du fichier de config)
    p.add_argument("--name")
    p.add_argument("--dataset", choices=["fashion_mnist", "cifar10", "cifar100"])
    p.add_argument("--arch", help="Architecture des agents (ex. resnet20).")
    p.add_argument("--num-agents", type=int)
    p.add_argument("--loss-type", choices=list(LOSS_FUNCTIONS))
    p.add_argument("--strategy", choices=list(STRATEGIES), help="Stratégie utilisée pour alpha > 0.")
    p.add_argument("--alphas", type=float, nargs="+", help="Valeurs de alpha à tester.")
    p.add_argument("--temperature", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--optimizer", nargs="+", choices=available_optimizers(),
                   help="Un nom pour tous les agents, ou un nom par agent.")
    p.add_argument("--lr", type=float, nargs="+", help="Un lr pour tous, ou un par agent.")
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
    p.add_argument("--resume", metavar="DOSSIER",
                   help="Reprend un sweep interrompu dans ce dossier de résultats, avec sa config ; "
                        "les couples (alpha, graine) déjà faits sont sautés.")
    p.add_argument("--cpu", action="store_true", help="Force le CPU même si un GPU est disponible.")
    p.add_argument("--verbose", type=int, default=2)
    return p.parse_args()


def load_config(args) -> dict:
    """Valeurs par défaut, puis fichier de config, puis surcharges de la ligne de commande."""
    config = dict(DEFAULTS)
    if args.resume:
        # Reprise : la config enregistrée du sweep fait foi, sans autres surcharges
        path = os.path.join(args.resume, "config.json")
        if not os.path.exists(path):
            raise SystemExit(f"Pas de config.json dans {os.path.abspath(args.resume)}")
        with open(path) as f:
            saved = json.load(f)
        config.update({k: v for k, v in saved.items() if k in DEFAULTS})
        print(f"[reprise] {os.path.abspath(args.resume)}")
        return config
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
    names = config["optimizer"] if isinstance(config["optimizer"], list) else [config["optimizer"]]
    unknown = [n for n in names if n.lower() not in available_optimizers()]
    if unknown:
        raise SystemExit(f"Optimiseur(s) inconnu(s) : {unknown}. Choix : {available_optimizers()}")
    return config


def strategy_for(alpha: float, config: dict) -> str:
    """Vote majoritaire sans interaction (alpha = 0), sinon la stratégie du config."""
    return STRATEGY_NO_INTERACTION if alpha == 0 else config["strategy"]


# =============================================================================
# EXPÉRIENCE
# =============================================================================

def disagreement(probs: np.ndarray) -> float:
    """Proportion moyenne d'exemples où deux agents prédisent une classe différente."""
    preds = probs.argmax(-1)                     # (N, K)
    k = preds.shape[1]
    pairs = [(preds[:, i] != preds[:, j]).mean() for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(pairs)) if pairs else 0.0


def run_one(data: dict, config: dict, alpha: float, seed: int, verbose: int, probs_dir: str) -> dict:
    """Entraîne les K agents pour un couple (alpha, graine)."""
    strategy = strategy_for(alpha, config)
    print(f"\n=== Graine {seed} | alpha = {alpha:g} | {config['num_agents']} x {config['arch']} "
          f"({config['loss_type']} + {strategy}) ===")
    res = run_multi_experiment(
        data, archs=config["arch"], num_agents=config["num_agents"], seed=seed,
        loss_type=config["loss_type"], strategy=strategy,
        alpha=alpha, temperature=config["temperature"],
        per_agent_views=config["per_agent_views"],
        optimizer=config["optimizer"], lr=config["lr"], optimizer_kwargs=config["optimizer_kwargs"],
        **train_kwargs(config, verbose),
    )
    probs = res.pop("test_probabilities")
    np.save(os.path.join(probs_dir, f"agents_alpha{alpha:g}_s{seed}.npy"), probs)

    accs = res["accuracy_agents"]
    return {
        "alpha": alpha,
        "strategy": strategy,
        "seed": seed,
        "accuracy_consensus": res["accuracy_consensus"],
        "accuracy_mean_ensemble": res["accuracy_mean_ensemble"],
        "accuracy_agents": accs,
        "accuracy_agent_median": float(np.median(accs)),
        "accuracy_agent_best": float(np.max(accs)),
        "nll_consensus": res["nll_consensus"],     # None pour un consensus à labels durs (majority)
        "ece_consensus": res["ece_consensus"],
        "nll_agents": res["nll_agents"],
        "ece_agents": res["ece_agents"],
        "disagreement": disagreement(probs),
        "history": res["history"],
    }


METRICS = {
    "Consensus": "accuracy_consensus",
    "Moyenne simple": "accuracy_mean_ensemble",
    "Agent (médiane)": "accuracy_agent_median",
    "Meilleur agent": "accuracy_agent_best",
    "Désaccord": "disagreement",
}


def summary_by_alpha(runs: list) -> dict:
    """Médiane et quantiles 5 % / 95 % sur les graines, pour chaque alpha."""
    out = {}
    for alpha in sorted({r["alpha"] for r in runs}):
        rs = [r for r in runs if r["alpha"] == alpha]
        out[f"{alpha:g}"] = {
            "alpha": alpha,
            "strategy": rs[0]["strategy"],
            "n_seeds": len(rs),
            **{key: summarize([r[key] for r in rs]) for key in METRICS.values()},
            "nll_agent_median": summarize([float(np.median(r["nll_agents"])) for r in rs]),
            "ece_agent_median": summarize([float(np.median(r["ece_agents"])) for r in rs]),
        }
    return out


def print_table(summary: dict, config: dict):
    width = 18 + 25 * len(METRICS)
    print("\n" + "=" * width)
    print(f"{config['num_agents']} x {config['arch']} | {config['loss_type']} | "
          f"médiane [quantile 5 % ; quantile 95 %] sur les graines")
    print("-" * width)
    print(f"{'alpha (strat.)':18s}" + "".join(f"{name:>25s}" for name in METRICS))
    fmt = lambda m: f"{m['median']:.4f} [{m['q05']:.4f};{m['q95']:.4f}]"
    for s in summary.values():
        label = f"{s['alpha']:g} ({s['strategy'][:4]})"
        print(f"{label:18s}" + "".join(f"{fmt(s[key]):>25s}" for key in METRICS.values()))
    print("=" * width)


def main():
    args = parse_args()
    config = load_config(args)
    device = configure_device(force_cpu=args.cpu, precision=config["precision"])

    # Données : séparation train / validation fixe pour toutes les graines
    data = load_dataset(config["dataset"], val_size=config["val_size"], seed=0)
    if config.get("train_subset"):
        data["x_train"] = data["x_train"][:config["train_subset"]]
        data["y_train"] = data["y_train"][:config["train_subset"]]

    params = count_trainable_params(config["arch"], data["input_shape"], data["num_classes"])
    print(f"[taille] {config['num_agents']} x {config['arch']} : {params:,} param. chacun")
    print("[alphas] " + ", ".join(f"{a:g} ({strategy_for(a, config)})" for a in config["alphas"]))

    runs, elapsed = [], 0.0
    if args.resume:
        # Reprise : on garde les runs déjà terminés du dossier existant
        run_dir = args.resume
        results_path = os.path.join(run_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                previous = json.load(f)
            runs, elapsed = previous["runs"], previous.get("train_time_s", 0.0)
        print(f"[reprise] {len(runs)} run(s) déjà terminé(s)")
    else:
        # Dossier de résultats + config enregistrée avant l'entraînement
        run_dir = os.path.join(config["out_dir"], f"{datetime.now():%Y%m%d-%H%M%S}_{config['name']}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({**config, "device": device, "agent_params": params,
                       "strategy_alpha0": STRATEGY_NO_INTERACTION}, f, indent=2)
    probs_dir = os.path.join(run_dir, "probs")
    os.makedirs(probs_dir, exist_ok=True)

    done = {(float(r["alpha"]), r["seed"]) for r in runs}
    start = time.time() - elapsed
    for seed in config["seeds"]:
        for alpha in config["alphas"]:
            if (float(alpha), seed) in done:
                continue
            runs.append(run_one(data, config, float(alpha), seed, args.verbose, probs_dir))
            _cleanup()  # libère les modèles du run (sinon la RAM grossit jusqu'au « Killed »)
            # Réécrit après chaque run : les résultats partiels sont gardés si le job s'arrête
            with open(os.path.join(run_dir, "results.json"), "w") as f:
                json.dump({"experiment": "alpha_sweep", "summary": summary_by_alpha(runs), "runs": runs,
                           "agent_params": params, "train_time_s": time.time() - start}, f, indent=2)

    summary = summary_by_alpha(runs)

    print_table(summary, config)
    print(f"Résultats : {run_dir}")


if __name__ == "__main__":
    main()
