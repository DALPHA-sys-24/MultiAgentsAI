"""
Expérience « impact de la collaboration » : balayage du poids alpha.

Les K agents (même architecture) sont entraînés avec la loss multi-agents
L_k * (L_k + alpha * I_k) pour chaque valeur de alpha :
    - alpha = 0 : pas d'interaction, chaque agent est entraîné seul (loss L_k²) ;
      le consensus est un vote majoritaire (strategy = "majority") ;
    - alpha > 0 : collaboration de plus en plus forte ; le consensus utilise la
      stratégie de la config (par défaut "simplicial").

Chaque couple (alpha, run) est une tâche indépendante (a<alpha>_r<run>) : sur le cluster,
les len(alphas) x n_runs tâches tournent en parallèle dans un job array (sbatch slurm/alpha_sweep.sh).

Exemples (depuis la racine du dépôt) :
    python -m experiments.alpha_sweep                                     # configs/alpha_sweep.json
    python -m experiments.alpha_sweep --dataset fashion_mnist --alphas 0 1 10 --n-runs 5
    python -m experiments.alpha_sweep --cpu --train-subset 2000 --epochs 2 --alphas 0 1 --n-runs 1   # test rapide
    sbatch slurm/alpha_sweep.sh          # cluster (réglages en tête du script)

Ajouter des alphas ou des runs à un balayage existant : relancer avec la liste complète ;
les tâches déjà faites sont sautées.

Sorties : runs/alpha_sweep_<arch>_<dataset>_x<K>[_<tag>]/ (voir experiments/common.py).
"""
import json
import os

import numpy as np

from dml.evaluation.metrics import disagreement
from dml.evaluation.reporting import print_sweep_table, summary_by_alpha, write_json
from dml.training.runner import count_trainable_params, run_multi_experiment
from dml.utils.config import STRATEGY_NO_INTERACTION, AlphaSweepConfig
from experiments.common import Experiment, Task, copy_atomic, log


class AlphaSweep(Experiment):
    NAME = "alpha_sweep"
    CONFIG_CLS = AlphaSweepConfig
    DESCRIPTION = __doc__

    def tasks(self, cfg):
        return [Task(f"a{float(alpha):g}_r{run}", {"alpha": float(alpha), "run": run, "seed": seed})
                for run, seed in enumerate(cfg.seeds) for alpha in cfg.alphas]

    def prepare(self, cfg, data):
        params = count_trainable_params(cfg.arch, data["input_shape"], data["num_classes"])
        log.info("[taille] %d x %s : %s param. chacun", cfg.num_agents, cfg.arch, f"{params:,}")
        log.info("[alphas] %s", ", ".join(f"{a:g} ({cfg.strategy_for(a)})" for a in cfg.alphas))
        return {"agent_params": params, "strategy_alpha0": STRATEGY_NO_INTERACTION}

    def run_task(self, task, cfg, data, ctx, part_dir, verbose):
        alpha, seed = task.params["alpha"], task.params["seed"]
        strategy = cfg.strategy_for(alpha)
        log.info("alpha = %g | %d x %s (%s + %s) | graine %d", alpha, cfg.num_agents, cfg.arch,
                 cfg.loss_type, strategy, seed)
        res = run_multi_experiment(
            data, archs=cfg.arch, num_agents=cfg.num_agents, seed=seed,
            loss_type=cfg.loss_type, strategy=strategy, alpha=alpha, temperature=cfg.temperature,
            per_agent_views=cfg.per_agent_views,
            optimizer=cfg.optimizer, lr=cfg.lr, optimizer_kwargs=cfg.optimizer_kwargs,
            checkpoint_dir=os.path.join(part_dir, "checkpoint"), csv_log=os.path.join(part_dir, "history.csv"),
            **cfg.train_kwargs(verbose),
        )
        probs = res.pop("test_probabilities")
        np.save(os.path.join(part_dir, "probs.npy"), probs)
        accs = res["accuracy_agents"]
        return {
            "strategy": strategy,
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
            "evaluated_on": res["evaluated_on"],
            "best_epoch": res["best_epoch"],
            "history": res["history"],
        }

    def merge(self, cfg, run_dir, parts):
        if not parts:
            log.info("Aucune tâche terminée pour l'instant.")
            return
        probs_dir = os.path.join(run_dir, "probs")
        os.makedirs(probs_dir, exist_ok=True)
        runs = []
        for task_id, part in sorted(parts.items(), key=lambda kv: (kv[1]["run"], kv[1]["alpha"])):
            src = os.path.join(run_dir, "parts", task_id, "probs.npy")
            copy_atomic(src, os.path.join(probs_dir, f"agents_alpha{part['alpha']:g}_s{part['seed']}.npy"))
            runs.append({k: v for k, v in part.items() if k not in ("task", "task_time_s")})
        with open(os.path.join(run_dir, "config.json")) as f:
            agent_params = json.load(f).get("agent_params")
        summary = summary_by_alpha(runs)
        write_json(os.path.join(run_dir, "results.json"), {
            "experiment": self.NAME, "summary": summary, "runs": runs, "agent_params": agent_params,
            "train_time_s": sum(p.get("task_time_s", 0.0) for p in parts.values()),   # somme des tâches
        })
        print_sweep_table(summary, cfg.num_agents, cfg.arch, cfg.loss_type)


if __name__ == "__main__":
    AlphaSweep().main()
