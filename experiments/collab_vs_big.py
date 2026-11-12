"""
Expérience « collaboration vs grand modèle ».

Compare, à nombre de paramètres entraînables égal :
    - K petits agents (même architecture) entraînés ensemble avec la loss multi-agents
      (par défaut global_interaction + simplicial) ;
    - un grand agent seul, de la même architecture élargie (width) pour avoir autant
      de paramètres que les K petits réunis. Sa loss est la cross-entropy
      (reference_loss = "ce") ou la même forme que les petits agents avec I_k = 0
      (reference_loss = "same_as_agents", soit L_k² en multiplicatif).

Chaque run (graine 0 à n_runs - 1) donne 2 tâches indépendantes : les petits agents
(small_r<i>) et le grand agent (big_r<i>). Sur le cluster, les 2 x n_runs tâches tournent
en parallèle dans un job array (sbatch slurm/collab_vs_big.sh).

Exemples (depuis la racine du dépôt) :
    python -m experiments.collab_vs_big                                    # configs/collab_vs_big.json
    python -m experiments.collab_vs_big --dataset fashion_mnist --num-agents 3 --lr 0.01 --epochs 20 --n-runs 10
    python -m experiments.collab_vs_big --cpu --train-subset 2000 --epochs 1 --n-runs 1     # test rapide
    sbatch slurm/collab_vs_big.sh        # cluster (réglages en tête du script)

Sorties : runs/collab_vs_big_<arch>_<dataset>_x<K>[_<tag>]/ (voir experiments/common.py).
"""
import json
import os

import numpy as np

from dml.evaluation.reporting import print_collab_table, summary_collab, write_json
from dml.training.runner import count_trainable_params, match_width, run_multi_experiment, run_single_reference
from dml.utils.config import CollabVsBigConfig
from experiments.common import Experiment, Task, copy_atomic, log


class CollabVsBig(Experiment):
    NAME = "collab_vs_big"
    CONFIG_CLS = CollabVsBigConfig
    DESCRIPTION = __doc__

    def tasks(self, cfg):
        return [Task(f"{part}_r{run}", {"part": part, "run": run, "seed": seed})
                for run, seed in enumerate(cfg.seeds) for part in ("small", "big")]

    def prepare(self, cfg, data):
        # Largeur du grand agent : autant de paramètres que les K petits réunis
        shape, num_classes = data["input_shape"], data["num_classes"]
        small_params = count_trainable_params(cfg.small_arch, shape, num_classes)
        target = cfg.num_agents * small_params
        big = match_width(cfg.small_arch, target, shape, num_classes)
        log.info("[taille] petit agent : %s param. | %d agents : %s | grand agent width=%s : %s (%+.2f%%)",
                 f"{small_params:,}", cfg.num_agents, f"{target:,}", big["width"], f"{big['params']:,}",
                 100 * big["relative_gap"])
        return {"model_sizes": {
            "small_params_each": small_params,
            "small_params_total": target,
            "big_width": big["width"],
            "big_params": big["params"],
            "params_relative_gap": big["relative_gap"],
        }}

    def run_task(self, task, cfg, data, ctx, part_dir, verbose):
        seed, common = task.params["seed"], cfg.train_kwargs(verbose)
        paths = dict(checkpoint_dir=os.path.join(part_dir, "checkpoint"),
                     csv_log=os.path.join(part_dir, "history.csv"))
        if task.params["part"] == "small":
            log.info("%d x %s (%s + %s, alpha=%g), graine %d", cfg.num_agents, cfg.small_arch,
                     cfg.loss_type, cfg.strategy, cfg.alpha, seed)
            res = run_multi_experiment(
                data, archs=cfg.small_arch, num_agents=cfg.num_agents, seed=seed,
                loss_type=cfg.loss_type, strategy=cfg.strategy, alpha=cfg.alpha, temperature=cfg.temperature,
                per_agent_views=cfg.per_agent_views,
                optimizer=cfg.optimizer, lr=cfg.lr, optimizer_kwargs=cfg.optimizer_kwargs,
                **paths, **common,
            )
        else:
            big_opt = cfg.big_agent_optimizer()
            single_loss = "ce" if cfg.reference_loss == "ce" else "form"
            width = ctx["model_sizes"]["big_width"]
            log.info("grand agent %s width=%s (%s, %s), graine %d", cfg.small_arch, width,
                     "cross-entropy" if single_loss == "ce" else f"{cfg.loss_form}, I_k=0", big_opt["optimizer"], seed)
            res = run_single_reference(
                data, arch=cfg.small_arch, widths=width, seed=seed, single_agent_loss=single_loss,
                **big_opt, **paths, **common,
            )
        np.save(os.path.join(part_dir, "probs.npy"), res.pop("test_probabilities"))
        return res

    def merge(self, cfg, run_dir, parts):
        runs = []
        probs_dir = os.path.join(run_dir, "probs")
        os.makedirs(probs_dir, exist_ok=True)
        for run in sorted({p["run"] for p in parts.values()}):
            small, big = parts.get(f"small_r{run}"), parts.get(f"big_r{run}")
            if small is None or big is None:
                continue   # run pas encore complet
            seed = small["seed"]
            for name, part in ((f"small_agents_s{seed}.npy", f"small_r{run}"), (f"big_agent_s{seed}.npy", f"big_r{run}")):
                copy_atomic(os.path.join(run_dir, "parts", part, "probs.npy"), os.path.join(probs_dir, name))
            accs = small["accuracy_agents"]
            runs.append({
                "run": run,
                "seed": seed,
                "small_agents": {
                    "accuracy_consensus": small["accuracy_consensus"],
                    "accuracy_mean_ensemble": small["accuracy_mean_ensemble"],
                    "accuracy_agents": accs,
                    "accuracy_agent_median": float(np.median(accs)),
                    "accuracy_agent_best": float(np.max(accs)),
                    "nll_consensus": small["nll_consensus"],
                    "ece_consensus": small["ece_consensus"],
                    "nll_agents": small["nll_agents"],
                    "ece_agents": small["ece_agents"],
                    "evaluated_on": small.get("evaluated_on"),
                    "best_epoch": small.get("best_epoch"),
                    "history": small["history"],
                },
                "big_agent": {
                    "accuracy": big["accuracy_reference"],
                    "nll": big["nll_reference"],
                    "ece": big["ece_reference"],
                    "evaluated_on": big.get("evaluated_on"),
                    "best_epoch": big.get("best_epoch"),
                    "history": big["history"],
                },
            })
        if not runs:
            log.info("Aucun run complet pour l'instant (%d tâche(s) terminée(s)).", len(parts))
            return
        with open(os.path.join(run_dir, "config.json")) as f:
            sizes = json.load(f).get("model_sizes", {})
        summary = summary_collab(runs)
        write_json(os.path.join(run_dir, "results.json"), {
            "experiment": self.NAME, "n_runs_done": len(runs), "summary": summary, "runs": runs,
            "model_sizes": sizes,
            "train_time_s": sum(p.get("task_time_s", 0.0) for p in parts.values()),   # somme des tâches
        })
        if sizes:
            print_collab_table(summary, cfg, sizes, len(runs))


if __name__ == "__main__":
    CollabVsBig().main()
