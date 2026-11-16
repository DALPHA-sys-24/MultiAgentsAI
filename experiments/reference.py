"""
Expérience « reference » : le grand modèle seul, référence des K agents.

La même architecture que les agents, élargie (width) pour avoir autant de paramètres entraînables
que `num_agents` agents réunis. Sa loss est la cross-entropy (loss = "ce") ou la même forme que les
agents avec I_k = 0 (loss = "same_as_agents", L_k² en multiplicatif). Il ne dépend ni de alpha ni de
la loss des agents : entraîné une fois, il est réutilisé par toutes les campagnes qui ont le même
dataset, la même architecture, le même K et les mêmes réglages d'entraînement.

Tâches : seed<graine> pour les graines 0 à n_runs - 1.

Exemples (depuis la racine du dépôt) :
    python -m experiments.reference                                  # configs/reference.yaml
    python -m experiments.reference --dataset fashion_mnist --num-agents 3 --lr 0.05 --n-runs 5

Sorties : results/reference/<dataset>/<arch>_x<K>params_<optimiseur>_lr<lr>[_<tag>]/
    config.json, results.json (accuracy, NLL, ECE : médiane [q05 ; q95] et détail par graine),
    test_probabilities/seed<s>.npy, tasks/seed<s>/ (result.json, history.csv, checkpoint/)
"""
import json
import os

import numpy as np

from collaborative_learning.evaluation.reporting import fmt_summary, summarize, write_json
from collaborative_learning.training.runner import count_trainable_params, match_width, run_single_reference
from collaborative_learning.utils.config import ReferenceConfig
from experiments.common import PROBA_DIR, PROBA_FILE, TASKS_DIR, Experiment, Task, copy_atomic, log

METRICS = ("accuracy", "nll", "ece")


class Reference(Experiment):
    NAME = "reference"
    CONFIG_CLS = ReferenceConfig
    DESCRIPTION = __doc__

    def tasks(self, cfg):
        return [Task(f"seed{seed}", {"run": run, "seed": seed}) for run, seed in enumerate(cfg.seeds)]

    def prepare(self, cfg, data):
        shape, num_classes = data["input_shape"], data["num_classes"]
        small = count_trainable_params(cfg.arch, shape, num_classes)
        target = cfg.num_agents * small
        if cfg.width is not None:
            big = {"width": cfg.width, "params": count_trainable_params(cfg.arch, shape, num_classes, cfg.width)}
            big["relative_gap"] = (big["params"] - target) / target
        else:
            big = match_width(cfg.arch, target, shape, num_classes)
        log.info("[taille] %d x %s : %s param. | grand modèle width=%s : %s (%+.2f%%)", cfg.num_agents, cfg.arch,
                 f"{target:,}", big["width"], f"{big['params']:,}", 100 * big["relative_gap"])
        return {"model_sizes": {
            "small_params_each": small,
            "small_params_total": target,
            "big_width": big["width"],
            "big_params": big["params"],
            "params_relative_gap": big["relative_gap"],
        }}

    def run_task(self, task, cfg, data, ctx, task_dir, verbose):
        seed = task.params["seed"]
        width = ctx["model_sizes"]["big_width"]
        single_loss = "ce" if cfg.loss == "ce" else "form"
        log.info("grand modèle %s width=%s (%s, %s lr=%g), graine %d", cfg.arch, width,
                 "cross-entropy" if single_loss == "ce" else f"{cfg.loss_form}, I_k=0", cfg.optimizer, cfg.lr, seed)
        train = {**cfg.train_kwargs(verbose), "form": cfg.loss_form}
        res = run_single_reference(
            data, arch=cfg.arch, widths=width, seed=seed, single_agent_loss=single_loss,
            optimizer=cfg.optimizer, lr=cfg.lr, optimizer_kwargs=cfg.optimizer_kwargs,
            checkpoint_dir=os.path.join(task_dir, "checkpoint"), csv_log=os.path.join(task_dir, "history.csv"),
            **train,
        )
        np.save(os.path.join(task_dir, PROBA_FILE), res.pop("test_probabilities"))
        return {
            "accuracy": res["accuracy_reference"],
            "nll": res["nll_reference"],
            "ece": res["ece_reference"],
            "evaluated_on": res.get("evaluated_on"),
            "best_epoch": res.get("best_epoch"),
            "history": res["history"],
        }

    def merge(self, cfg, run_dir, tasks):
        if not tasks:
            log.info("Aucune tâche terminée pour l'instant.")
            return
        proba_dir = os.path.join(run_dir, PROBA_DIR)
        os.makedirs(proba_dir, exist_ok=True)
        for task_id in tasks:
            copy_atomic(os.path.join(run_dir, TASKS_DIR, task_id, PROBA_FILE), os.path.join(proba_dir, f"{task_id}.npy"))
        drop = ("task", "task_time_s", "model")
        runs = sorted(({k: v for k, v in t.items() if k not in drop} for t in tasks.values()), key=lambda r: r["seed"])
        with open(os.path.join(run_dir, "config.json")) as f:
            saved = json.load(f)
        summary = {m: summarize([r[m] for r in runs]) for m in METRICS}
        write_json(os.path.join(run_dir, "results.json"), {
            "experiment": self.NAME, "summary": summary, "runs": runs, "model_sizes": saved.get("model_sizes"),
            "train_time_s": sum(t.get("task_time_s", 0.0) for t in tasks.values()),
        })
        log.info("Grand modèle seul (%d graine(s)) : accuracy %s | NLL %s | ECE %s", len(runs),
                 *(fmt_summary(summary[m]) for m in METRICS))


if __name__ == "__main__":
    Reference().main()
