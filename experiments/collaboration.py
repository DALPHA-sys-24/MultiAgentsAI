"""
Expérience « collaboration » : K agents entraînés ensemble, et en option un grand modèle seul.

Les K agents (même architecture) sont entraînés avec la loss multi-agents
(par défaut L_k * (L_k + alpha * I_k)) pour chaque valeur de alpha de la liste `alphas` :
    - une seule valeur (--alphas 1) : un run simple ;
    - plusieurs valeurs (--alphas 0 1 10) : un balayage de alpha ;
    - alpha = 0 : agents indépendants : chaque agent est entraîné seul en cross-entropy classique
      (loss "independent", forme additive), et le consensus est un vote majoritaire ("majority") ;
    - alpha > 0 : la loss, la forme et la stratégie de la config (par défaut global_interaction,
      multiplicative, simplicial).

Avec --big-model true, on entraîne aussi un grand modèle seul : la même architecture élargie
(width) pour avoir autant de paramètres que les K agents réunis. Sa loss est la cross-entropy
(reference_loss = "ce") ou la même forme que les agents avec I_k = 0 (reference_loss = "same_as_agents").
Il ne dépend pas de alpha : un seul grand modèle par graine, comparé à chaque alpha.

Tâches indépendantes (une par dossier tasks/<tâche>/), pour chaque graine 0 à n_runs - 1 :
    alpha<alpha>_seed<graine>     les K agents pour cet alpha
    big_model_seed<graine>        le grand modèle (si big_model)
Sur le cluster, toutes les tâches tournent en parallèle dans un job array (sbatch slurm/collaboration.sh).
Relancer avec plus d'alphas, plus de runs ou --big-model true ajoute les tâches manquantes au même dossier.

Exemples (depuis la racine du dépôt) :
    python -m experiments.collaboration                                    # configs/collaboration.json
    python -m experiments.collaboration --dataset fashion_mnist --alphas 1 --num-agents 3 --n-runs 5
    python -m experiments.collaboration --alphas 0 1 10 --big-model true --n-runs 5
    python -m experiments.collaboration --cpu --train-subset 2000 --epochs 1 --alphas 1 --n-runs 1   # test rapide
    sbatch slurm/collaboration.sh        # cluster (réglages en tête du script)

Sorties : results/collaboration/<dataset>/<arch>_<K>agents[_<tag>]/ (voir experiments/common.py).
"""
import json
import os

import numpy as np

from collaborative_learning.evaluation.metrics import disagreement
from collaborative_learning.evaluation.reporting import (print_collab_table, print_sweep_table, summarize, summary_by_alpha,
                                      summary_collab, write_json)
from collaborative_learning.training.runner import count_trainable_params, match_width, run_multi_experiment, run_single_reference
from collaborative_learning.utils.config import CollaborationConfig
from experiments.common import PROBA_DIR, PROBA_FILE, TASKS_DIR, Experiment, Task, copy_atomic, log

AGENTS, BIG_MODEL = "agents", "big_model"


class Collaboration(Experiment):
    NAME = "collaboration"
    CONFIG_CLS = CollaborationConfig
    DESCRIPTION = __doc__

    def tasks(self, cfg):
        tasks = []
        for run, seed in enumerate(cfg.seeds):
            for alpha in cfg.alphas:
                tasks.append(Task(f"alpha{float(alpha):g}_seed{seed}",
                                  {"model": AGENTS, "alpha": float(alpha), "run": run, "seed": seed}))
            if cfg.big_model:
                tasks.append(Task(f"big_model_seed{seed}", {"model": BIG_MODEL, "run": run, "seed": seed}))
        return tasks

    def prepare(self, cfg, data):
        shape, num_classes = data["input_shape"], data["num_classes"]
        params = count_trainable_params(cfg.arch, shape, num_classes)
        log.info("[taille] %d x %s : %s param. chacun", cfg.num_agents, cfg.arch, f"{params:,}")
        for a in cfg.alphas:
            s = cfg.setup_for(a)
            log.info("[alpha = %g] loss %s (%s), consensus %s", a, s["loss_type"], s["loss_form"], s["strategy"])
        ctx = {"agent_params": params, "setup_alpha0": cfg.setup_for(0.0)}
        if cfg.big_model:
            # Largeur du grand modèle : autant de paramètres que les K agents réunis
            target = cfg.num_agents * params
            big = match_width(cfg.arch, target, shape, num_classes)
            log.info("[taille] %d agents : %s param. | grand modèle width=%s : %s (%+.2f%%)",
                     cfg.num_agents, f"{target:,}", big["width"], f"{big['params']:,}", 100 * big["relative_gap"])
            ctx["model_sizes"] = {
                "small_params_each": params,
                "small_params_total": target,
                "big_width": big["width"],
                "big_params": big["params"],
                "params_relative_gap": big["relative_gap"],
            }
        return ctx

    def run_task(self, task, cfg, data, ctx, task_dir, verbose):
        seed = task.params["seed"]
        paths = dict(checkpoint_dir=os.path.join(task_dir, "checkpoint"),
                     csv_log=os.path.join(task_dir, "history.csv"))
        if task.params["model"] == BIG_MODEL:
            return self._run_big_model(cfg, data, ctx, seed, paths, task_dir, verbose)

        alpha = task.params["alpha"]
        setup = cfg.setup_for(alpha)          # alpha = 0 : independent (CE) + majority
        strategy = setup["strategy"]
        log.info("alpha = %g | %d x %s (%s %s + %s) | graine %d", alpha, cfg.num_agents, cfg.arch,
                 setup["loss_type"], setup["loss_form"], strategy, seed)
        train = {**cfg.train_kwargs(verbose), "form": setup["loss_form"]}
        res = run_multi_experiment(
            data, archs=cfg.arch, num_agents=cfg.num_agents, seed=seed,
            loss_type=setup["loss_type"], strategy=strategy, alpha=alpha, temperature=cfg.temperature,
            per_agent_views=cfg.per_agent_views,
            optimizer=cfg.optimizer, lr=cfg.lr, optimizer_kwargs=cfg.optimizer_kwargs,
            **paths, **train,
        )
        probs = res.pop("test_probabilities")
        np.save(os.path.join(task_dir, PROBA_FILE), probs)
        accs = res["accuracy_agents"]
        return {
            "loss_type": setup["loss_type"],
            "loss_form": setup["loss_form"],
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

    def _run_big_model(self, cfg, data, ctx, seed, paths, task_dir, verbose):
        big_opt = cfg.big_model_optimizer()
        single_loss = "ce" if cfg.reference_loss == "ce" else "form"
        width = ctx["model_sizes"]["big_width"]
        log.info("grand modèle %s width=%s (%s, %s), graine %d", cfg.arch, width,
                 "cross-entropy" if single_loss == "ce" else f"{cfg.loss_form}, I_k=0", big_opt["optimizer"], seed)
        res = run_single_reference(
            data, arch=cfg.arch, widths=width, seed=seed, single_agent_loss=single_loss,
            **big_opt, **paths, **cfg.train_kwargs(verbose),
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
        for task_id in tasks:     # alpha<a>_seed<s>.npy, big_model_seed<s>.npy
            copy_atomic(os.path.join(run_dir, TASKS_DIR, task_id, PROBA_FILE), os.path.join(proba_dir, f"{task_id}.npy"))

        drop = ("task", "task_time_s", "model")
        runs = sorted(({k: v for k, v in t.items() if k not in drop} for t in tasks.values() if t["model"] == AGENTS),
                      key=lambda r: (r["run"], r["alpha"]))
        big_runs = sorted(({k: v for k, v in t.items() if k not in drop} for t in tasks.values()
                           if t["model"] == BIG_MODEL), key=lambda r: r["run"])
        with open(os.path.join(run_dir, "config.json")) as f:
            saved = json.load(f)

        summary = summary_by_alpha(runs) if runs else {}
        out = {"experiment": self.NAME, "summary": summary, "runs": runs, "agent_params": saved.get("agent_params"),
               "train_time_s": sum(t.get("task_time_s", 0.0) for t in tasks.values())}   # somme des tâches
        if big_runs:
            big_by_seed = {r["seed"]: r for r in big_runs}
            out["big_model"] = {
                "summary": {m: summarize([r[m] for r in big_runs]) for m in ("accuracy", "nll", "ece")},
                "runs": big_runs,
            }
            # Pour chaque alpha : agents et grand modèle de la même graine, côte à côte
            out["vs_big_model"] = {}
            for key, s in summary.items():
                paired = [{"small_agents": r, "big_model": big_by_seed[r["seed"]]}
                          for r in runs if r["alpha"] == s["alpha"] and r["seed"] in big_by_seed]
                if paired:
                    out["vs_big_model"][key] = {"n_seeds": len(paired), **summary_collab(paired)}
            out["model_sizes"] = saved.get("model_sizes")
        write_json(os.path.join(run_dir, "results.json"), out)

        if summary:
            print_sweep_table(summary, cfg.num_agents, cfg.arch, cfg.loss_type)
        for key, s in out.get("vs_big_model", {}).items():
            if out.get("model_sizes"):
                rows = {k: v for k, v in s.items() if k != "n_seeds"}
                print_collab_table(rows, cfg, out["model_sizes"], s["n_seeds"], alpha=float(key))


if __name__ == "__main__":
    Collaboration().main()
