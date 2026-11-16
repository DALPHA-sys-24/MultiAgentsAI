"""
Expérience « collaboration » : K agents entraînés ensemble.

Les K agents (même architecture) sont entraînés avec la loss multi-agents
(par défaut L_k * (L_k + alpha * I_k)) pour chaque valeur de alpha de la liste `alphas` :
    - une seule valeur (--alphas 1) : un run simple ;
    - plusieurs valeurs (--alphas 0 1 10) : un balayage de alpha ;
    - alpha = 0 : agents indépendants : chaque agent est entraîné seul en cross-entropy classique
      (loss "independent", forme additive), et le consensus est un vote majoritaire ("majority") ;
    - alpha > 0 : la loss, la forme et la stratégie de la config (par défaut global_interaction,
      multiplicative, simplicial).

Le grand modèle seul (même nombre de paramètres que les K agents) est une expérience à part :
experiments/reference.py. Pour plusieurs méthodes, alphas, lr... et le grand modèle d'un coup,
utiliser une campagne (experiments/campaign.py, configs/campaigns/).

Tâches indépendantes (une par dossier tasks/<tâche>/), pour chaque graine 0 à n_runs - 1 :
    alpha<alpha>_seed<graine>     les K agents pour cet alpha
Relancer avec plus d'alphas ou plus de runs ajoute les tâches manquantes au même dossier.

Exemples (depuis la racine du dépôt) :
    python -m experiments.collaboration                                    # configs/collaboration.yaml
    python -m experiments.collaboration --dataset fashion_mnist --alphas 1 --num-agents 3 --n-runs 5
    python -m experiments.collaboration --config configs/methods/dml.yaml --alphas 0 1
    python -m experiments.collaboration --cpu --train-subset 2000 --epochs 1 --alphas 1 --n-runs 1   # test rapide

Sorties : results/collaboration/<dataset>/<arch>_<K>agents/<méthode>[_<tag>]/ (voir experiments/common.py).
"""
import json
import os

import numpy as np

from collaborative_learning.evaluation.metrics import disagreement
from collaborative_learning.evaluation.reporting import print_sweep_table, summary_by_alpha, write_json
from collaborative_learning.training.runner import count_trainable_params, run_multi_experiment
from collaborative_learning.utils.config import CollaborationConfig
from experiments.common import PROBA_DIR, PROBA_FILE, TASKS_DIR, Experiment, Task, copy_atomic, log


class Collaboration(Experiment):
    NAME = "collaboration"
    CONFIG_CLS = CollaborationConfig
    DESCRIPTION = __doc__

    def tasks(self, cfg):
        tasks = []
        for run, seed in enumerate(cfg.seeds):
            for alpha in cfg.alphas:
                tasks.append(Task(f"alpha{float(alpha):g}_seed{seed}", {"alpha": float(alpha), "run": run, "seed": seed}))
        return tasks

    def prepare(self, cfg, data):
        shape, num_classes = data["input_shape"], data["num_classes"]
        params = count_trainable_params(cfg.arch, shape, num_classes)
        log.info("[taille] %d x %s : %s param. chacun", cfg.num_agents, cfg.arch, f"{params:,}")
        for a in cfg.alphas:
            s = cfg.setup_for(a)
            log.info("[alpha = %g] loss %s (%s), consensus %s", a, s["loss_type"], s["loss_form"], s["strategy"])
        return {"agent_params": params}

    def run_task(self, task, cfg, data, ctx, task_dir, verbose):
        seed = task.params["seed"]
        paths = dict(checkpoint_dir=os.path.join(task_dir, "checkpoint"),
                     csv_log=os.path.join(task_dir, "history.csv"))
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

    def merge(self, cfg, run_dir, tasks):
        if not tasks:
            log.info("Aucune tâche terminée pour l'instant.")
            return
        proba_dir = os.path.join(run_dir, PROBA_DIR)
        os.makedirs(proba_dir, exist_ok=True)
        for task_id in tasks:     # alpha<a>_seed<s>.npy
            copy_atomic(os.path.join(run_dir, TASKS_DIR, task_id, PROBA_FILE), os.path.join(proba_dir, f"{task_id}.npy"))

        drop = ("task", "task_time_s", "model")
        runs = sorted(({k: v for k, v in t.items() if k not in drop} for t in tasks.values() if "alpha" in t),
                      key=lambda r: (r["run"], r["alpha"]))
        with open(os.path.join(run_dir, "config.json")) as f:
            saved = json.load(f)
        summary = summary_by_alpha(runs) if runs else {}
        out = {"experiment": self.NAME, "method": cfg.method_name, "summary": summary, "runs": runs,
               "agent_params": saved.get("agent_params"),
               "train_time_s": sum(t.get("task_time_s", 0.0) for t in tasks.values())}   # somme des tâches
        write_json(os.path.join(run_dir, "results.json"), out)
        if summary:
            print_sweep_table(summary, cfg.num_agents, cfg.arch, cfg.method_name)


if __name__ == "__main__":
    Collaboration().main()
