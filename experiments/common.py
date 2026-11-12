"""
Socle commun des expériences : dossier de run, découpage en tâches, fusion des résultats.

Une expérience = une sous-classe de Experiment qui définit :
    tasks(cfg)                        la liste des tâches indépendantes (ex. un run = une graine)
    prepare(cfg, data)                calculs communs à toutes les tâches (ex. largeur du grand agent)
    run_task(task, cfg, data, ctx, part_dir, verbose)   une tâche ; renvoie un dict de résultats
    merge(cfg, run_dir, parts)        rassemble les tâches terminées dans results.json

Organisation d'un dossier de run :

    runs/<expérience>_<arch>_<dataset>_x<K>[_<tag>]/
    ├── config.json          réglages, tailles des modèles, environnement (commit Git, versions)
    ├── results.json         résumé sur les runs (médiane, quantiles 5 % / 95 %) et détail par run
    ├── probs/               probabilités de test de chaque run (.npy), pour les notebooks
    ├── parts/<tâche>/       une tâche : result.json, probs.npy, history.csv, checkpoint/
    ├── logs/                journal de chaque processus

En local, toutes les tâches tournent l'une après l'autre. Sur le cluster, chaque tâche est
une tâche du job array (--task-index), et toutes écrivent dans le même dossier. Une tâche
déjà terminée est sautée ; une tâche interrompue reprend à sa dernière époque.
"""
import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import tensorflow as tf

from dml.data import download_dataset, load_dataset
from dml.evaluation.reporting import write_json
from dml.training.runner import cleanup
from dml.utils.config import add_config_arguments, environment_info, load_config, overrides_from_args
from dml.utils.device import configure_device, default_device
from dml.utils.log import setup_logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = logging.getLogger("dml")


def copy_atomic(src: str, dst: str) -> None:
    """Copie src vers dst ; plusieurs tâches peuvent copier le même fichier en même temps."""
    if not os.path.exists(src):
        return
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return   # déjà à jour
    tmp = f"{dst}.{os.getpid()}.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


@dataclass
class Task:
    """Une unité de travail indépendante (un run, une valeur de alpha, un worker de recherche...)."""
    id: str                                   # nom du dossier parts/<id>
    params: Dict[str, Any] = field(default_factory=dict)


class Experiment:
    NAME = "experiment"
    CONFIG_CLS = None
    DESCRIPTION = ""
    USES_PARTS = True          # False : les tâches n'écrivent pas dans parts/ (ex. workers Optuna)

    # --- à définir dans chaque expérience ------------------------------------
    def tasks(self, cfg) -> List[Task]:
        raise NotImplementedError

    def prepare(self, cfg, data) -> Dict[str, Any]:
        """Infos communes aux tâches, enregistrées dans config.json."""
        return {}

    def run_task(self, task: Task, cfg, data, ctx: Dict[str, Any], part_dir: str, verbose: int) -> Dict[str, Any]:
        raise NotImplementedError

    def merge(self, cfg, run_dir: str, parts: Dict[str, Dict[str, Any]]) -> None:
        raise NotImplementedError

    def is_done(self, part_dir: str) -> bool:
        return os.path.exists(os.path.join(part_dir, "result.json"))

    # --- ligne de commande ---------------------------------------------------
    def parse_args(self, argv=None):
        p = argparse.ArgumentParser(description=self.DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("--config", help=f"Fichier de config JSON (défaut : configs/{self.NAME}.json).")
        p.add_argument("--run-dir", metavar="DOSSIER",
                       help="Dossier du run (défaut : <out_dir>/<expérience>_<arch>_<dataset>_x<K>[_<tag>]).")
        p.add_argument("--task-index", type=int, metavar="I",
                       help="N'exécute que la tâche I (job array SLURM : $SLURM_ARRAY_TASK_ID).")
        p.add_argument("--list-tasks", action="store_true",
                       help="Affiche « TASKS <nombre> <dossier> » puis s'arrête .")
        p.add_argument("--merge-only", action="store_true", help="Rassemble les tâches terminées, sans rien entraîner.")
        p.add_argument("--cpu", action="store_true", help="Force le CPU même si un GPU est disponible.")
        p.add_argument("--verbose", type=int, default=2)
        add_config_arguments(p, self.CONFIG_CLS)
        return p.parse_args(argv)

    def load_config(self, args):
        path = args.config or os.path.join(ROOT, "configs", f"{self.NAME}.json")
        if not os.path.exists(path):
            path = None
        return load_config(self.CONFIG_CLS, path, overrides_from_args(args, self.CONFIG_CLS)), path

    # --- dossier de run ------------------------------------------------------
    @staticmethod
    def check_run_dir(cfg, run_dir: str) -> None:
        """Refuse de mélanger deux expériences différentes dans le même dossier."""
        path = os.path.join(run_dir, "config.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            saved = json.load(f)
        current = cfg.signature()
        diffs = [f"{k} : {saved[k]!r} (dossier) -> {v!r} (maintenant)"
                 for k, v in current.items() if k in saved and saved[k] != v]
        if diffs:
            raise SystemExit(
                f"Le dossier {run_dir} contient déjà une expérience avec d'autres réglages :\n  - "
                + "\n  - ".join(diffs)
                + "\nPour lancer cette variante à part, ajoute un suffixe : --tag <nom> "
                  "(ou \"tag\" dans la config), ou choisis un autre --run-dir.")

    @staticmethod
    def write_config(cfg, run_dir: str, **extra) -> None:
        """config.json : réglages actuels ; n_runs / alphas étendus si le dossier en avait plus."""
        path = os.path.join(run_dir, "config.json")
        payload = {**cfg.to_dict(), **extra}
        if os.path.exists(path):
            with open(path) as f:
                saved = json.load(f)
            if isinstance(saved.get("n_runs"), int):
                payload["n_runs"] = max(saved["n_runs"], payload.get("n_runs") or 0)
            if isinstance(saved.get("alphas"), list) and "alphas" in payload:
                payload["alphas"] = sorted(set(map(float, saved["alphas"])) | set(map(float, payload["alphas"])))
        write_json(path, payload)

    @staticmethod
    def read_parts(run_dir: str) -> Dict[str, Dict[str, Any]]:
        parts = {}
        parts_dir = os.path.join(run_dir, "parts")
        if not os.path.isdir(parts_dir):
            return parts
        for name in sorted(os.listdir(parts_dir)):
            path = os.path.join(parts_dir, name, "result.json")
            if os.path.exists(path):
                with open(path) as f:
                    parts[name] = json.load(f)
        return parts

    # --- exécution -----------------------------------------------------------
    def main(self, argv=None):
        args = self.parse_args(argv)
        cfg, path = self.load_config(args)
        tasks = self.tasks(cfg)
        run_dir = args.run_dir or cfg.run_dir()
        self.run_dir = run_dir

        if args.list_tasks:
            # avant de soumettre (nœud de connexion) : réglages compatibles avec le dossier, et
            # données téléchargées une fois (les nœuds de calcul n'ont souvent pas Internet)
            self.check_run_dir(cfg, run_dir)
            download_dataset(cfg.dataset)
            print(f"TASKS {len(tasks)} {run_dir}")
            return

        log.info("[config] %s", os.path.abspath(path) if path else "aucun fichier, valeurs par défaut")
        self.check_run_dir(cfg, run_dir)
        os.makedirs(run_dir, exist_ok=True)
        tag = "merge" if args.merge_only else ("all" if args.task_index is None else f"task_{args.task_index:03d}")
        setup_logging(os.path.join(run_dir, "logs"), filename=f"{tag}.txt")
        log.info("[run] %s | %d tâche(s)", os.path.abspath(run_dir), len(tasks))

        if args.merge_only:
            self._merge(cfg, run_dir)
            return

        if args.task_index is not None:
            if not 0 <= args.task_index < len(tasks):
                log.info("Tâche %d hors de la liste (%d tâches) : rien à faire.", args.task_index, len(tasks))
                return
            selected = [tasks[args.task_index]]
        else:
            selected = tasks
        todo = [t for t in selected if not self.is_done(os.path.join(run_dir, "parts", t.id))]
        if not todo:
            log.info("Tout est déjà terminé.")
            self._merge(cfg, run_dir)
            return

        device = configure_device(force_cpu=args.cpu, precision=cfg.precision)
        data = load_dataset(cfg.dataset, val_size=cfg.val_size, seed=0)
        if cfg.train_subset:
            data["x_train"] = data["x_train"][:cfg.train_subset]
            data["y_train"] = data["y_train"][:cfg.train_subset]
        ctx = self.prepare(cfg, data)
        self.write_config(cfg, run_dir, **ctx, device=device, environment=environment_info(device))

        with tf.device(default_device()):
            for task in todo:
                part_dir = os.path.join(run_dir, "parts", task.id)
                if self.USES_PARTS:
                    os.makedirs(part_dir, exist_ok=True)
                log.info("=== tâche %s %s ===", task.id, json.dumps(task.params))
                t0 = time.time()
                result = self.run_task(task, cfg, data, ctx, part_dir, args.verbose)
                cleanup()  # libère les modèles (sinon la RAM grossit à chaque tâche)
                if result is not None:
                    write_json(os.path.join(part_dir, "result.json"),
                               {"task": task.id, **task.params, **result, "task_time_s": time.time() - t0})
                self._merge(cfg, run_dir)   # résultats à jour après chaque tâche
        log.info("Résultats : %s", run_dir)

    def _merge(self, cfg, run_dir: str) -> None:
        try:
            self.merge(cfg, run_dir, self.read_parts(run_dir))
        except Exception:  # une fusion ratée ne doit pas faire échouer la tâche : le job de fusion refera le travail
            log.exception("Fusion des résultats impossible pour l'instant")
