"""
Campagne : un fichier YAML (configs/campaigns/) décrit tout ce qu'on lance d'un coup — datasets,
méthodes, valeurs à balayer (grid) et grand modèle de référence — et ce script en fait une liste
de tâches numérotées, une par tâche du job array SLURM.

    # configs/campaigns/lr_sweep.yaml
    experiment: collaboration
    datasets: [fashion_mnist, cifar10]
    methods: [global_interaction, dml]     # configs/methods/<nom>.yaml
    use_tuned_params: true                 # results/hyperparameter_search/<dataset>/<arch>/tuned_params.yaml
    settings: {arch: resnet20, num_agents: 3, epochs: 200, n_runs: 5}
    grid:                                  # produit cartésien : un dossier par point
      alpha: [1]
      lr: [0.1, 0.01, 0.001, 0.0001]
    reference: true                        # grand modèle seul au lr de base (lr*)

Config d'un point (la dernière valeur lue gagne) :
    configs/collaboration.yaml < configs/datasets/<dataset>.yaml < configs/methods/<méthode>.yaml
    < tuned_params.yaml (si use_tuned_params) < settings < valeurs du point
alpha = 0 : agents indépendants (configs/methods/independent.yaml), entraînés une seule fois
quelle que soit la méthode.

Grand modèle (reference: true) : un par (dataset, arch, K), avec les réglages de base (tuned_params
et settings, pas ceux de la grille), dans results/reference/... : partagé entre campagnes.

Dossiers :
    results/<campagne>/<dataset>/<arch>_<K>agents/<méthode>/<point>/   (ex. dml/alpha1_lr0.1)
    results/<campagne>/<dataset>/<arch>_<K>agents/figures/            (fusion : figures + summary.csv)
    results/reference/<dataset>/<arch>_x<K>params_<optimiseur>_lr<lr>[_tuned]/

Exemples (depuis la racine du dépôt) :
    python -m experiments.campaign lr_sweep --list         # tâches, dossiers, commande sbatch
    python -m experiments.campaign lr_sweep --task 7       # une tâche (ce que fait le job array)
    python -m experiments.campaign lr_sweep --merge        # results.json de chaque dossier + figures
    python -m experiments.campaign quick_test --cpu        # tout, l'un après l'autre (test local)
    python -m experiments.campaign lr_sweep --list --set epochs=2 train_subset=500
    sbatch --array=0-89 slurm/campaign.sh lr_sweep         # cluster
"""
import argparse
import dataclasses
import difflib
import itertools
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from collaborative_learning.utils.config import (CollaborationConfig, HyperparamSearchConfig, ReferenceConfig,
                                                 config_from_dict, fmt_value)
from collaborative_learning.utils.config_files import CONFIG_DIR, read_config_file, read_raw, resolve_config_path

log = logging.getLogger("collaborative_learning")

CAMPAIGN_DIR = os.path.join(CONFIG_DIR, "campaigns")
CAMPAIGN_KEYS = ("name", "experiment", "datasets", "methods", "use_tuned_params", "settings", "grid", "reference")
EXPERIMENTS = ("collaboration", "hyperparameter_search")
INDEPENDENT = "independent"                 # méthode utilisée pour alpha = 0
GROUP_KEYS = ("dataset", "arch", "num_agents")   # déjà dans le chemin, pas dans le nom du point
TUNED_FILE, LEGACY_TUNED_FILE = "tuned_params.yaml", "tuned_collaboration.json"
TUNED_KEYS = ("batch_size", "optimizer", "lr", "weight_decay", "wd_exclude_norm_bias", "optimizer_kwargs",
              "warmup_epochs", "augment", "val_size", "precision")


@dataclass
class Run:
    """Un dossier de run de la campagne et sa config."""
    experiment: str                  # collaboration | reference | hyperparameter_search
    dataset: str
    cfg: Any
    run_dir: str
    method: Optional[str] = None
    point: Dict[str, Any] = field(default_factory=dict)   # valeurs de la grille (alpha, lr...)
    n_tasks: int = 0

    @property
    def group(self) -> Tuple[str, str, int]:
        return self.dataset, self.cfg.arch, int(getattr(self.cfg, "num_agents", 1))


def experiment_class(name: str):
    if name == "collaboration":
        from experiments.collaboration import Collaboration
        return Collaboration
    if name == "reference":
        from experiments.reference import Reference
        return Reference
    if name == "hyperparameter_search":
        from experiments.hyperparameter_search import HyperparameterSearch
        return HyperparameterSearch
    raise SystemExit(f"Expérience inconnue : {name}")


def point_name(point: Dict[str, Any]) -> str:
    """Nom du dossier d'un point : alpha1_lr0.1 (clés dans l'ordre de la grille)."""
    parts = [f"{k}{fmt_value(v)}" for k, v in point.items() if k not in GROUP_KEYS]
    return "_".join(parts) or "run"


def parse_set(items: List[str]) -> Dict[str, Any]:
    """--set epochs=2 lr=0.1 alphas=[0,1] -> dict (valeurs lues comme du YAML)."""
    import yaml
    out = {}
    for item in items or []:
        key, eq, value = item.partition("=")
        if not eq:
            raise SystemExit(f"--set attend clé=valeur, reçu : {item}")
        out[key.strip().replace("-", "_")] = yaml.safe_load(value)
    return out


class Campaign:
    def __init__(self, name_or_path: str, overrides: Optional[Dict[str, Any]] = None):
        path = resolve_config_path(name_or_path, base_dir=CAMPAIGN_DIR)
        self.path = path
        spec = read_raw(path)
        unknown = [k for k in spec if k not in CAMPAIGN_KEYS and not k.startswith("_")]
        if unknown:
            hints = [f"'{k}'" + (f" (voulais-tu dire '{c[0]}' ?)" if (c := difflib.get_close_matches(k, CAMPAIGN_KEYS, 1))
                                 else "") for k in unknown]
            raise SystemExit(f"{path} : clé(s) inconnue(s) " + ", ".join(hints)
                             + f" ; clés possibles : {', '.join(CAMPAIGN_KEYS)}")
        self.name = spec.get("name") or os.path.splitext(os.path.basename(path))[0]
        self.experiment = spec.get("experiment", "collaboration")
        if self.experiment not in EXPERIMENTS:
            raise SystemExit(f"{path} : experiment = {self.experiment!r}, choix possibles {list(EXPERIMENTS)}")
        self.datasets = list(spec.get("datasets") or [])
        if not self.datasets:
            raise SystemExit(f"{path} : donne au moins un dataset (datasets: [cifar10])")
        self.methods = list(spec.get("methods") or ["global_interaction"])
        self.use_tuned = bool(spec.get("use_tuned_params", False))
        self.settings = {**(spec.get("settings") or {}), **(overrides or {})}
        self.grid = {k: (v if isinstance(v, list) else [v]) for k, v in (spec.get("grid") or {}).items()}
        self.reference = bool(spec.get("reference", False))
        self._runs: Optional[List[Run]] = None

    # --- configs -------------------------------------------------------------
    def tuned_values(self, dataset: str, arch: str, out_dir: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Paramètres de la recherche d'hyperparamètres du dataset (ou {} s'il n'y en a pas)."""
        if not self.use_tuned:
            return {}, None
        folder = os.path.join(out_dir, "hyperparameter_search", dataset, arch)
        for name in (TUNED_FILE, LEGACY_TUNED_FILE):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                values = read_raw(path)
                return {k: v for k, v in values.items() if k in TUNED_KEYS}, path
        return {}, None

    def _values(self, base: str, dataset: str, method: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[str]]:
        values = read_config_file(base)
        values.update(read_config_file(f"datasets/{dataset}"))
        if method:
            values.update(read_config_file(f"methods/{method}"))
        arch = self.settings.get("arch", values.get("arch", "resnet20"))
        out_dir = self.settings.get("out_dir", values.get("out_dir", "results"))
        tuned, tuned_path = self.tuned_values(dataset, arch, out_dir)
        values.update(tuned)
        values.update(self.settings)
        return values, tuned_path

    def points(self) -> List[Dict[str, Any]]:
        keys = list(self.grid)
        return [dict(zip(keys, combo)) for combo in itertools.product(*(self.grid[k] for k in keys))] or [{}]

    def runs(self) -> List[Run]:
        if self._runs is None:
            self._runs = self._build_runs()
        return self._runs

    def _build_runs(self) -> List[Run]:
        runs: List[Run] = []
        seen = set()
        for dataset in self.datasets:
            if self.experiment == "hyperparameter_search":
                values, _ = self._values("hyperparameter_search", dataset)
                cfg = config_from_dict(HyperparamSearchConfig, values).validate()
                runs.append(Run("hyperparameter_search", dataset, cfg, cfg.run_dir()))
                continue
            references = {}
            for method in self.methods:
                for point in self.points():
                    alpha = point.get("alpha")
                    m = INDEPENDENT if alpha is not None and float(alpha) == 0 else method
                    values, tuned_path = self._values("collaboration", dataset, m)
                    for key, value in point.items():
                        if key == "alpha":
                            values["alphas"] = [float(value)]
                        else:
                            values[key] = value
                    cfg = config_from_dict(CollaborationConfig, values).validate()
                    run_dir = os.path.join(cfg.out_dir, self.name, dataset, f"{cfg.arch}_{cfg.num_agents}agents",
                                           cfg.method_name, point_name(point))
                    if run_dir in seen:          # alpha = 0 : un seul dossier pour toutes les méthodes
                        continue
                    seen.add(run_dir)
                    runs.append(Run("collaboration", dataset, cfg, run_dir, cfg.method_name, dict(point)))
                    if self.reference:
                        references.setdefault((cfg.arch, cfg.num_agents), tuned_path)
            for (arch, k), tuned_path in references.items():
                values, _ = self._values("reference", dataset)
                fields = {f.name for f in dataclasses.fields(ReferenceConfig)}
                values = {key: v for key, v in values.items() if key in fields or key.startswith("_")}
                values.update(arch=arch, num_agents=k)
                if tuned_path:
                    values["tag"] = "tuned"
                cfg = config_from_dict(ReferenceConfig, values).validate()
                runs.append(Run("reference", dataset, cfg, cfg.run_dir()))
        for run in runs:
            run.n_tasks = len(experiment_class(run.experiment)().tasks(run.cfg))
        return runs

    def tasks(self) -> List[Tuple[Run, int]]:
        """Liste numérotée de toutes les tâches : (dossier, indice de la tâche dans ce dossier)."""
        return [(run, i) for run in self.runs() for i in range(run.n_tasks)]

    def reference_for(self, run: Run) -> Optional[Run]:
        for r in self.runs():
            if r.experiment == "reference" and r.group == run.group:
                return r
        return None

    # --- actions -------------------------------------------------------------
    def describe(self) -> str:
        tasks = self.tasks()
        lines = [f"Campagne {self.name} ({os.path.relpath(self.path)}) : {self.experiment}"]
        start = 0
        for run in self.runs():
            extra = f" {run.method} {point_name(run.point)}" if run.experiment == "collaboration" else ""
            lines.append(f"  [{start:3d}-{start + run.n_tasks - 1:3d}] {run.experiment:22s}{extra:30s} "
                         f"{os.path.relpath(run.run_dir)}")
            start += run.n_tasks
        for dataset in self.datasets:
            arch = self.settings.get("arch", "resnet20")
            _, tuned_path = self.tuned_values(dataset, arch, self.settings.get("out_dir", "results"))
            if self.use_tuned:
                lines.append(f"  paramètres {dataset} : " + (tuned_path or "pas de recherche -> valeurs des configs"))
        lines.append(f"TASKS {len(tasks)}")
        lines.append(f"Cluster : sbatch --array=0-{len(tasks) - 1} slurm/campaign.sh {self.name}")
        return "\n".join(lines)

    def check(self) -> None:
        """Avant de soumettre : réglages compatibles avec les dossiers existants, données téléchargées."""
        from collaborative_learning.data import download_dataset
        from experiments.common import Experiment
        for run in self.runs():
            Experiment.check_run_dir(run.cfg, run.run_dir)
        for dataset in self.datasets:
            download_dataset(dataset)

    def run_task(self, index: int, cpu: bool = False, verbose: int = 2) -> None:
        tasks = self.tasks()
        if not 0 <= index < len(tasks):
            log.info("Tâche %d hors de la campagne (%d tâches) : rien à faire.", index, len(tasks))
            return
        run, i = tasks[index]
        log.info("[campagne %s] tâche %d/%d : %s, tâche %d de %s", self.name, index, len(tasks) - 1,
                 run.experiment, i, run.run_dir)
        experiment_class(run.experiment)().execute(run.cfg, run.run_dir, task_index=i, cpu=cpu, verbose=verbose)

    def run_all(self, cpu: bool = False, verbose: int = 2) -> None:
        for run in self.runs():
            experiment_class(run.experiment)().execute(run.cfg, run.run_dir, cpu=cpu, verbose=verbose)
        self.merge(plots=True)

    def merge(self, plots: bool = True, formats=("png", "pdf")) -> None:
        for run in self.runs():
            experiment_class(run.experiment)().execute(run.cfg, run.run_dir, merge_only=True, verbose=0)
        if plots and self.experiment == "collaboration":
            try:
                from collaborative_learning.evaluation.plots import plot_campaign
                plot_campaign(self, formats=formats)
            except ImportError as e:
                log.warning("Figures non faites (%s) : python -m experiments.plot_results %s", e, self.name)


def main(argv=None):
    from experiments.common import normalize_flags
    from collaborative_learning.utils.log import setup_logging
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("campaign", help="Nom (configs/campaigns/<nom>.yaml) ou chemin du fichier de campagne.")
    p.add_argument("--list", action="store_true", help="Liste les tâches et dossiers, vérifie, puis s'arrête.")
    p.add_argument("--task", type=int, metavar="I", help="N'exécute que la tâche I ($SLURM_ARRAY_TASK_ID).")
    p.add_argument("--merge", action="store_true", help="Fusionne chaque dossier puis fait les figures.")
    p.add_argument("--check-array", type=int, metavar="MAX", help="Prévient si --array=0-MAX ne couvre pas les tâches.")
    p.add_argument("--set", nargs="+", metavar="CLÉ=VALEUR", help="Remplace des settings (ex. epochs=2 n_runs=1).")
    p.add_argument("--cpu", action="store_true", help="Force le CPU.")
    p.add_argument("--verbose", type=int, default=2)
    args = p.parse_args(normalize_flags(sys.argv[1:] if argv is None else argv))
    setup_logging(None)
    campaign = Campaign(args.campaign, parse_set(args.set))

    if args.check_array is not None:
        n = len(campaign.tasks())
        if args.check_array + 1 != n:
            print(f"ATTENTION : --array s'arrête à {args.check_array} mais la campagne {campaign.name} a {n} tâches "
                  f"(sbatch --array=0-{n - 1} slurm/campaign.sh {campaign.name}).")
        return
    if args.list:
        print(campaign.describe())
        campaign.check()
        return
    if args.merge:
        campaign.merge()
        return
    if args.task is not None:
        campaign.run_task(args.task, cpu=args.cpu, verbose=args.verbose)
        return
    campaign.run_all(cpu=args.cpu, verbose=args.verbose)


if __name__ == "__main__":
    main()
