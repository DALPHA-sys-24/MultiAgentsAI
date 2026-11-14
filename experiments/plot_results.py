"""
Figures (PNG et PDF) et tableau récapitulatif des résultats de experiments.collaboration, sans Jupyter.

Compare tous les dossiers results/collaboration/<dataset>/<arch>_<K>agents[_<tag>]/ d'un même
dataset, d'une même architecture et d'un même K (ta méthode, DML, variantes _tuned...).
Médiane et bande des quantiles 5 % / 95 % sur les graines partout.

Exemples (depuis la racine du dépôt) :
    python -m experiments.plot_results                                   # tous les groupes trouvés
    python -m experiments.plot_results --dataset cifar10 --arch resnet20 --num-agents 3
    python -m experiments.plot_results --dataset cifar10 --formats pdf   # PDF seulement
    python -m experiments.plot_results --run-dirs results/collaboration/cifar10/resnet20_3agents \\
        results/collaboration/cifar10/resnet20_3agents_dml --out-dir figures_article

Sorties : results/comparisons/<dataset>/<arch>_<K>agents/
    summary.csv, accuracy_vs_alpha, paired_difference, diversity_vs_alpha, learning_curves (.png, .pdf)

Sur le cluster, le job de fusion (slurm/merge_results.sh) lance ce script tout seul.
Pas de GPU nécessaire : quelques secondes sur le nœud de connexion.
"""
import argparse
import logging
import os
import sys

from collaborative_learning.evaluation.plots import all_groups, load_run, plot_group
from collaborative_learning.utils.log import setup_logging
from experiments.common import normalize_flags

log = logging.getLogger("collaborative_learning")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results", help="Dossier racine des résultats (défaut : results).")
    p.add_argument("--dataset", help="Un dataset (défaut : tous ceux trouvés).")
    p.add_argument("--arch", help="Une architecture (défaut : toutes).")
    p.add_argument("--num-agents", type=int, help="Un K (défaut : tous).")
    p.add_argument("--run-dirs", nargs="+", metavar="DOSSIER", help="Liste explicite de dossiers à comparer.")
    p.add_argument("--out-dir", help="Dossier de sortie (défaut : results/comparisons/<dataset>/<arch>_<K>agents).")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"],
                   help="Formats des figures (défaut : png pdf).")
    args = p.parse_args(normalize_flags(sys.argv[1:] if argv is None else argv))
    setup_logging(None)

    if args.run_dirs:
        cfg = load_run(args.run_dirs[0])["config"]
        groups = [(cfg["dataset"], cfg["arch"], int(cfg["num_agents"]))]
    else:
        groups = [g for g in all_groups(args.results_root)
                  if (args.dataset is None or g[0] == args.dataset)
                  and (args.arch is None or g[1] == args.arch)
                  and (args.num_agents is None or g[2] == args.num_agents)]
    if not groups:
        raise SystemExit(f"Aucun résultat trouvé dans {os.path.abspath(args.results_root)}/collaboration "
                         "(lancer d'abord experiments.collaboration, ou vérifier --dataset / --arch / --num-agents).")
    for dataset, arch, k in groups:
        out = plot_group(args.results_root, dataset, arch, k, formats=args.formats,
                         run_dirs=args.run_dirs, out_dir=args.out_dir)
        for name, paths in out.items():
            log.info("  %-20s %s", name, ", ".join(os.path.basename(x) for x in paths))


if __name__ == "__main__":
    main()
