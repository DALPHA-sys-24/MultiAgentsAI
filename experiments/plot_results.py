"""
Figures (PNG et PDF) et tableau récapitulatif, sans Jupyter. Médiane et bande des quantiles
5 % / 95 % sur les graines partout.

Une campagne (le cas normal ; la fusion de la campagne le fait déjà toute seule) :
    python -m experiments.plot_results lr_sweep
    python -m experiments.plot_results alpha_sweep --formats pdf
    -> results/<campagne>/<dataset>/<arch>_<K>agents/figures/
       summary.csv, accuracy_vs_<param>, diversity_vs_<param>, paired_difference, learning_curves

Des dossiers donnés à la main (ex. runs lancés sans campagne) :
    python -m experiments.plot_results --run-dirs results/collaboration/cifar10/resnet20_3agents/dml \\
        results/collaboration/cifar10/resnet20_3agents/global_interaction_multiplicative \\
        --reference results/reference/cifar10/resnet20_x3params_sgd_lr0.01 --params alpha --out-dir figures_article

Pas de GPU nécessaire : quelques secondes sur le nœud de connexion.
"""
import argparse
import sys

from collaborative_learning.evaluation.plots import plot_campaign, plot_run_dirs
from collaborative_learning.utils.log import setup_logging
from experiments.common import normalize_flags


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("campaign", nargs="?", help="Nom (configs/campaigns/<nom>.yaml) ou chemin d'une campagne.")
    p.add_argument("--run-dirs", nargs="+", metavar="DOSSIER", help="Dossiers de K agents à comparer (sans campagne).")
    p.add_argument("--reference", metavar="DOSSIER", help="Dossier du grand modèle (avec --run-dirs).")
    p.add_argument("--params", nargs="+", default=["alpha"], help="Paramètres en abscisse (avec --run-dirs).")
    p.add_argument("--out-dir", default="figures", help="Dossier de sortie (avec --run-dirs).")
    p.add_argument("--set", nargs="+", metavar="CLÉ=VALEUR", help="Mêmes --set que pour la campagne.")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"],
                   help="Formats des figures (défaut : png pdf).")
    args = p.parse_args(normalize_flags(sys.argv[1:] if argv is None else argv))
    setup_logging(None)
    if args.run_dirs:
        plot_run_dirs(args.run_dirs, args.out_dir, args.reference, args.params, args.formats)
    elif args.campaign:
        from experiments.campaign import Campaign, parse_set
        plot_campaign(Campaign(args.campaign, parse_set(args.set)), formats=args.formats)
    else:
        p.error("donne une campagne (ex. lr_sweep) ou --run-dirs")


if __name__ == "__main__":
    main()
