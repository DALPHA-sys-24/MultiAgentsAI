"""
Rassemble les jobs d'un sweep de alpha lancé sur le cluster (un job par couple alpha, graine).

Chaque job écrit dans <campagne>/jobs/a<alpha>_s<graine>/ ; ce script crée dans <campagne>/
un config.json, un results.json et un dossier probs/ au même format qu'un sweep lancé
d'un seul bloc, lisibles directement par notebooks/resultats_sweep.ipynb.

Exemple :
    python -m experiments.merge_sweep results/alpha_sweep_cifar10
"""
import argparse
import glob
import json
import os
import shutil

from dml.utils.config import STRATEGY_NO_INTERACTION
from dml.utils.log import setup_logging
from dml.evaluation.reporting import print_sweep_table, summary_by_alpha, write_json

log = setup_logging()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("campaign_dir", help="Dossier de la campagne (contient jobs/).")
    args = p.parse_args()

    job_dirs = sorted(d for d in glob.glob(os.path.join(args.campaign_dir, "jobs", "*"))
                      if os.path.exists(os.path.join(d, "results.json")))
    if not job_dirs:
        raise SystemExit(f"Aucun job terminé dans {os.path.abspath(args.campaign_dir)}/jobs")

    probs_dir = os.path.join(args.campaign_dir, "probs")
    os.makedirs(probs_dir, exist_ok=True)
    runs, train_time, config, agent_params = {}, 0.0, None, None
    for d in job_dirs:
        with open(os.path.join(d, "results.json")) as f:
            res = json.load(f)
        for r in res["runs"]:
            runs[(float(r["alpha"]), r["seed"])] = r     # un doublon (job relancé) remplace l'ancien
        train_time += res.get("train_time_s", 0.0)
        agent_params = res.get("agent_params", agent_params)
        if config is None:
            with open(os.path.join(d, "config.json")) as f:
                config = json.load(f)
        for npy in glob.glob(os.path.join(d, "probs", "*.npy")):
            shutil.copy2(npy, probs_dir)
        # Historiques par époque (.csv) des jobs, pour le notebook
        for csv in glob.glob(os.path.join(d, "history", "*.csv")):
            os.makedirs(os.path.join(args.campaign_dir, "history"), exist_ok=True)
            shutil.copy2(csv, os.path.join(args.campaign_dir, "history"))

    runs = [runs[key] for key in sorted(runs)]
    alphas = sorted({r["alpha"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    config.update(alphas=alphas, seeds=seeds, strategy_alpha0=STRATEGY_NO_INTERACTION)
    summary = summary_by_alpha(runs)

    with open(os.path.join(args.campaign_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    write_json(os.path.join(args.campaign_dir, "results.json"),
               {"experiment": "alpha_sweep", "summary": summary, "runs": runs,
                "agent_params": agent_params, "train_time_s": train_time})

    log.info("%d job(s), %d run(s) rassemblés", len(job_dirs), len(runs))
    missing = [(a, s) for a in alphas for s in seeds if (float(a), s) not in {(float(r['alpha']), r['seed']) for r in runs}]
    if missing:
        log.warning("Couples (alpha, graine) manquants : " + ", ".join(f"({a:g}, {s})" for a, s in missing))
    print_sweep_table(summary, config["num_agents"], config["arch"], config["loss_type"])
    log.info("Résultats : %s", args.campaign_dir)


if __name__ == "__main__":
    main()
