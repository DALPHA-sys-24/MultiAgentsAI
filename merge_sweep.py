"""
Rassemble les jobs d'un sweep de alpha lancé sur le cluster (un job par couple alpha, graine).

Chaque job écrit dans <campagne>/jobs/a<alpha>_s<graine>/ ; ce script crée dans <campagne>/
un config.json, un results.json et un dossier probs/ au même format qu'un sweep lancé
d'un seul bloc, lisibles directement par resultats_sweep.ipynb.

Exemple :
    python merge_sweep.py results/alpha_sweep_cifar10
"""
import argparse
import glob
import json
import os
import shutil

from run_alpha_sweep import STRATEGY_NO_INTERACTION, print_table, summary_by_alpha


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

    runs = [runs[key] for key in sorted(runs)]
    alphas = sorted({r["alpha"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    config.update(alphas=alphas, seeds=seeds, strategy_alpha0=STRATEGY_NO_INTERACTION)
    summary = summary_by_alpha(runs)

    with open(os.path.join(args.campaign_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.campaign_dir, "results.json"), "w") as f:
        json.dump({"experiment": "alpha_sweep", "summary": summary, "runs": runs,
                   "agent_params": agent_params, "train_time_s": train_time}, f, indent=2)

    print(f"{len(job_dirs)} job(s), {len(runs)} run(s) rassemblés")
    missing = [(a, s) for a in alphas for s in seeds if (float(a), s) not in {(float(r['alpha']), r['seed']) for r in runs}]
    if missing:
        print("Couples (alpha, graine) manquants : " + ", ".join(f"({a:g}, {s})" for a, s in missing))
    print_table(summary, config)
    print(f"Résultats : {args.campaign_dir}")


if __name__ == "__main__":
    main()
