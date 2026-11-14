"""
Figures et tableau récapitulatif des résultats de experiments.collaboration, sans Jupyter.

Un « groupe » = tous les dossiers results/collaboration/<dataset>/<arch>_<K>agents[_<tag>]/ (ta méthode,
DML, variantes _tuned...), comparés entre eux. Toutes les statistiques sont des médianes avec la bande
des quantiles 5 % / 95 % sur les graines. Sorties dans results/comparisons/<dataset>/<arch>_<K>agents/ :

    summary.csv                     une ligne par (dossier, alpha) et par grand modèle
    accuracy_vs_alpha.png/.pdf      consensus et agent médian selon alpha, grand modèle en référence
    paired_difference.png/.pdf      écart de chaque méthode avec la référence, graine par graine
    diversity_vs_alpha.png/.pdf     désaccord entre agents selon alpha
    learning_curves.png/.pdf        accuracy de validation par époque

    from collaborative_learning.evaluation.plots import plot_group
    plot_group("results", "cifar10", "resnet20", 3)
"""
import csv
import glob
import json
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)   # sinon le sous-ensemble de polices du PDF remplit les logs

Q_LO, Q_HI = 0.05, 0.95
# Palette catégorielle (une couleur par dossier) ; gris foncé pour le grand modèle
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#c9a227", "#d6457a", "#4b5563"]
BIG_COLOR = "#3a3935"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
FORM_SHORT = {"multiplicative": "mult.", "additive": "add."}


# =============================================================================
# LECTURE
# =============================================================================

def group_dirs(results_root: str, dataset: str, arch: str, num_agents: int) -> List[str]:
    """Dossiers results/collaboration/<dataset>/<arch>_<K>agents[_<tag>] qui ont un results.json."""
    base = os.path.join(results_root, "collaboration", dataset, f"{arch}_{num_agents}agents")
    dirs = [base] + sorted(glob.glob(base + "_*"))
    return [d for d in dirs if os.path.exists(os.path.join(d, "results.json"))]


def all_groups(results_root: str) -> List[tuple]:
    """Tous les (dataset, arch, K) présents sous results/collaboration/."""
    groups = set()
    for path in glob.glob(os.path.join(results_root, "collaboration", "*", "*", "config.json")):
        try:
            with open(path) as f:
                cfg = json.load(f)
            groups.add((cfg["dataset"], cfg["arch"], int(cfg["num_agents"])))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return sorted(groups)


def method_label(cfg: dict) -> str:
    loss = cfg.get("loss_type", "?")
    name = "DML" if loss == "dml" else loss
    label = f"{name} ({FORM_SHORT.get(cfg.get('loss_form'), cfg.get('loss_form'))})"
    tag = cfg.get("tag")
    extra = [t for t in (tag or "").split("_") if t and t != "dml"] if tag else []
    return label + (f" [{'_'.join(extra)}]" if extra else "")


def load_run(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    return {"dir": run_dir, "name": os.path.basename(run_dir), "config": cfg, "results": res,
            "label": method_label(cfg)}


def stats(values) -> tuple:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return (np.nan, np.nan, np.nan)
    return float(np.median(v)), float(np.quantile(v, Q_LO)), float(np.quantile(v, Q_HI))


# =============================================================================
# FIGURES
# =============================================================================

def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, out_dir: str, name: str, formats: Sequence[str]) -> List[str]:
    """Écriture atomique (deux jobs de fusion peuvent écrire la même figure en même temps)."""
    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        tmp = f"{path}.{os.getpid()}.tmp.{fmt}"
        fig.savefig(tmp, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        os.replace(tmp, path)
        paths.append(path)
    return paths


def _alpha_axis(ax, alphas):
    ax.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas])
    ax.set_xlabel("alpha  (0 = agents indépendants, cross-entropy + vote majoritaire)", color=INK)
    ax.set_xlim(-0.3, len(alphas) - 0.7)


def _big_model_band(ax, runs, key="accuracy", scale=100):
    for run in runs:
        big = run["results"].get("big_model")
        if big:
            m, lo, hi = stats([r[key] * scale for r in big["runs"]])
            ax.axhspan(lo, hi, color=BIG_COLOR, alpha=0.08, lw=0)
            ax.axhline(m, color=BIG_COLOR, ls=":", lw=2, label="grand modèle seul")
            return


def plot_vs_alpha(runs, alphas, keys, titles, ylabel, scale=100, big_key=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 4.2), squeeze=False, sharey=True)
    for ax, key, title in zip(axes[0], keys, titles):
        for i, run in enumerate(runs):
            color = COLORS[i % len(COLORS)]
            by_alpha = {}
            for r in run["results"]["runs"]:
                by_alpha.setdefault(float(r["alpha"]), []).append(r.get(key))
            xs = [j for j, a in enumerate(alphas) if a in by_alpha]
            if not xs:
                continue
            s = np.array([stats([v * scale for v in by_alpha[alphas[j]] if v is not None]) for j in xs])
            ax.fill_between(xs, s[:, 1], s[:, 2], color=color, alpha=0.15, lw=0)
            ax.plot(xs, s[:, 0], color=color, marker="o", ms=6, mec=SURFACE, mew=1.2, lw=2, label=run["label"])
        if big_key:
            _big_model_band(ax, runs, big_key, scale)
        _alpha_axis(ax, alphas)
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
        _style(ax)
    axes[0][0].set_ylabel(ylabel, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 0.92), frameon=False)
    fig.tight_layout()
    return fig


def paired_rows(runs) -> tuple:
    """Écart d'accuracy (consensus) avec la référence, graine par graine. Référence : grand modèle, sinon alpha = 0."""
    ref_name, ref = None, None
    for run in runs:
        big = run["results"].get("big_model")
        if big:
            ref_name, ref = "grand modèle seul", {r["seed"]: r["accuracy"] for r in big["runs"]}
            break
    if ref is None:
        for run in runs:
            base = [r for r in run["results"]["runs"] if float(r["alpha"]) == 0.0]
            if base:
                ref_name = f"{run['label']} α=0"
                ref = {r["seed"]: r["accuracy_consensus"] for r in base}
                break
    rows = []
    if ref is None:
        return ref_name, rows
    for i, run in enumerate(runs):
        by_alpha = {}
        for r in run["results"]["runs"]:
            by_alpha.setdefault(float(r["alpha"]), []).append(r)
        for a in sorted(by_alpha):
            name = f"{run['label']} α={a:g}"
            if name == ref_name:
                continue
            diffs = [(r["accuracy_consensus"] - ref[r["seed"]]) * 100 for r in by_alpha[a] if r["seed"] in ref]
            if diffs:
                rows.append((name, COLORS[i % len(COLORS)], np.array(diffs)))
    return ref_name, rows


def plot_paired(runs):
    import matplotlib.pyplot as plt
    ref_name, rows = paired_rows(runs)
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: np.median(r[2]))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.5))
    for i, (name, color, diffs) in enumerate(rows):
        jitter = np.linspace(-0.15, 0.15, len(diffs)) if len(diffs) > 1 else [0.0]
        ax.plot(diffs, i + np.asarray(jitter), "o", ms=7, color=color, mec=SURFACE, mew=1.5)
        ax.plot(np.median(diffs), i, "|", ms=18, mew=2.5, color=INK)
        wins = int((diffs > 0).sum())
        ax.annotate(f"{np.median(diffs):+.2f}  ({wins}/{len(diffs)})", (1.0, i), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel(f"écart d'accuracy du consensus avec « {ref_name} » (points), même graine", color=INK)
    ax.set_title("Différence appariée par graine (trait : médiane ; à droite : médiane et graines gagnées)",
                 color=INK, loc="left", fontweight="bold", fontsize=10.5)
    _style(ax, grid_axis="x")
    fig.tight_layout()
    return fig


def plot_learning_curves(runs, key="val_accuracy"):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 4), squeeze=False, sharey=True)
    for ax, run in zip(axes[0], runs):
        by_alpha = {}
        for r in run["results"]["runs"]:
            if r["history"].get(key):
                by_alpha.setdefault(float(r["alpha"]), []).append(r["history"][key])
        ramp = plt.cm.Blues(np.linspace(0.4, 0.95, max(len(by_alpha), 1)))
        for color, a in zip(ramp, sorted(by_alpha)):
            m = min(len(h) for h in by_alpha[a])
            arr = np.array([h[:m] for h in by_alpha[a]]) * 100
            x = np.arange(1, m + 1)
            ax.fill_between(x, np.quantile(arr, Q_LO, 0), np.quantile(arr, Q_HI, 0), color=color, alpha=0.15, lw=0)
            ax.plot(x, np.median(arr, 0), color=color, lw=2, label=f"α = {a:g}")
        big = run["results"].get("big_model")
        if big:
            hs = [r["history"][key] for r in big["runs"] if r["history"].get(key)]
            if hs:
                m = min(len(h) for h in hs)
                arr = np.array([h[:m] for h in hs]) * 100
                x = np.arange(1, m + 1)
                ax.fill_between(x, np.quantile(arr, Q_LO, 0), np.quantile(arr, Q_HI, 0), color=BIG_COLOR, alpha=0.1, lw=0)
                ax.plot(x, np.median(arr, 0), color=BIG_COLOR, ls=":", lw=2, label="grand modèle")
        ax.set_title(run["label"], color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("époque", color=INK)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(frameon=False, fontsize=9)
        _style(ax)
    axes[0][0].set_ylabel("accuracy de validation (%)" + (" — consensus" if key == "val_accuracy" else ""), color=INK)
    fig.tight_layout()
    return fig


# =============================================================================
# TABLEAU
# =============================================================================

SUMMARY_METRICS = [("consensus", "accuracy_consensus"), ("mean_ensemble", "accuracy_mean_ensemble"),
                   ("agent_median", "accuracy_agent_median"), ("agent_best", "accuracy_agent_best"),
                   ("disagreement", "disagreement"), ("nll_consensus", "nll_consensus"),
                   ("ece_consensus", "ece_consensus")]


def write_summary_csv(runs, path: str) -> None:
    header = ["run_dir", "method", "alpha", "loss", "strategy", "n_seeds"]
    for name, _ in SUMMARY_METRICS:
        header += [f"{name}_median", f"{name}_q05", f"{name}_q95"]
    rows = []
    for run in runs:
        by_alpha = {}
        for r in run["results"]["runs"]:
            by_alpha.setdefault(float(r["alpha"]), []).append(r)
        for a in sorted(by_alpha):
            rs = by_alpha[a]
            row = [run["name"], run["label"], f"{a:g}", f"{rs[0].get('loss_type', run['config'].get('loss_type'))} "
                   f"({rs[0].get('loss_form', run['config'].get('loss_form'))})", rs[0].get("strategy"), len(rs)]
            for _, key in SUMMARY_METRICS:
                row += [f"{v:.6g}" for v in stats([r.get(key) for r in rs])]
            rows.append(row)
        big = run["results"].get("big_model")
        if big:
            row = [run["name"], "grand modèle seul", "", run["config"].get("reference_loss", "ce"), "", len(big["runs"])]
            for name, _ in SUMMARY_METRICS:
                key = {"consensus": "accuracy", "nll_consensus": "nll", "ece_consensus": "ece"}.get(name)
                row += [f"{v:.6g}" for v in stats([r[key] for r in big["runs"]])] if key else ["", "", ""]
            rows.append(row)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def plot_group(results_root: str, dataset: str, arch: str, num_agents: int,
               formats: Sequence[str] = ("png", "pdf"), run_dirs: Optional[List[str]] = None,
               out_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """Toutes les figures et le tableau d'un groupe (dataset, arch, K). Renvoie {nom: [fichiers]}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "text.color": INK, "pdf.fonttype": 42})

    dirs = run_dirs or group_dirs(results_root, dataset, arch, num_agents)
    runs = [load_run(d) for d in dirs]
    runs = [r for r in runs if r["results"].get("runs")]
    if not runs:
        log.info("[figures] aucun résultat pour %s / %s / K = %d", dataset, arch, num_agents)
        return {}
    out_dir = out_dir or os.path.join(results_root, "comparisons", dataset, f"{arch}_{num_agents}agents")
    os.makedirs(out_dir, exist_ok=True)
    alphas = sorted({float(r["alpha"]) for run in runs for r in run["results"]["runs"]})
    title = f"{num_agents} × {arch} — {dataset}"

    out = {}
    path = os.path.join(out_dir, "summary.csv")
    write_summary_csv(runs, path)
    out["summary"] = [path]

    fig = plot_vs_alpha(runs, alphas, ["accuracy_consensus", "accuracy_agent_median"],
                        [f"Consensus — {title}", "Agent médian (un agent seul)"], "accuracy de test (%)",
                        big_key="accuracy")
    out["accuracy_vs_alpha"] = _save(fig, out_dir, "accuracy_vs_alpha", formats)
    plt.close(fig)

    fig = plot_vs_alpha(runs, alphas, ["disagreement"], [f"Diversité des agents — {title}"],
                        "désaccord entre agents (%)")
    out["diversity_vs_alpha"] = _save(fig, out_dir, "diversity_vs_alpha", formats)
    plt.close(fig)

    fig = plot_paired(runs)
    if fig is not None:
        out["paired_difference"] = _save(fig, out_dir, "paired_difference", formats)
        plt.close(fig)

    fig = plot_learning_curves(runs)
    out["learning_curves"] = _save(fig, out_dir, "learning_curves", formats)
    plt.close(fig)

    log.info("[figures] %d dossier(s) -> %s", len(runs), os.path.abspath(out_dir))
    return out
