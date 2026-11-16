"""
Figures et tableau récapitulatif d'une campagne (ou de dossiers donnés à la main), sans Jupyter.

Pour chaque groupe (dataset, arch, K) d'une campagne, dans results/<campagne>/<dataset>/<arch>_<K>agents/figures/ :

    summary.csv                      une ligne par (méthode, point de la grille) et pour le grand modèle
    accuracy_vs_<param>.png/.pdf     consensus et agent médian selon chaque paramètre balayé (alpha, lr...),
                                     une courbe par méthode, grand modèle seul en pointillés
    diversity_vs_<param>.png/.pdf    désaccord entre agents selon ce paramètre
    paired_difference.png/.pdf       écart d'accuracy avec le grand modèle, graine par graine
    learning_curves.png/.pdf         accuracy de validation par époque, un panneau par méthode

Toutes les statistiques : médiane et bande des quantiles 5 % / 95 % sur les graines.

    from collaborative_learning.evaluation.plots import plot_campaign, plot_run_dirs
    plot_campaign(Campaign("lr_sweep"))
    plot_run_dirs(["results/.../dml/alpha1", "results/.../dml/alpha10"], reference_dir=..., out_dir="figures")
"""
import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)   # sinon le sous-ensemble de polices du PDF remplit les logs

Q_LO, Q_HI = 0.05, 0.95
# Palette catégorielle (une couleur par méthode) ; gris foncé pour le grand modèle
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#c9a227", "#d6457a", "#4b5563"]
BIG_COLOR = "#3a3935"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
INDEPENDENT = "independent"
METHOD_LABELS = {"global_interaction": "interaction globale", "dml": "DML", INDEPENDENT: "agents indépendants"}
PARAM_LABELS = {"alpha": "alpha  (0 = agents indépendants, cross-entropy + vote majoritaire)",
                "lr": "learning rate des agents (échelle log)"}
LOG_PARAMS = ("lr", "weight_decay")


# =============================================================================
# LECTURE
# =============================================================================

def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


METHOD_ORDER = ["global_interaction", "dml", INDEPENDENT]   # même couleur d'une figure à l'autre


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_color(method: str) -> str:
    if method not in METHOD_ORDER:
        METHOD_ORDER.append(method)
    return COLORS[METHOD_ORDER.index(method) % len(COLORS)]


def load_entry(run_dir: str, method: Optional[str] = None, point: Optional[Dict[str, Any]] = None) -> Optional[dict]:
    """Un dossier de K agents fusionné : config, résultats, méthode et valeurs de la grille."""
    cfg, res = _read_json(os.path.join(run_dir, "config.json")), _read_json(os.path.join(run_dir, "results.json"))
    if cfg is None or res is None or not res.get("runs"):
        return None
    method = method or res.get("method") or cfg.get("method") or f"{cfg.get('loss_type')}_{cfg.get('loss_form')}"
    return {"dir": run_dir, "config": cfg, "results": res, "method": method, "point": dict(point or {})}


def load_reference(run_dir: Optional[str]) -> Optional[dict]:
    if not run_dir:
        return None
    cfg, res = _read_json(os.path.join(run_dir, "config.json")), _read_json(os.path.join(run_dir, "results.json"))
    if cfg is None or res is None or not res.get("runs"):
        return None
    return {"dir": run_dir, "config": cfg, "results": res, "lr": cfg.get("lr")}


def stats(values) -> tuple:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return (np.nan, np.nan, np.nan)
    return float(np.median(v)), float(np.quantile(v, Q_LO)), float(np.quantile(v, Q_HI))


def rows_of(entry: dict) -> List[dict]:
    """Une ligne par (alpha, graine), avec les valeurs de la grille du dossier."""
    out = []
    for r in entry["results"]["runs"]:
        row = dict(r)
        row.update({k: v for k, v in entry["point"].items() if k != "alpha"})
        out.append(row)
    return out


def value_of(entry: dict, row: dict, param: str):
    if param == "alpha":
        return float(row["alpha"])
    if param in entry["point"]:
        return entry["point"][param]
    value = entry["config"].get(param)
    return value[0] if isinstance(value, list) and len(set(map(str, value))) == 1 else value


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
    """Écriture atomique (fichier temporaire puis remplacement)."""
    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        tmp = f"{path}.{os.getpid()}.tmp.{fmt}"
        fig.savefig(tmp, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        os.replace(tmp, path)
        paths.append(path)
    return paths


def _reference_band(ax, reference, key="accuracy", scale=100):
    if not reference:
        return
    m, lo, hi = stats([r[key] * scale for r in reference["results"]["runs"] if r.get(key) is not None])
    ax.axhspan(lo, hi, color=BIG_COLOR, alpha=0.08, lw=0)
    lr = reference.get("lr")
    label = "grand modèle seul" + (f" (lr* = {lr:g})" if isinstance(lr, (int, float)) else "")
    ax.axhline(m, color=BIG_COLOR, ls=":", lw=2, label=label)


def series_by_method(entries, param: str, other_params: Sequence[str]) -> Dict[str, Dict[Any, list]]:
    """
    {libellé de courbe: {valeur du paramètre: [lignes]}}. Une courbe par méthode (et par valeur des
    autres paramètres balayés). En abscisse alpha, les agents indépendants (alpha = 0) sont le premier
    point de chaque méthode.
    """
    series: Dict[str, Dict[Any, list]] = {}
    others = [e for e in entries if e["method"] != INDEPENDENT]
    share_independent = param == "alpha" and bool(others)
    methods = []
    for e in others:
        if e["method"] not in methods:
            methods.append(e["method"])

    def label_of(method, e, row):
        fixed = ", ".join(f"{p} = {fmt(value_of(e, row, p))}" for p in other_params)
        return method, method_label(method) + (f" ({fixed})" if fixed else "")

    for e in entries:
        for row in rows_of(e):
            targets = methods if (share_independent and e["method"] == INDEPENDENT) else [e["method"]]
            for method in targets:
                series.setdefault(label_of(method, e, row), {}).setdefault(value_of(e, row, param), []).append(row)
    return series


def fmt(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def plot_vs_param(entries, reference, param, keys, titles, ylabel, other_params=(), scale=100, show_reference=True):
    import matplotlib.pyplot as plt
    series = series_by_method(entries, param, other_params)
    by_method_color = len({m for m, _ in series}) == len(series)   # une courbe par méthode : sa couleur
    values = sorted({v for s in series.values() for v in s})
    log_x = param in LOG_PARAMS and all(isinstance(v, (int, float)) and v > 0 for v in values)
    fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 4.2), squeeze=False, sharey=True)
    for ax, key, title in zip(axes[0], keys, titles):
        for i, ((method, label), by_value) in enumerate(series.items()):
            color = method_color(method) if by_method_color else COLORS[i % len(COLORS)]
            xs = sorted(by_value)
            s = np.array([stats([r.get(key) * scale for r in by_value[x] if r.get(key) is not None]) for x in xs])
            pos = xs if log_x else [values.index(x) for x in xs]
            ax.fill_between(pos, s[:, 1], s[:, 2], color=color, alpha=0.15, lw=0)
            ax.plot(pos, s[:, 0], color=color, marker="o", ms=6, mec=SURFACE, mew=1.2, lw=2, label=label)
        if show_reference:
            _reference_band(ax, reference)
        if log_x:
            ax.set_xscale("log")
            ax.set_xticks(values, [fmt(v) for v in values])
            ax.minorticks_off()
        else:
            ax.set_xticks(range(len(values)), [fmt(v) for v in values])
            ax.set_xlim(-0.3, len(values) - 0.7)
        ax.set_xlabel(PARAM_LABELS.get(param, param), color=INK)
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
        _style(ax)
    axes[0][0].set_ylabel(ylabel, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 0.92), frameon=False)
    fig.tight_layout()
    return fig


def entry_name(e: dict) -> str:
    point = ", ".join(f"{'α' if k == 'alpha' else k}={fmt(v)}" for k, v in e["point"].items())
    if not point:
        alphas = sorted({float(r["alpha"]) for r in e["results"]["runs"]})
        point = "α=" + "/".join(fmt(a) for a in alphas)
    return f"{method_label(e['method'])} {point}"


def plot_paired(entries, reference):
    """Écart d'accuracy du consensus avec le grand modèle (sinon les agents indépendants), même graine."""
    import matplotlib.pyplot as plt
    if reference:
        ref_name = "grand modèle seul"
        ref = {r["seed"]: r["accuracy"] for r in reference["results"]["runs"]}
    else:
        base = [r for e in entries if e["method"] == INDEPENDENT for r in e["results"]["runs"]]
        if not base:
            return None
        ref_name, ref = "agents indépendants", {r["seed"]: r["accuracy_consensus"] for r in base}
    rows = []
    for e in entries:
        if not reference and e["method"] == INDEPENDENT:
            continue
        for alpha in sorted({float(r["alpha"]) for r in e["results"]["runs"]}):
            diffs = [(r["accuracy_consensus"] - ref[r["seed"]]) * 100
                     for r in e["results"]["runs"] if float(r["alpha"]) == alpha and r["seed"] in ref]
            if diffs:
                name = entry_name(e) if e["point"] else f"{method_label(e['method'])} α={fmt(alpha)}"
                rows.append((name, method_color(e["method"]), np.array(diffs)))
    if not rows:
        return None
    rows.sort(key=lambda r: np.median(r[2]))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.5))
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


def _curve(ax, histories, color, label, ls="-"):
    from matplotlib.ticker import MaxNLocator
    m = min(len(h) for h in histories)
    arr = np.array([h[:m] for h in histories]) * 100
    x = np.arange(1, m + 1)
    ax.fill_between(x, np.quantile(arr, Q_LO, 0), np.quantile(arr, Q_HI, 0), color=color, alpha=0.12, lw=0)
    ax.plot(x, np.median(arr, 0), color=color, lw=2, ls=ls, label=label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_learning_curves(entries, reference, key="val_accuracy"):
    """Un panneau par méthode, une courbe par point de la grille (bleu clair -> foncé), grand modèle en pointillés."""
    import matplotlib.pyplot as plt
    methods = []
    for e in entries:
        if e["method"] not in methods:
            methods.append(e["method"])
    n = len(methods)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4 * nrows), squeeze=False, sharey=True)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    ref_hist = [r["history"][key] for r in reference["results"]["runs"] if r.get("history", {}).get(key)] \
        if reference else []
    for ax, method in zip(axes.flat, methods):
        curves = []
        for e in entries:
            if e["method"] != method:
                continue
            for alpha in sorted({float(r["alpha"]) for r in e["results"]["runs"]}):
                hs = [r["history"][key] for r in e["results"]["runs"]
                      if float(r["alpha"]) == alpha and r["history"].get(key)]
                if hs:
                    label = ", ".join(f"{'α' if k == 'alpha' else k} = {fmt(v)}" for k, v in e["point"].items()) \
                        or f"α = {fmt(alpha)}"
                    curves.append((label, hs))
        ramp = plt.cm.Blues(np.linspace(0.4, 0.95, max(len(curves), 1)))
        for color, (label, hs) in zip(ramp, curves):
            _curve(ax, hs, color, label)
        if ref_hist:
            _curve(ax, ref_hist, BIG_COLOR, "grand modèle seul", ls=":")
        ax.set_title(method_label(method), color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("époque", color=INK)
        ax.legend(frameon=False, fontsize=8.5)
        _style(ax)
    for row in axes:
        row[0].set_ylabel("accuracy de validation (%)" + (" — consensus" if key == "val_accuracy" else ""), color=INK)
    fig.tight_layout()
    return fig


# =============================================================================
# TABLEAU
# =============================================================================

SUMMARY_METRICS = [("consensus", "accuracy_consensus"), ("mean_ensemble", "accuracy_mean_ensemble"),
                   ("agent_median", "accuracy_agent_median"), ("agent_best", "accuracy_agent_best"),
                   ("disagreement", "disagreement"), ("nll_consensus", "nll_consensus"),
                   ("ece_consensus", "ece_consensus")]


def write_summary_csv(entries, reference, params: Sequence[str], path: str) -> None:
    header = ["method", *params, "alpha", "loss", "strategy", "n_seeds", "run_dir"]
    for name, _ in SUMMARY_METRICS:
        header += [f"{name}_median", f"{name}_q05", f"{name}_q95"]
    rows = []
    for e in entries:
        by_alpha = {}
        for r in e["results"]["runs"]:
            by_alpha.setdefault(float(r["alpha"]), []).append(r)
        for a in sorted(by_alpha):
            rs = by_alpha[a]
            row = [e["method"], *[fmt(value_of(e, rs[0], p)) for p in params], fmt(a),
                   f"{rs[0].get('loss_type')} ({rs[0].get('loss_form')})", rs[0].get("strategy"), len(rs), e["dir"]]
            for _, key in SUMMARY_METRICS:
                row += [f"{v:.6g}" for v in stats([r.get(key) for r in rs])]
            rows.append(row)
    if reference:
        runs = reference["results"]["runs"]
        lr = reference.get("lr")
        row = ["grand modèle seul", *[fmt(lr) if p == "lr" else "" for p in params], "",
               reference["config"].get("loss", "ce"), "", len(runs), reference["dir"]]
        for name, _ in SUMMARY_METRICS:
            key = {"consensus": "accuracy", "nll_consensus": "nll", "ece_consensus": "ece"}.get(name)
            row += [f"{v:.6g}" for v in stats([r[key] for r in runs])] if key else ["", "", ""]
        rows.append(row)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


# =============================================================================
# POINTS D'ENTRÉE
# =============================================================================

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "text.color": INK, "pdf.fonttype": 42})
    return plt


def plot_entries(entries, reference, params: Sequence[str], out_dir: str, title: str,
                 formats: Sequence[str] = ("png", "pdf")) -> Dict[str, List[str]]:
    """Toutes les figures d'un groupe : `params` = paramètres balayés (une figure selon chacun)."""
    plt = _setup_matplotlib()
    if not entries:
        log.info("[figures] aucun résultat pour %s", title)
        return {}
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    path = os.path.join(out_dir, "summary.csv")
    write_summary_csv(entries, reference, [p for p in params if p != "alpha"], path)
    out["summary"] = [path]
    for param in params:
        others = [p for p in params if p != param]
        fig = plot_vs_param(entries, reference, param, ["accuracy_consensus", "accuracy_agent_median"],
                            [f"Consensus — {title}", "Agent médian (un agent seul)"], "accuracy de test (%)",
                            other_params=others)
        out[f"accuracy_vs_{param}"] = _save(fig, out_dir, f"accuracy_vs_{param}", formats)
        plt.close(fig)
        fig = plot_vs_param(entries, reference, param, ["disagreement"], [f"Diversité des agents — {title}"],
                            "désaccord entre agents (%)", other_params=others, show_reference=False)
        out[f"diversity_vs_{param}"] = _save(fig, out_dir, f"diversity_vs_{param}", formats)
        plt.close(fig)
    fig = plot_paired(entries, reference)
    if fig is not None:
        out["paired_difference"] = _save(fig, out_dir, "paired_difference", formats)
        plt.close(fig)
    fig = plot_learning_curves(entries, reference)
    out["learning_curves"] = _save(fig, out_dir, "learning_curves", formats)
    plt.close(fig)
    log.info("[figures] %s -> %s", title, os.path.abspath(out_dir))
    for name, paths in out.items():
        log.info("  %-22s %s", name, ", ".join(os.path.basename(x) for x in paths))
    return out


def swept_params(campaign) -> List[str]:
    """Paramètres de la grille qui prennent au moins deux valeurs (hors dataset / arch / K : groupes)."""
    return [k for k, v in campaign.grid.items() if len(v) >= 2 and k not in ("dataset", "arch", "num_agents")]


def plot_campaign(campaign, formats: Sequence[str] = ("png", "pdf")) -> Dict[tuple, Dict[str, List[str]]]:
    """Figures de chaque groupe (dataset, arch, K) d'une campagne, dans .../<arch>_<K>agents/figures/."""
    params = swept_params(campaign) or ["alpha"]
    groups: Dict[tuple, list] = {}
    for run in campaign.runs():
        if run.experiment == "collaboration":
            groups.setdefault(run.group, []).append(run)
    out = {}
    for group, runs in groups.items():
        dataset, arch, k = group
        entries = [e for e in (load_entry(r.run_dir, r.method, r.point) for r in runs) if e is not None]
        ref_run = campaign.reference_for(runs[0])
        reference = load_reference(ref_run.run_dir) if ref_run else None
        out_dir = os.path.join(os.path.dirname(os.path.dirname(runs[0].run_dir)), "figures")
        out[group] = plot_entries(entries, reference, params, out_dir, f"{k} × {arch} — {dataset}", formats)
    return out


def plot_run_dirs(run_dirs: Sequence[str], out_dir: str, reference_dir: Optional[str] = None,
                  params: Sequence[str] = ("alpha",), formats: Sequence[str] = ("png", "pdf")):
    """Figures de dossiers donnés à la main (paramètres lus dans leur config.json)."""
    entries = [e for e in (load_entry(d) for d in run_dirs) if e is not None]
    for e in entries:
        e["point"] = {p: value_of(e, {}, p) for p in params if p != "alpha"}
    cfg = entries[0]["config"] if entries else {}
    title = f"{cfg.get('num_agents', '?')} × {cfg.get('arch', '?')} — {cfg.get('dataset', '?')}"
    return plot_entries(entries, load_reference(reference_dir), list(params), out_dir, title, formats)
