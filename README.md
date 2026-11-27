# Beyond Deep Mutual Learning

K réseaux (ResNet, MobileNet) entraînés **ensemble** avec une loss multi-agents, puis combinés par un consensus, et comparés à des agents indépendants, à la Deep Mutual Learning ([Zhang et al., 2018](https://arxiv.org/abs/1706.00384)) et à un grand modèle seul de même taille. TensorFlow / Keras (TF 2.15 et 2.21), GPU ou CPU.

Tout ce qu'on lance (datasets, méthodes, valeurs de α, lr, K…) est décrit dans une **campagne** : un fichier de `configs/campaigns/`. Une seule commande la lance, en local comme sur le cluster.

Toutes les commandes se lancent **depuis la racine du dépôt** (le dossier qui contient `collaborative_learning/`). Le détail (options, résultats, ajout d'une méthode…) est dans [docs/GUIDE.md](docs/GUIDE.md).

## 1. Installation

```bash
# en local (Linux ou WSL2)
pip install -r requirements_gpu.txt      # GPU NVIDIA ; sinon requirements_cpu.txt
python -m pytest                         # tests, quelques minutes

# sur le cluster : pas d'installation, les modules sont chargés par ce script
source slurm/modules.sh
```

Détails (Windows / WSL2, cluster) : [INSTALL.md](INSTALL.md).

## 2. Premier essai (2 minutes)

```bash
python -m experiments.campaign quick_test --cpu
```

2 agents, 2 époques, 500 images : vérifie que tout tourne, jusqu'aux figures dans `results_quick_test/`. Les accuracy n'ont pas de sens.

## 3. Sur le cluster (SLURM)

```bash
source slurm/modules.sh
python -m experiments.campaign alpha_sweep --list          # nombre de tâches N et commande exacte, sans rien lancer
sbatch --array=0-169 slurm/campaign.sh alpha_sweep         # N tâches GPU en parallèle
squeue -u $USER                                            # suivi ; sorties dans slurm_logs/
```

- La fusion et les figures se font **toutes seules** quand le job array est fini.
- **Relancer la même commande** ne refait que les tâches manquantes ; une tâche coupée reprend à sa dernière époque.
- Une seule tâche : `sbatch --array=7 slurm/campaign.sh alpha_sweep`.

**Tout lancer dans l'ordre**, N calculé automatiquement :

```bash
source slurm/modules.sh
submit() {   # submit <campagne> [dépendance]
  N=$(python -m experiments.campaign "$1" --list | awk '/^TASKS/{print $2}')
  sbatch --parsable ${2:+--dependency=$2} --array=0-$((N-1)) slurm/campaign.sh "$1"
}
S=$(submit hyperparameter_search)          # 1. meilleurs lr / optimiseur de chaque dataset et architecture
A=$(submit alpha_sweep afterok:$S)         # 2. effet de α
L=$(submit lr_sweep afterany:$A)           # 3. effet du lr
# 4. regarder alpha_sweep, mettre le meilleur α dans tau_sweep, k_sweep, heterogeneous, puis :
T=$(submit tau_sweep)
K=$(submit k_sweep afterany:$T)
H=$(submit heterogeneous)
```

Les campagnes qui partagent un grand modèle (même dataset, arch et K) sont lancées l'une après l'autre (`afterany`) et doivent avoir les mêmes `epochs`. Pour analyser en local : `rsync -av cluster:MultiAgentsAI/results/ results/`.

## 4. En local

```bash
python -m experiments.campaign alpha_sweep --list          # tâches et dossiers
python -m experiments.campaign alpha_sweep                 # toutes les tâches l'une après l'autre, puis figures
python -m experiments.campaign alpha_sweep --task 7        # une seule tâche
python -m experiments.campaign alpha_sweep --merge         # fusion + figures (sans entraîner)
python -m experiments.plot_results alpha_sweep             # refaire seulement les figures

# version réduite pour tester une campagne sur une seule carte
python -m experiments.campaign k_sweep --set epochs=5 n_runs=1 train_subset=5000 out_dir=results_local
```

Sans campagne, une expérience seule (`--help` pour les options) :

```bash
python -m experiments.collaboration --dataset cifar10 --num-agents 3 --alphas 0 1 10 --n-runs 3
python -m experiments.collaboration --config configs/methods/global_interaction_kl.yaml --alphas 1
python -m experiments.collaboration --arch resnet32+mobilenetv2 --dataset cifar100 --alphas 0 1   # agents différents
python -m experiments.reference --dataset cifar10 --num-agents 3                                  # grand modèle seul
```

## 5. Dans Jupyter

```bash
jupyter notebook notebooks/          # Jupyter est installé avec les requirements
```

| Notebook | Ce qu'il montre |
|---|---|
| `effect_of_alpha.ipynb` | une méthode selon α : accuracy, gain sur les agents indépendants, diversité, calibration |
| `agents_vs_big_model.ipynb` | les K agents face au grand modèle seul, graine par graine |
| `compare_methods.ipynb` | toutes les méthodes d'une campagne côte à côte |

Régler la première cellule, puis *Run All* :

```python
RESULTS_ROOT = "../results"     # "../results_quick_test" pour quick_test
CAMPAIGN = "alpha_sweep"        # nom dans configs/campaigns/
DATASET = "cifar10"             # None = le premier de la campagne
K = 3
METHOD = "global_interaction"
```

## 6. Depuis Python (script ou notebook)

**Lire les résultats d'une campagne** (médiane [q05 ; q95] sur les graines) :

```python
import sys; sys.path.insert(0, "..")          # depuis notebooks/ ; inutile depuis la racine
from experiments.campaign import Campaign
from collaborative_learning.evaluation.plots import campaign_groups, rows_of, stats, entry_name

campaign = Campaign("alpha_sweep", {"out_dir": "../results"})
for (dataset, arch, k), group in campaign_groups(campaign).items():
    for e in group["entries"]:                    # un dossier = une méthode x un point de la grille
        med, q05, q95 = stats([r["accuracy_consensus"] for r in rows_of(e)])
        print(dataset, k, entry_name(e), f"{100 * med:.2f} [{100 * q05:.2f} ; {100 * q95:.2f}]")
```

**Poids de confiance w_k** par époque et par classe :

```python
from collaborative_learning.evaluation.plots import load_confidence, plot_agent_weights, plot_class_weights

e = group["entries"][0]
conf = load_confidence(e)                          # un dict par run : {"alpha10_seed0": {...}, ...}
run = next(iter(conf.values()))
run["val_per_class"][-1]                           # tableau (classes, K) à la dernière époque
plot_class_weights(e, "val")                       # figure classe x agent
plot_agent_weights(e, "val")                       # w_k de chaque agent selon l'époque
```

**Entraîner K agents** directement :

```python
from collaborative_learning import load_dataset
from collaborative_learning.training.runner import run_multi_experiment

data = load_dataset("fashion_mnist", val_size=5000)
res = run_multi_experiment(data, archs="resnet20", num_agents=3, loss_type="global_interaction_kl",
                           form="additive", alpha=1.0, epochs=5, lr=0.01, seed=0, confidence_weights=True)
print(res["accuracy_consensus"], res["accuracy_agents"])
```

**Refaire les figures** : `from collaborative_learning.evaluation.plots import plot_campaign; plot_campaign(Campaign("alpha_sweep"))`.

## 7. Campagnes et méthodes fournies

| Campagne | Question | Tâches |
|---|---|---|
| `quick_test` | le code tourne-t-il ? | 8 |
| `hyperparameter_search` | meilleurs lr, optimiseur, weight decay par dataset et architecture | 96 |
| `alpha_sweep` | effet de α | 170 |
| `lr_sweep` | effet du learning rate | 210 |
| `tau_sweep` | effet de la température τ, poids w_k | 210 |
| `k_sweep` | effet du nombre d'agents K | 280 |
| `heterogeneous` | agents d'architectures différentes (resnet32+mobilenetv2…) | 90 |

Méthodes (`configs/methods/`) : `global_interaction`, `global_interaction_additive`, `global_interaction_kl`, `global_interaction_kl_additive`, `dml` ; α = 0 donne toujours les agents indépendants. Formules : [docs/GUIDE.md](docs/GUIDE.md#méthode).

Nouvelle étude : copier une campagne et changer `methods`, `settings` ou `grid`. Essai sans toucher au fichier : `--set epochs=2 n_runs=1`.

## 8. Où sont les résultats

```
results/<campagne>/<dataset>/<arch>_<K>agents/
├── <méthode>/<point>/      ex. dml/alpha10 : results.json, history, poids w_k, checkpoints
└── figures/                figures (.png, .pdf) et summary.csv
results/reference/...       grand modèle seul, partagé entre campagnes
results/hyperparameter_search/<dataset>/<arch>/tuned_params.yaml
```

## Référence

J. Ketcha Kouakep, S. Uann, T. Carletti. *Multi-Agent Learning with Cooperation-Driven Optimization Dynamics*. arXiv:2609.16917, 2026.

```bibtex
@misc{ketchakouakep2026multiagent,
  title         = {Multi-Agent Learning with Cooperation-Driven Optimization Dynamics},
  author        = {Ketcha Kouakep, Jarod and Uann, Sreyvi and Carletti, Timoteo},
  year          = {2026},
  eprint        = {2609.16917},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MA},
  url           = {https://arxiv.org/abs/2609.16917}
}
```
