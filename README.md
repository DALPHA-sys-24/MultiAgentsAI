# Beyond Deep Mutual Learning

Apprentissage collaboratif de K réseaux (ResNet, MobileNet) entraînés ensemble avec une loss multi-agents, puis combinés par un consensus. Le projet mesure ce que la collaboration apporte face à :

- K agents **indépendants** (cross-entropy seule, vote majoritaire) ;
- la **Deep Mutual Learning** (DML, [Zhang et al., CVPR 2018](https://arxiv.org/abs/1706.00384)) ;
- un **grand modèle seul** de même architecture, élargi pour avoir autant de paramètres que les K agents réunis.

TensorFlow / Keras, testé sous TF 2.15 (Keras 2) et TF 2.21 (Keras 3), sur GPU ou CPU.

## Méthode

Chaque agent k a sa cross-entropy L_k = CE(y, y^k) et un terme d'interaction I_k qui dépend des prédictions des autres agents. Sa loss est :

| Forme | Loss de l'agent k |
|---|---|
| multiplicative (défaut) | L_k · (L_k + α · I_k) |
| additive | L_k + α · I_k |

- **α = 0** : agents indépendants, chacun entraîné en cross-entropy, consensus par vote majoritaire.
- **α > 0** : interaction choisie par `loss_type` (`global_interaction` par défaut, `leave_one_out`, `majority_loo`, `dml`...) et consensus par `strategy` (`simplicial` par défaut, `majority`, `vote`, `k_fold`).
- **DML** : `loss_type = dml`, forme additive, I_k = moyenne des KL(y^j ‖ y^k) sur les autres agents.

Les résultats sont donnés sur plusieurs graines, en **médiane [quantile 5 % ; quantile 95 %]**.

## Installation

```bash
git clone <url-du-dépôt> && cd <dépôt>
pip install -r requirements_gpu.txt     # GPU NVIDIA sous Linux / WSL2
pip install -r requirements_cpu.txt     # sinon
python -m pytest                        # quelques minutes sur CPU
```

Python 3.10 à 3.13. Détails (Windows / WSL2, cluster) dans [INSTALL.md](INSTALL.md). Toutes les commandes se lancent depuis la racine du dépôt.

## Démarrage rapide

Tout ce qu'on lance d'un coup (datasets, méthodes, valeurs de α ou de lr, grand modèle) est décrit dans une **campagne** : un petit fichier YAML de `configs/campaigns/`.

```bash
python -m experiments.campaign quick_test --cpu     # test complet en quelques minutes sur CPU
python -m experiments.campaign alpha_sweep --list   # tâches, dossiers et commande sbatch, sans rien lancer
python -m experiments.campaign alpha_sweep          # toute la campagne en local, puis les figures
```

Exemple de campagne (`configs/campaigns/lr_sweep.yaml`) :

```yaml
experiment: collaboration
datasets: [fashion_mnist, cifar10]
methods: [global_interaction, dml, global_interaction_additive, global_interaction_kl, global_interaction_kl_additive]   # configs/methods/<nom>.yaml
use_tuned_params: true                 # meilleurs paramètres de la recherche d'hyperparamètres
settings: {arch: resnet20, num_agents: 3, epochs: 200, n_runs: 5}
grid:                                  # produit cartésien : un dossier par point
  alpha: [1]
  lr: [0.1, 0.01, 0.001, 0.0001]
reference: true                        # grand modèle seul, au lr de base (lr*)
```

Agents hétérogènes : `arch: resnet32+mobilenetv2` (une architecture par agent, campagne `heterogeneous`). Nombre d'agents : `grid: {num_agents: [2, 3, 5, 8]}` (campagne `k_sweep`). Détails dans docs/GUIDE.md.

Pour une nouvelle étude, copier une campagne et changer `grid` ou `settings`. `--set clé=valeur` remplace un réglage sans toucher au fichier (`--set epochs=2 n_runs=1`).

## Méthodes disponibles

| Fichier `configs/methods/` | Loss de l'agent k | I_k |
|---|---|---|
| `global_interaction` | L_k · (L_k + α I_k) | CE(y^cons, y^k) + CE(y, y^cons) |
| `global_interaction_additive` | L_k + α I_k | CE(y^cons, y^k) + CE(y, y^cons) |
| `global_interaction_kl` | L_k · (L_k + α I_k) | KL(y^cons ‖ y^k) + CE(y, y^cons) |
| `global_interaction_kl_additive` | L_k + α I_k | KL(y^cons ‖ y^k) + CE(y, y^cons) |
| `dml` | L_k + α I_k | moyenne des KL(y^j ‖ y^k), j ≠ k (Zhang et al., 2018) |
| `independent` | L_k | 0 (utilisée automatiquement pour α = 0, à ne pas mettre dans `methods:`) |

## Configs

```
configs/
├── defaults.yaml              réglages communs (epochs, optimiseur, lr, augmentation...)
├── datasets/<dataset>.yaml    ce qui est propre à un jeu de données
├── methods/<méthode>.yaml     global_interaction[_additive], global_interaction_kl[_additive], dml, independent
├── collaboration.yaml         K agents      (defaults + cifar10 + global_interaction)
├── reference.yaml             grand modèle  (defaults + cifar10)
├── hyperparameter_search.yaml espace de recherche Optuna
└── campaigns/                 alpha_sweep, lr_sweep, tau_sweep, k_sweep, heterogeneous, hyperparameter_search, quick_test
```

Un fichier peut reprendre d'autres fichiers avec `defaults: [defaults, methods/dml]` ; la dernière valeur lue gagne. Pour un point de campagne, l'ordre est : `collaboration.yaml` < dataset < méthode < paramètres de la recherche < `settings` < valeurs de la grille. Une nouvelle méthode = un fichier dans `configs/methods/`.

Chaque expérience se lance aussi seule, avec les options en ligne de commande (`--help`) :

```bash
python -m experiments.collaboration --dataset cifar10 --num-agents 3 --alphas 0 1 10 --n-runs 5
python -m experiments.collaboration --config configs/methods/dml.yaml --alphas 1
python -m experiments.reference --dataset cifar10 --num-agents 3      # grand modèle seul
python -m experiments.hyperparameter_search --dataset cifar10 --arch resnet20
```

## Expériences

| Commande | Rôle |
|---|---|
| `experiments.campaign` | lance, fusionne et trace une campagne entière |
| `experiments.collaboration` | K agents pour chaque α de `--alphas` |
| `experiments.reference` | grand modèle seul, autant de paramètres que les K agents réunis |
| `experiments.hyperparameter_search` | meilleurs hyperparamètres d'**un** modèle (Optuna) ; écrit `tuned_params.yaml`, repris par les campagnes avec `use_tuned_params: true` |
| `experiments.plot_results` | figures et `summary.csv` d'une campagne, sans Jupyter |

## Résultats

```
results/
├── <campagne>/<dataset>/<arch>_<K>agents/
│   ├── <méthode>/<point>/          ex. dml/alpha1_lr0.1, independent/alpha0
│   │   ├── config.json             réglages et environnement (commit Git, versions)
│   │   ├── results.json            résumé (médiane, q05, q95) et détail par graine
│   │   ├── test_probabilities/     probabilités de test de chaque run (.npy)
│   │   ├── confidence_weights/     poids de confiance w_k par époque, au total et par classe (.npz)
│   │   └── tasks/<tâche>/          une tâche : résultat, historique, checkpoint
│   └── figures/                    figures (.png, .pdf) et summary.csv de la campagne
├── reference/<dataset>/<arch>_x<K>params_<optimiseur>_lr<lr>[_tuned]/   grand modèle, partagé entre campagnes
└── hyperparameter_search/<dataset>/<arch>/   best.json, trials.csv, tuned_params.yaml
```

- **Un run = une graine** : `n_runs: 5` lance les graines 0 à 4.
- **α = 0** : agents indépendants, entraînés une seule fois et partagés par toutes les méthodes de la campagne.
- **Grand modèle** : un par (dataset, arch, K), réutilisé par toutes les campagnes qui ont les mêmes réglages de base.
- **Reprise** : relancer la même campagne saute les tâches finies et reprend une tâche coupée à sa dernière époque. Ajouter des graines ou des valeurs à la grille n'ajoute que les tâches manquantes.

Figures (médiane et bande q05–q95 sur les graines), refaites à chaque fusion ou avec `python -m experiments.plot_results <campagne>` :

| Figure | Contenu |
|---|---|
| `accuracy_vs_<param>` | accuracy du consensus et de l'agent médian de chaque méthode selon α ou lr, grand modèle en référence |
| `diversity_vs_<param>` | désaccord entre agents |
| `paired_difference` | écart d'accuracy graine par graine avec le grand modèle (sinon avec les agents indépendants) |
| `learning_curves` | accuracy de validation par époque, un panneau par méthode |
| `<arch>_accuracy_vs_num_agents` | accuracy selon K (campagne `k_sweep`), dans `results/<campagne>/<dataset>/figures/` |
| `confidence_*` | poids de confiance w_k : par époque, selon τ, par agent et par classe (`confidence_weights: true`, voir docs/GUIDE.md) |

## Exemples d'exécution

Toutes les commandes se lancent depuis la racine du dépôt (le dossier qui contient `collaborative_learning/`).

### En local (GPU ou CPU)

```bash
python -m experiments.campaign quick_test --cpu        # test complet en quelques minutes (2 époques, 500 images)
python -m experiments.campaign alpha_sweep --list      # tâches numérotées et dossiers, sans rien lancer
python -m experiments.campaign alpha_sweep             # toutes les tâches l'une après l'autre, puis fusion et figures
python -m experiments.campaign alpha_sweep --task 7    # une seule tâche (reprend si elle a été coupée)
python -m experiments.campaign alpha_sweep --merge     # results.json de chaque dossier + figures
python -m experiments.campaign tau_sweep --set epochs=2 n_runs=1 train_subset=2000   # essai rapide d'une campagne
python -m experiments.plot_results alpha_sweep         # refaire seulement les figures
python -m pytest                                       # tests
```

Sans campagne, une expérience seule (options : `--help`) :

```bash
python -m experiments.collaboration --dataset fashion_mnist --num-agents 3 --alphas 0 1 10 --n-runs 3
python -m experiments.collaboration --config configs/methods/global_interaction_kl_additive.yaml --alphas 1
python -m experiments.reference --dataset cifar10 --num-agents 3          # grand modèle seul
python -m experiments.hyperparameter_search --dataset cifar10 --arch resnet20
```

### Sur le cluster (SLURM)

Sur le nœud de connexion, depuis la racine du dépôt (les modules sont chargés par `slurm/modules.sh`, PyYAML est nécessaire) :

```bash
source slurm/modules.sh
python -m experiments.campaign lr_sweep --list         # N tâches, commande sbatch exacte, données téléchargées
sbatch --array=0-209 slurm/campaign.sh lr_sweep        # une tâche GPU par (dataset, méthode, point, graine) + grand modèle
sbatch --array=7 slurm/campaign.sh lr_sweep            # relancer une seule tâche
squeue -u $USER                                        # suivi ; sorties dans slurm_logs/campaign_<job>_<tâche>.out/.err
```

La tâche 0 vérifie que `--array` couvre toutes les tâches et soumet la fusion (`slurm/merge_campaign.sh`), qui démarre quand tout le job array est fini : `results.json` de chaque dossier, puis les figures. Relancer la même commande ne refait que les tâches non terminées ; une tâche coupée reprend à sa dernière époque. Les K agents d'un run restent dans la même tâche.

Deux campagnes qui partagent le grand modèle (ex. `alpha_sweep` et `lr_sweep`) : les lancer l'une après l'autre.

```bash
JOB=$(sbatch --parsable --array=0-169 slurm/campaign.sh alpha_sweep)
sbatch --dependency=afterany:$JOB --array=0-209 slurm/campaign.sh lr_sweep
```

Ordre habituel : `hyperparameter_search` (optionnel, écrit `tuned_params.yaml`), puis les campagnes. Ensuite, rapatrier `results/` sur la machine locale (ex. `rsync -av cluster:MultiAgentsAI/results/ results/`) pour les notebooks.

### Dans Jupyter

```bash
pip install jupyterlab && jupyter lab notebooks/
```

| Notebook | Contenu |
|---|---|
| `effect_of_alpha.ipynb` | une méthode : accuracy, gain sur les agents indépendants, diversité, calibration selon α |
| `agents_vs_big_model.ipynb` | un point (méthode, α...) face au grand modèle seul, graine par graine |
| `compare_methods.ipynb` | toutes les méthodes et tous les points d'un groupe (dataset, K) |

Régler la première cellule puis *Run All* ; les figures vont dans `results/<campagne>/<dataset>/<arch>_<K>agents/figures/notebooks/` :

```python
RESULTS_ROOT = "../results"   # "../results_quick_test" pour quick_test
CAMPAIGN = "lr_sweep"         # nom dans configs/campaigns/
DATASET = "cifar10"           # None = le premier de la campagne
K = 3
METHOD = "global_interaction_kl"
```

### Fonctions importantes (Python, script ou notebook)

Lire les résultats d'une campagne :

```python
import sys; sys.path.insert(0, "..")        # depuis notebooks/ : racine du dépôt
from experiments.campaign import Campaign
from collaborative_learning.evaluation.plots import campaign_groups, rows_of, stats, entry_name

campaign = Campaign("alpha_sweep", {"out_dir": "../results"})
for (dataset, arch, k), group in campaign_groups(campaign).items():
    for e in group["entries"]:                        # un dossier = une méthode x un point de la grille
        med, q05, q95 = stats([r["accuracy_consensus"] for r in rows_of(e)])
        print(dataset, k, entry_name(e), f"{100 * med:.2f} [{100 * q05:.2f} ; {100 * q95:.2f}]")
```

Poids de confiance w_k par classe et par époque, et leurs figures :

```python
from collaborative_learning.evaluation.plots import load_confidence, plot_agent_weights, plot_class_weights

e = group["entries"][0]
conf = load_confidence(e)                             # {"alpha10_seed0": {...}, ...}
conf["alpha10_seed0"]["val_per_class"][-1]            # (classes, K) à la dernière époque
plot_class_weights(e, "val")                          # figure classe x agent, un panneau par graine
plot_agent_weights(e, "val")                          # w_k de chaque agent par époque
```

Entraîner K agents directement :

```python
from collaborative_learning import load_dataset
from collaborative_learning.training.runner import run_multi_experiment

data = load_dataset("fashion_mnist", val_size=5000)
res = run_multi_experiment(data, archs="resnet20", num_agents=3, loss_type="global_interaction_kl",
                           form="additive", alpha=1.0, temperature=1.0, epochs=5, lr=0.01, seed=0,
                           confidence_weights=True)
print(res["accuracy_consensus"], res["accuracy_agents"], res["history"]["val_w_max"])
```

Lire une config comme le font les scripts, et toutes les figures d'une campagne :

```python
from collaborative_learning.utils.config import CollaborationConfig, load_config
from collaborative_learning.evaluation.plots import plot_campaign

cfg = load_config(CollaborationConfig, "configs/collaboration.yaml", overrides={"epochs": 2})
plot_campaign(Campaign("lr_sweep"))                   # = python -m experiments.plot_results lr_sweep
```

Ajouter une loss : `@register_loss("mon_nom")` dans `collaborative_learning/core/losses.py`, puis un fichier `configs/methods/mon_nom.yaml` (voir docs/GUIDE.md).

## Organisation du code

```
collaborative_learning/     package Python
├── core/                   MultiAgentModel, losses, consensus, callbacks
├── models/                 ResNet, MobileNet, AgentNetwork
├── data/                   jeux de données et pipelines tf.data
├── training/               optimiseurs, checkpoints, boucles d'entraînement
├── tuning/                 recherche d'hyperparamètres (Optuna)
├── evaluation/             métriques, résumés, figures
└── utils/                  configs (dataclasses + YAML), GPU/CPU, logs
experiments/                scripts lancés avec python -m experiments.<nom>
configs/                    configs YAML et campagnes
slurm/                      campaign.sh, merge_campaign.sh
notebooks/                  analyses détaillées d'une campagne (Jupyter)
tests/                      tests pytest
```

Losses, stratégies de consensus, architectures, optimiseurs et jeux de données sont enregistrés par nom (`@register_loss`, `@register_strategy`, `@register_model`, `@register_optimizer`, `@register_dataset`) : un nouvel élément enregistré est utilisable aussitôt dans les YAML et en ligne de commande.

## Documentation

[docs/GUIDE.md](docs/GUIDE.md) : toutes les options, le contenu de `results.json`, l'ajout d'une loss ou d'une architecture, les optimiseurs, la recherche d'hyperparamètres et le cluster en détail.

## Référence

Ce code accompagne l'article :

J. Ketcha Kouakep, S. Uann, T. Carletti. *Multi-Agent Learning with Cooperation-Driven Optimization Dynamics*. arXiv:2609.16917, 2026. [arxiv.org/abs/2609.16917](https://arxiv.org/abs/2609.16917)

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
