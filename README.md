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
methods: [global_interaction, dml]     # configs/methods/<nom>.yaml
use_tuned_params: true                 # meilleurs paramètres de la recherche d'hyperparamètres
settings: {arch: resnet20, num_agents: 3, epochs: 200, n_runs: 5}
grid:                                  # produit cartésien : un dossier par point
  alpha: [1]
  lr: [0.1, 0.01, 0.001, 0.0001]
reference: true                        # grand modèle seul, au lr de base (lr*)
```

Pour une nouvelle étude, copier une campagne et changer `grid` ou `settings`. `--set clé=valeur` remplace un réglage sans toucher au fichier (`--set epochs=2 n_runs=1`).

## Configs

```
configs/
├── defaults.yaml              réglages communs (epochs, optimiseur, lr, augmentation...)
├── datasets/<dataset>.yaml    ce qui est propre à un jeu de données
├── methods/<méthode>.yaml     global_interaction, dml, independent (loss, forme, consensus)
├── collaboration.yaml         K agents      (defaults + cifar10 + global_interaction)
├── reference.yaml             grand modèle  (defaults + cifar10)
├── hyperparameter_search.yaml espace de recherche Optuna
└── campaigns/                 alpha_sweep, lr_sweep, tau_sweep, hyperparameter_search, quick_test
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
| `confidence_*` | poids de confiance w_k : par époque, selon τ, par agent et par classe (`confidence_weights: true`, voir docs/GUIDE.md) |

## Cluster SLURM

Un seul script pour toutes les campagnes :

```bash
python -m experiments.campaign lr_sweep --list      # affiche N et la commande exacte
sbatch --array=0-89 slurm/campaign.sh lr_sweep      # une tâche GPU par graine x point x méthode x dataset
```

La tâche 0 vérifie que `--array` couvre bien toutes les tâches et soumet la fusion (`slurm/merge_campaign.sh`), qui démarre quand tout le job array est fini : `results.json` de chaque dossier, puis les figures. Les K agents d'un run restent dans la même tâche. Les modules chargés sont dans `slurm/modules.sh` (PyYAML est nécessaire). Ordre habituel :

```bash
sbatch --array=0-15 slurm/campaign.sh hyperparameter_search   # optionnel
sbatch --array=0-59 slurm/campaign.sh alpha_sweep
sbatch --array=0-89 slurm/campaign.sh lr_sweep
```

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
