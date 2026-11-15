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
python -m pytest                        # ~1 min sur CPU
```

Python 3.10 à 3.13. Détails (Windows / WSL2, cluster) dans [INSTALL.md](INSTALL.md). Toutes les commandes se lancent depuis la racine du dépôt.

## Démarrage rapide

```bash
# test rapide sur CPU (quelques minutes)
python -m experiments.collaboration --cpu --train-subset 2000 --epochs 2 --alphas 1 --n-runs 1

# 3 ResNet-20 sur CIFAR-10, plusieurs α, comparés au grand modèle, 5 graines
python -m experiments.collaboration --dataset cifar10 --arch resnet20 --num-agents 3 \
    --alphas 0 1 10 --big-model true --n-runs 5

# baseline DML
python -m experiments.collaboration --dataset cifar10 --loss-type dml --loss-form additive --alphas 0 1 --tag dml

# figures (PNG + PDF) et tableau récapitulatif
python -m experiments.plot_results --dataset cifar10
```

Les réglages par défaut sont dans `configs/collaboration.json`. La ligne de commande a priorité sur le JSON (`--help` liste toutes les options).

## Expériences

| Commande | Rôle |
|---|---|
| `python -m experiments.collaboration` | K agents pour chaque α de `--alphas`, et le grand modèle seul avec `--big-model true` |
| `python -m experiments.hyperparameter_search` | meilleurs hyperparamètres d'**un** modèle (Optuna) ; écrit une config réutilisable pour les K agents |
| `python -m experiments.plot_results` | figures et `summary.csv` à partir des résultats, sans Jupyter |

Utiliser les hyperparamètres trouvés :

```bash
python -m experiments.hyperparameter_search --dataset cifar10 --arch resnet20
python -m experiments.collaboration --config results/hyperparameter_search/cifar10/resnet20/tuned_collaboration.json --num-agents 3
```

## Résultats

```
results/
├── collaboration/<dataset>/<arch>_<K>agents[_<tag>]/
│   ├── config.json            réglages et environnement (commit Git, versions)
│   ├── results.json           résumé par α, grand modèle, détail par graine
│   ├── test_probabilities/    probabilités de test de chaque run (.npy)
│   └── tasks/<tâche>/         une tâche : résultat, historique, checkpoint
├── hyperparameter_search/<dataset>/<arch>/   best.json, trials.csv, tuned_collaboration.json
└── comparisons/<dataset>/<arch>_<K>agents/   figures (.png, .pdf) et summary.csv
```

- **Un run = une graine** : `--n-runs 5` lance les graines 0 à 4.
- **Un dossier = un jeu de réglages.** Pour une variante, ajouter `--tag <nom>`. On peut ajouter des graines, des α ou le grand modèle à un dossier existant.
- **Reprise** : relancer la même commande saute les tâches finies et reprend une tâche coupée à sa dernière époque.

Figures produites par `plot_results` (médiane et bande q05–q95 sur les graines) :

| Figure | Contenu |
|---|---|
| `accuracy_vs_alpha` | accuracy du consensus et de l'agent médian selon α, grand modèle en référence |
| `paired_difference` | écart d'accuracy graine par graine avec le grand modèle (sinon α = 0) |
| `diversity_vs_alpha` | désaccord entre agents selon α |
| `learning_curves` | accuracy de validation par époque |
| `accuracy_vs_lr`, `diversity_vs_lr` | si une méthode a été lancée avec plusieurs lr (`collaboration_lr.sh`) |

Des notebooks d'analyse plus détaillés sont dans `notebooks/`.

## Cluster SLURM

```bash
sbatch slurm/hyperparameter_search.sh   # optionnel : recherche d'hyperparamètres
sbatch slurm/collaboration.sh           # toutes les méthodes, datasets, graines et α en parallèle
sbatch slurm/collaboration_lr.sh        # α fixé, plusieurs lr : méthodes collaboratives vs grand modèle à lr*
```

Les réglages (datasets, méthodes, `N_RUNS`, `ALPHAS`...) sont en tête de chaque script. Une tâche = un dataset × une méthode × une graine × (un α ou le grand modèle) ; les K agents d'un run restent dans la même tâche. Un job de fusion est soumis automatiquement : il écrit `results.json` puis les figures. Si une donnée change le nombre de tâches, mettre à jour `#SBATCH --array` (calcul en tête du script).

## Organisation du code

```
collaborative_learning/     package Python
├── core/                   MultiAgentModel, losses, consensus, callbacks
├── models/                 ResNet, MobileNet, AgentNetwork
├── data/                   jeux de données et pipelines tf.data
├── training/               optimiseurs, checkpoints, boucles d'expérience
├── tuning/                 recherche d'hyperparamètres (Optuna)
├── evaluation/             métriques, résumés, figures
└── utils/                  configs, GPU/CPU, logs
experiments/                scripts lancés avec python -m experiments.<nom>
configs/                    configs JSON par défaut
slurm/                      scripts sbatch
notebooks/                  analyses des résultats
tests/                      tests pytest
```

Losses, stratégies de consensus, architectures, optimiseurs et jeux de données sont enregistrés par nom (`@register_loss`, `@register_strategy`, `@register_model`, `@register_optimizer`, `@register_dataset`) : un nouvel élément enregistré est utilisable aussitôt dans le JSON et en ligne de commande.

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
