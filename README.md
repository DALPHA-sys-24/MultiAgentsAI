# Multi-agents avec ResNet / MobileNet

Apprentissage collaboratif de K agents (ResNet, MobileNet) avec les losses multi-agents de DALPHA, comparé à des agents indépendants, à la DML (Zhang et al., 2018) et à un grand modèle seul au même nombre de paramètres. TensorFlow / Keras, testé sous TF 2.15 (Keras 2, modules du cluster) et TF 2.21 (Keras 3).

## Organisation du code

```
.
├── collaborative_learning/     package Python (tout le code réutilisable ; s'appelait dml/)
│   ├── core/                   le système multi-agents
│   │   ├── multiagent.py       MultiAgentModel : K agents, un optimiseur par agent, train/test step
│   │   ├── losses.py           les losses (5 du projet + DML), registre @register_loss
│   │   ├── consensus.py        poids de confiance, 4 stratégies, registre @register_strategy
│   │   └── callbacks.py        AgentHistoryCallback, ConvergenceTracker
│   ├── models/                 architectures des agents, registre @register_model
│   │   ├── registry.py, layers.py
│   │   ├── resnet.py, mobilenet.py
│   │   └── agent.py            AgentNetwork = backbone + tête
│   ├── data/
│   │   ├── datasets.py         Fashion-MNIST, CIFAR-10/100, registre @register_dataset
│   │   └── pipelines.py        tf.data : mélange, augmentation, batchs
│   ├── training/
│   │   ├── optimizers.py       optimiseurs par nom (@register_optimizer), schedule, weight decay
│   │   ├── checkpoint.py       sauvegarde à chaque époque, reprise, meilleurs poids
│   │   └── runner.py           run_multi_experiment, run_single_reference, match_width, benchmark
│   ├── tuning/                 recherche d'hyperparamètres (Optuna)
│   │   ├── space.py            espace de recherche décrit en JSON
│   │   ├── pruning.py          arrêt précoce des essais mauvais
│   │   └── study.py            étude partagée entre jobs, export des résultats
│   ├── evaluation/
│   │   ├── metrics.py          accuracy, NLL, ECE, désaccord
│   │   └── reporting.py        médiane [q05 ; q95] sur les graines, tableaux
│   └── utils/
│       ├── config.py           configs des expériences (dataclasses), JSON, ligne de commande
│       ├── device.py           GPU si disponible, sinon CPU
│       └── log.py              journal console + fichiers de logs/
├── experiments/
│   ├── common.py               socle commun : tâches, dossier de résultats, reprise, fusion
│   ├── collaboration.py        K agents pour chaque alpha (+ grand modèle seul en option)
│   └── hyperparameter_search.py  meilleurs hyperparamètres d'UN modèle (Optuna)
├── configs/
│   ├── collaboration.json
│   └── hyperparameter_search.json
├── slurm/                      cluster : sbatch slurm/<expérience>.sh
│   ├── collaboration.sh        job array : dataset x méthode x run x (alpha | grand modèle)
│   ├── hyperparameter_search.sh  job array : dataset x worker Optuna
│   ├── merge_results.sh        fusion des résultats (soumise automatiquement)
│   └── helpers.sh              fonctions communes (vérif. de --array, soumission de la fusion)
├── notebooks/
│   ├── effect_of_alpha.ipynb       effet de alpha dans un run (plusieurs alphas)
│   ├── agents_vs_big_model.ipynb   K agents vs grand modèle, pour un alpha
│   ├── compare_methods.ipynb       plusieurs méthodes côte à côte (ta loss, DML, indépendants, grand modèle)
│   └── example_figures/
├── tests/                      tests pytest
├── results/                    résultats (ignoré par Git)
└── slurm_logs/                 sorties SLURM .out / .err (ignoré par Git)
```

### Dossier de résultats

Un dossier par expérience, puis par dataset, puis par variante :

```
results/
├── collaboration/
│   └── cifar10/
│       ├── resnet20_3agents/             ta méthode (+ grand modèle)
│       ├── resnet20_3agents_dml/         DML
│       └── resnet20_3agents_tuned/       avec les hyperparamètres de la recherche
├── hyperparameter_search/
│   └── cifar10/resnet20/                 best.json, trials.csv, tuned_collaboration.json
└── comparisons/
    └── cifar10/resnet20_3agents/         sorties de compare_methods.ipynb
```

Contenu d'un dossier `results/collaboration/<dataset>/<arch>_<K>agents[_<tag>]/` :

```
├── config.json             réglages, tailles des modèles, environnement (commit Git, versions)
├── results.json            résumé par alpha (médiane [q05 ; q95]), grand modèle, détail par run
├── test_probabilities/     probabilités de test (.npy) : alpha<a>_seed<s>.npy, big_model_seed<s>.npy
├── tasks/<tâche>/          une tâche : result.json, test_probabilities.npy, history.csv, checkpoint/
│                           (alpha0_seed0, alpha1_seed0, big_model_seed0, alpha0_seed1...)
├── figures/                figures enregistrées par les notebooks
└── logs/                   journal de chaque processus (all.txt, task_003.txt, merge.txt)
```

- **Un run = une graine.** `n_runs = 10` lance les graines 0 à 9 ; `results.json` donne la médiane et les quantiles 5 % / 95 % sur ces runs.
- **Même dossier = mêmes réglages.** Si le dossier existe avec d'autres réglages (lr, optimiseur, epochs...), le script s'arrête et le dit. Pour une variante, ajoute un suffixe : `--tag adamw` donne `results/collaboration/cifar10/resnet20_3agents_adamw/`. Peuvent s'ajouter à un dossier existant : des runs (`--n-runs 20`), des alphas (`--alphas 0 1 10 50`) et le grand modèle (`--big-model true`).
- **Reprise.** Une tâche terminée est sautée, une tâche interrompue reprend à sa dernière époque.
- **Fusion seule** (`--merge-only`) : relit les réglages enregistrés dans `config.json` du dossier ; `--run-dir <dossier> --merge-only` suffit.

**Toutes les commandes se lancent depuis la racine du dépôt** (le dossier qui contient `collaborative_learning/`), comme modules : `python -m experiments.collaboration`. Pas besoin d'installer le package ; `pip install -e .` est possible si tu veux importer `collaborative_learning` depuis n'importe quel dossier.

Où ajouter quoi :

| Ajouter | Où | Comment |
|---|---|---|
| une loss | `collaborative_learning/core/losses.py` | `@register_loss("nom")` |
| une stratégie de consensus | `collaborative_learning/core/consensus.py` | `@register_strategy("nom")` |
| une architecture | `collaborative_learning/models/<famille>.py` | `@register_model("nom")`, puis importer le module dans `collaborative_learning/models/__init__.py` |
| un optimiseur | `collaborative_learning/training/optimizers.py` | `@register_optimizer("nom")` |
| un jeu de données | `collaborative_learning/data/datasets.py` | `@register_dataset("nom")` |
| un réglage de config | `collaborative_learning/utils/config.py` | un champ `opt(...)` dans la dataclass (option `--nom` créée automatiquement) |
| un hyperparamètre à chercher | `configs/hyperparameter_search.json` | une entrée dans `search_space` (un réglage du modèle) |
| une méthode à comparer sur le cluster | `slurm/collaboration.sh` | une ligne dans `METHODS` |
| une expérience | `experiments/<nom>.py` + `configs/<nom>.json` | une sous-classe de `Experiment` (`experiments/common.py`) avec `tasks`, `run_task`, `merge` ; copier `collaboration.py` comme modèle, et `slurm/collaboration.sh` pour le cluster |

Le nom enregistré est accepté partout ensuite : dans `compile(...)`, dans le JSON et sur la ligne de commande.

## Changements du 2026-11-14 : noms lisibles, une seule expérience

`alpha_sweep` et `collab_vs_big` sont fusionnés dans **une seule expérience, `collaboration`** : un run simple est un balayage avec une seule valeur de alpha, et le grand modèle est une option. Les noms de fichiers et de dossiers sont en anglais, explicites.

| Avant | Maintenant |
|---|---|
| `python -m experiments.alpha_sweep --alphas 0 1 10` | `python -m experiments.collaboration --alphas 0 1 10` |
| `python -m experiments.collab_vs_big --alpha 1` | `python -m experiments.collaboration --alphas 1 --big-model true` |
| `python -m experiments.best_hyperparams` | `python -m experiments.hyperparameter_search` |
| `configs/alpha_sweep.json`, `configs/collab_vs_big.json` | `configs/collaboration.json` |
| `configs/best_hyperparams.json` | `configs/hyperparameter_search.json` |
| `slurm/alpha_sweep.sh`, `slurm/collab_vs_big.sh` | `slurm/collaboration.sh` |
| `slurm/best_hyperparams.sh` | `slurm/hyperparameter_search.sh` |
| `slurm/merge.sh`, `slurm/common.sh` | `slurm/merge_results.sh`, `slurm/helpers.sh` |
| `Logs/` | `slurm_logs/` |
| `runs/alpha_sweep_resnet20_cifar10_x5/` | `results/collaboration/cifar10/resnet20_5agents/` |
| `runs/collab_vs_big_resnet20_cifar10_x3_best_dml/` | `results/collaboration/cifar10/resnet20_3agents_dml_tuned/` |
| `runs/best_hyperparams_resnet20_cifar10/` | `results/hyperparameter_search/cifar10/resnet20/` |
| `best_collab_vs_big.json`, `best_alpha_sweep.json` | `tuned_collaboration.json` (tag `tuned`) |
| `USE_BEST=1` | `USE_TUNED_PARAMS=1` |
| `parts/<tâche>/`, `probs.npy` | `tasks/<tâche>/`, `test_probabilities.npy` |
| `probs/` | `test_probabilities/` |
| tâches `small_r0`, `big_r0`, `a1_r0`, `w0` | `alpha1_seed0`, `big_model_seed0`, `worker0` |
| `probs/small_agents_s0.npy`, `big_agent_s0.npy`, `agents_alpha1_s0.npy` | `test_probabilities/alpha1_seed0.npy`, `big_model_seed0.npy` |
| clé `big_agent` de `results.json` | `big_model` (et `vs_big_model` : K agents vs grand modèle pour chaque alpha) |
| `notebooks/resultats.ipynb` | `notebooks/agents_vs_big_model.ipynb` |
| `notebooks/resultats_sweep.ipynb` | `notebooks/effect_of_alpha.ipynb` |
| `notebooks/comparaison.ipynb` | `notebooks/compare_methods.ipynb` |
| `small_arch` (config) | `arch` |
| package `dml/` (`from dml import ...`) | `collaborative_learning/` (`from collaborative_learning import ...`) ; la loss `dml` garde son nom |

**Résultats déjà obtenus sur le cluster.** Les anciens dossiers `runs/` ne sont pas relus par le nouveau code. Pour réutiliser une recherche d'hyperparamètres déjà faite, il suffit de la déplacer et de réécrire la config K agents (une commande par dataset) :

```bash
mkdir -p results/hyperparameter_search/fashion_mnist
mv runs/best_hyperparams_resnet20_fashion_mnist results/hyperparameter_search/fashion_mnist/resnet20
python -m experiments.hyperparameter_search --run-dir results/hyperparameter_search/fashion_mnist/resnet20 --merge-only
# -> écrit tuned_collaboration.json ; l'étude Optuna (study.journal) est reprise telle quelle, de nouveaux essais s'y ajoutent
```

## Baseline DML (Zhang et al., 2018)

Loss `dml` : I_k = 1/(K−1) · Σ_{j≠k} KL(y^j ‖ y^k), les prédictions des autres agents étant des cibles fixes (`stop_gradient`).

| Variante | Réglages | Loss de l'agent k |
|---|---|---|
| DML classique (papier) | `--loss-type dml --loss-form additive`, α = 1 | CE(y, y^k) + 1/(K−1) Σ KL(y^j ‖ y^k) |
| DML dans la forme du projet | `--loss-type dml --loss-form multiplicative` | L_k · (L_k + α · KL) |

```bash
# DML classique (α = 1) et l'ensemble indépendant (α = 0, CE seule)
python -m experiments.collaboration --loss-type dml --loss-form additive --alphas 0 1 --tag dml
# variante multiplicative, plusieurs α
python -m experiments.collaboration --loss-type dml --loss-form multiplicative --alphas 0 1 10 --tag dml_mult
```

Sur le cluster : la DML est déjà dans la liste `METHODS` de `slurm/collaboration.sh` ; elle tourne dans le même job array que ta méthode.

Deux différences avec le papier :
- les K agents sont mis à jour en même temps à partir du même forward, alors que le papier les met à jour l'un après l'autre dans un pas (la mise à jour simultanée est celle de la plupart des implémentations publiques) ;
- le papier évalue chaque réseau seul : nos résultats le donnent aussi (chaque agent, agent médian, meilleur agent, moyenne simple), en plus du consensus.

## Notebooks

| Notebook | Lit | Montre |
|---|---|---|
| `effect_of_alpha.ipynb` | un dossier avec au moins 2 alphas | accuracy, gain, diversité, courbes et calibration selon alpha ; grand modèle en référence s'il existe |
| `agents_vs_big_model.ipynb` | un dossier avec `--big-model true`, et un alpha (`ALPHA`) | K agents vs grand modèle : tableau, accuracy par agent, courbes, calibration, diversité, accuracy par classe |
| `compare_methods.ipynb` | tous les dossiers `results/collaboration/<DATASET>/<ARCH>_<K>agents*` | ta méthode (chaque α), agents indépendants (α = 0), DML, grand modèle côte à côte |

`compare_methods.ipynb` :
- **Tableau** : consensus, moyenne simple, agent médian, meilleur agent, en médiane [q05 ; q95] sur les graines (`comparison.csv`).
- **Accuracy par méthode** : consensus (point plein) et agent médian (point creux), couleur par famille.
- **Différence appariée** : pour chaque graine, écart d'accuracy avec la référence (grand modèle, sinon agents indépendants) à graine égale, et nombre de graines gagnées.
- **Courbes d'apprentissage** : accuracy de validation par époque, médiane sur les graines.

Sorties dans `results/comparisons/<dataset>/<arch>_<K>agents/`. Pour que la comparaison soit juste, toutes les variantes doivent avoir le même dataset, la même architecture, le même K, les mêmes époques et les mêmes graines (`N_RUNS`).

## Fonctionnalités

- `--n-runs N` : graines 0 à N - 1. `steps_per_execution` (8 par défaut) : 8 pas d'entraînement par appel au GPU ; résultats identiques (vérifié sous TF 2.15 et 2.21).
- Le weight decay ne touche pas BatchNorm ni les biais (`"wd_exclude_norm_bias": false` pour l'ancien comportement).
- **Loss du grand modèle au choix** (`reference_loss`). `"ce"` (défaut) ou `"same_as_agents"` : le grand modèle est entraîné avec la même forme de loss que les agents et I_k = 0 (`L_k²` en multiplicatif, comme les agents à α = 0). Les deux camps ont alors la même échelle de gradient, ce qui enlève l'effet « lr effectif » vu entre SGD et Adam.
- **Forme de la loss** (`loss_form`) : `"multiplicative"` (défaut) ou `"additive"` (`L_k + α I_k`, qui redonne la CE à α = 0).
- **Reprise après coupure.** L'état complet (poids, BatchNorm, optimiseurs, historique) est sauvegardé à chaque époque. Relancer la même commande saute ce qui est terminé et reprend l'entraînement en cours à sa dernière époque. Les poids finaux et ceux de la meilleure `val_accuracy` sont gardés (`tasks/<tâche>/checkpoint/final_weights.npz`, `best_weights.npz`).
- **Meilleur modèle** : `"eval_on": "best"` évalue le test sur les poids de la meilleure époque de validation (défaut `"final"`, comme avant).
- **Config vérifiée.** Une clé inconnue dans le JSON arrête le script (`Clé(s) inconnue(s) dans la config : 'epoch' (voulais-tu dire 'epochs' ?)`), ainsi qu'un nom de loss, de stratégie ou d'optimiseur inconnu, ou une liste par agent de mauvaise longueur. Les clés qui commencent par `_` sont des commentaires.
- **Reproductibilité.** `config.json` contient une section `environment` : commit Git, branche, modifications non commitées (`git_dirty`), versions de Python / TF / Keras, machine, numéro de job SLURM, commande lancée.
- **Évaluation reproductible** : la stratégie `vote` tire l'agent avec un générateur remis à zéro à chaque `evaluate` / `predict`, donc la même accuracy à chaque évaluation.
- **Historique par époque** en CSV (`tasks/<tâche>/history.csv`), journaux dans `logs/`, arrêt immédiat si la loss devient NaN (`TerminateOnNaN`).
- **Une seule passe arrière ?** Testé : détacher les autres agents et dériver la somme des losses donne exactement les mêmes gradients, mais n'est pas plus rapide (≈ 500 ms par pas dans les deux cas, 5 × ResNet-20, CPU). TensorFlow ne parcourt déjà que le réseau de l'agent k pour le gradient de sa loss. Le `train_step` garde donc un gradient par agent.

## Configuration

Les valeurs par défaut sont dans `collaborative_learning/utils/config.py`, le fichier JSON les remplace, la ligne de commande remplace le JSON. Chaque champ a son option : `python -m experiments.collaboration --help` donne la liste. Les deux écritures marchent : `--n-runs 3` ou `--n_runs 3`.

Ajouter un réglage : ajouter un champ dans la dataclass (`mon_reglage: float = opt(0.5, "explication")`). L'option `--mon-reglage` et la vérification sont créées automatiquement ; il reste à le transmettre là où il sert.

| Clé | Valeurs | Effet |
|---|---|---|
| `loss_type`, `strategy` | noms enregistrés | loss et consensus |
| `loss_form` | `multiplicative`, `additive` | forme de la loss |
| `alphas` | liste de nombres | une valeur = run simple, plusieurs = balayage |
| `big_model` | `true`, `false` | ajoute le grand modèle seul |
| `reference_loss` | `ce`, `same_as_agents` | loss du grand modèle |
| `wd_exclude_norm_bias` | `true`, `false` | weight decay hors BatchNorm et biais |
| `eval_on` | `final`, `best` | poids évalués sur le test |
| `checkpoint_every` | entier | fréquence de sauvegarde de l'état de reprise |
| `optimizer_kwargs` | objet JSON | ex. `{"global_clipnorm": 1.0}` ou `{"use_ema": true}` |

## Ajouter une loss ou une stratégie

```python
# dans collaborative_learning/core/losses.py (ou un module importé avant l'entraînement)
@register_loss("ma_loss")
def ma_loss(ctx, outputs, log_p, raw_log_p, y):
    # outputs (B, K, C) probabilités ; log_p log-probabilités bornées à log(1e-7) ; y (B,)
    w_k = confidence_weights(outputs, ctx.temperature, raw_log_p)
    return (1.0 - w_k) * ce_soft(simplicial_proba(outputs, w_k), log_p)   # I_k, (B, K)
```

```python
# dans collaborative_learning/core/consensus.py
@register_strategy("meilleur_agent")
def meilleur_agent(outputs, ctx):
    w = confidence_weights(outputs, ctx.temperature)
    return pick_agent_labels(outputs, tf.argmax(w, axis=-1, output_type=tf.int32))   # labels (B,)
```

Le nom est alors accepté partout : `compile(loss_type="ma_loss")`, `"loss_type": "ma_loss"` dans le JSON, `--loss-type ma_loss`.

## Organisation des modèles

```
collaborative_learning/models/
├── __init__.py   # expose AgentNetwork, available_models, register_model
├── registry.py   # registre nom -> backbone
├── layers.py     # blocs communs (conv_bn, depthwise_bn)
├── resnet.py     # famille ResNet
├── mobilenet.py  # famille MobileNet
└── agent.py      # AgentNetwork = backbone + tête de classification
```

Seuls les noms enregistrés peuvent être chargés ; un nom inconnu lève une erreur qui liste les modèles disponibles.

Ajouter une architecture (exemple : un petit VGG) :

```python
# collaborative_learning/models/vgg.py
from .layers import conv_bn
from .registry import register_model
import tensorflow as tf

@register_model("vgg_small")
def vgg_small(inputs):
    x = inputs
    for filters, n in [(64, 1), (128, 1), (256, 2), (512, 2)]:
        for _ in range(n):
            x = conv_bn(x, filters)
        x = tf.keras.layers.MaxPooling2D()(x)
    return x
```

puis ajouter `vgg` à la ligne `from . import resnet, mobilenet` de `collaborative_learning/models/__init__.py`. Le backbone renvoie des caractéristiques (B, H, W, C) ; la tête est ajoutée par `AgentNetwork`.

## Installation

| Fichier | Pour |
|---|---|
| `requirements_cpu.txt` | Windows natif, macOS, Linux sans GPU |
| `requirements_gpu.txt` | GPU NVIDIA sous Linux ou WSL2 (CUDA/cuDNN installés par pip) |

Python 3.10 à 3.13. Commandes détaillées : voir `INSTALL.md`.

## GPU ou CPU

Les scripts utilisent automatiquement le GPU s'il est visible par TensorFlow, sinon le CPU (`--cpu` pour forcer le CPU). Il affiche au démarrage le matériel utilisé. Dans un notebook, appeler une fois `from collaborative_learning.utils.device import configure_device; configure_device()` avant de créer un modèle.

- **Windows** : TensorFlow >= 2.11 n'utilise pas le GPU nativement sous Windows. Il faut l'installer dans WSL2 (Ubuntu) : `pip install "tensorflow[and-cuda]"`, le pilote NVIDIA Windows suffit. Vérifier avec `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
- **RTX 4060 (8 Go)** : `--precision mixed_bfloat16` accélère l'entraînement et réduit la mémoire. La mémoire GPU est allouée au besoin, l'écran reste utilisable.

## Optimiseurs

Chaque optimiseur se choisit par son nom ; un nom inconnu lève une erreur qui liste les choix.

`adadelta`, `adafactor`, `adagrad`, `adam`, `adamax`, `adamw`, `ftrl`, `lamb`, `lion`, `muon`, `nadam`, `rmsprop`, `schedule_free_adamw`, `sgd` (ceux absents de la version de Keras installée sont ignorés ; `optimizers.available_optimizers()` donne la liste réelle). `sgd` utilise le momentum de Nesterov 0.9, comme avant.

```python
from collaborative_learning.training.optimizers import choix_optimiseur, build_agent_optimizers
opt = choix_optimiseur("adam", learning_rate=1e-3)

# un optimiseur différent par agent (lr et weight_decay : valeur commune ou liste)
opts = build_agent_optimizers(["sgd", "adam", "lion"], num_agents=3, lr=[0.1, 1e-3, 1e-4],
                              epochs=200, steps_per_epoch=390)
model.compile(optimizers_config=opts)          # ou simplement compile(["sgd", "adam", "lion"])
```

Dans l'expérience, `optimizer` et `lr` du fichier de config acceptent un nom / une valeur, ou une liste (un par agent). Le grand modèle prend le même optimiseur que les agents, sauf si `big_optimizer` / `big_lr` sont donnés (obligatoires quand les agents ont des optimiseurs différents). `optimizer_kwargs` passe des réglages supplémentaires (ex. `{"beta_1": 0.95}`).

```bash
python -m experiments.collaboration --big-model true --optimizer sgd adam adamw lion nadam --lr 0.1 1e-3 1e-3 1e-4 1e-3 \
    --big-optimizer sgd --big-lr 0.1
```

Ajouter un optimiseur : dans `collaborative_learning/training/optimizers.py`,

```python
@register_optimizer("mon_sgd")
def mon_sgd(**kwargs):
    return tf.keras.optimizers.SGD(momentum=0.95, **kwargs)
```

## Utilisation

### L'expérience `collaboration`

```bash
python -m experiments.collaboration                                   # configs/collaboration.json
# un run simple : un seul alpha
python -m experiments.collaboration --dataset fashion_mnist --num-agents 3 --alphas 1 --n-runs 5
# un balayage de alpha
python -m experiments.collaboration --alphas 0 1 10 --epochs 50
# K agents vs grand modèle (même nombre de paramètres)
python -m experiments.collaboration --alphas 1 --big-model true --n-runs 10
# test rapide (quelques minutes sur CPU)
python -m experiments.collaboration --cpu --train-subset 2000 --epochs 2 --warmup-epochs 1 --alphas 1 --n-runs 1
```

- **Agents** : `num_agents` agents `arch` entraînés ensemble avec `loss_type` pour chaque valeur de `alphas`, sur chaque graine. **alpha = 0** : pas d'interaction (`L_k²` en multiplicatif, CE en additif), consensus par **vote majoritaire**. **alpha > 0** : consensus avec la `strategy` de la config (par défaut `simplicial`).
- **Grand modèle** (`--big-model true`) : la même architecture élargie. La largeur est cherchée automatiquement pour avoir autant de paramètres entraînables que les K agents réunis (5 × ResNet-20 = 1,36 M ; grand ResNet-20 largeur ≈ 2,24, écart < 0,3 %). Il est entraîné seul, en cross-entropy (`reference_loss: "ce"`) ou avec la même forme de loss que les agents et I_k = 0 (`"same_as_agents"`), avec les mêmes données, augmentation, optimiseur, epochs et graine. Il ne dépend pas de alpha : un seul grand modèle par graine, comparé à chaque alpha.
- **Tâches** : `alpha<a>_seed<s>` (les K agents) et `big_model_seed<s>` (le grand modèle), indépendantes.
- **`results.json`** :
  - `summary` : pour chaque alpha, consensus, moyenne simple, agent médian, meilleur agent, désaccord, NLL/ECE, en médiane [q05 ; q95] ;
  - `runs` : une entrée par (alpha, graine), avec les accuracies de chaque agent et les historiques ;
  - `big_model` (si grand modèle) : résumé et une entrée par graine ;
  - `vs_big_model` : pour chaque alpha, grand modèle seul, consensus, moyenne simple, agent médian, meilleur agent côte à côte.
- Les tableaux de fin de run donnent les mêmes résumés.

Depuis Python :

```python
from collaborative_learning.data import load_dataset
from collaborative_learning.training.runner import benchmark_architectures

data = load_dataset("cifar10")
out = benchmark_architectures(
    data, configs=["resnet20", "resnet56", ["resnet32", "mobilenetv2"]],
    num_agents=2, n_runs=3, epochs=200,
    loss_type="global_interaction", strategy="simplicial",
)
print(out["summary"])
```

### Recherche des meilleurs hyperparamètres d'un modèle

La recherche porte sur **un seul modèle** (une architecture `arch`) sur un jeu de données, sans agents ni alpha. Les meilleurs paramètres trouvés (lr, weight decay, optimiseur...) servent ensuite à construire K agents de cette architecture.

```bash
pip install optuna                                       # en local, une fois (cluster : module Optuna/3.5.0-foss-2023a, déjà dans les scripts)
python -m experiments.hyperparameter_search              # configs/hyperparameter_search.json
python -m experiments.hyperparameter_search --dataset cifar10 --arch resnet32 --n-trials 50
python -m experiments.hyperparameter_search --cpu --train-subset 2000 --epochs 3 --n-trials 2 --n-workers 1   # test rapide
sbatch slurm/hyperparameter_search.sh                    # cluster : 8 GPU sur la même recherche

# ensuite : K agents avec les meilleurs paramètres du modèle seul
python -m experiments.collaboration --config results/hyperparameter_search/fashion_mnist/resnet20/tuned_collaboration.json --num-agents 3
```

J'utilise [Optuna](https://optuna.org) plutôt que Keras Tuner :
- il ne demande pas de réécrire le modèle sous forme de `HyperModel` ;
- son sampler TPE (bayésien) trouve de bonnes régions plus vite qu'une grille ou que le hasard ;
- il arrête tôt les essais mauvais (pruning médian ou Hyperband) ;
- plusieurs jobs SLURM peuvent travailler sur la même étude, grâce au fichier journal, qui marche aussi sur un disque réseau ;
- il calcule l'importance de chaque hyperparamètre.

**Ce qui est cherché.** `search_space` dans `configs/hyperparameter_search.json` contient une entrée par hyperparamètre du modèle (`lr`, `weight_decay`, `optimizer`, `batch_size`, `optimizer_kwargs`, `warmup_epochs`, `augment`...) :

```json
"search_space": {
  "lr":           {"type": "float", "low": 0.0001, "high": 0.3, "log": true},
  "weight_decay": {"type": "float", "low": 0.00001, "high": 0.01, "log": true},
  "batch_size":   {"type": "int", "low": 64, "high": 256, "step": 64},
  "optimizer":    {"type": "categorical", "choices": ["sgd", "adam", "adamw"]}
}
```

Les autres réglages (jeu de données, `arch`, `epochs` par essai...) sont fixés dans le même fichier.

**Loss du modèle** (`loss`) : `ce` (cross-entropy, défaut) ou `same_as_agents` (`L_k²` en multiplicatif, comme un agent à α = 0). Avec SGD, la loss multiplicative des agents a un gradient environ `2 L_k` fois plus grand que la CE : un lr trouvé en CE peut être trop grand pour les K agents (lr 0.1 fait diverger la loss multiplicative). `same_as_agents` donne un lr directement transposable aux agents ; avec Adam / AdamW la différence est faible.

**Critère.** `metric` vaut `val_accuracy` (défaut) ou `val_loss` du modèle. Chaque essai est noté sur la meilleure époque de **validation**, et le test n'est jamais utilisé pendant la recherche.

**Parallélisation.** Une tâche = un worker (`worker0`, `worker1`...) qui lance `n_trials` essais ; les `n_workers` workers partagent la même étude (`study.journal`), soit `n_workers × n_trials` essais au total. En local, les workers tournent l'un après l'autre.

**Résultats** dans `results/hyperparameter_search/<dataset>/<arch>/` :
- `best.json` : meilleur essai, ses paramètres, importance de chaque hyperparamètre ;
- `trials.csv` : tous les essais, avec leur état (terminé, arrêté tôt, échoué) ;
- `tuned_collaboration.json` : la config `configs/collaboration.json` avec l'architecture, le jeu de données et les meilleurs paramètres du modèle. K se choisit au lancement (`--num-agents`), le nombre d'époques final reste celui de `configs/collaboration.json`. Sortie dans `results/collaboration/<dataset>/<arch>_<K>agents_tuned/`.

Relancer la même commande ajoute des essais à la recherche existante.

### Sur le cluster

Un script sbatch par expérience. Les réglages sont en tête du script (datasets, `N_RUNS`, `ALPHAS`, `METHODS`, hyperparamètres) ; ceux qui n'y sont pas viennent de `configs/<expérience>.json`. Depuis la racine du dépôt :

```bash
sbatch slurm/hyperparameter_search.sh   # d'abord (optionnel) : 1 dataset x 8 workers = 8 tâches -> --array=0-7
sbatch slurm/collaboration.sh           # ensuite : 2 datasets x (ta méthode 5 x (3 alphas + grand modèle) + DML 5 x 3) = 70 tâches -> --array=0-69
```

- **Une tâche = un dataset × une méthode × un run × (un alpha ou le grand modèle).** L'indice `SLURM_ARRAY_TASK_ID` donne tout cela ; toutes les tâches tournent en parallèle. Les K agents d'un run sont dans la même tâche (loss commune).
- **Méthodes** : `METHODS` en tête de `collaboration.sh` liste les losses à comparer, une par ligne, au format `"loss_type:forme:tag:grand modèle"`. Par défaut : ta méthode avec le grand modèle (`global_interaction:multiplicative::1`) et la DML classique sans (`dml:additive:dml:0`) : le grand modèle ne dépend pas de la loss des agents, il est entraîné une seule fois. Chaque méthode a son dossier (le tag est ajouté au nom), donc `compare_methods.ipynb` les retrouve toutes. Ajouter une ligne (ex. `"dml:multiplicative:dml_mult:0"`) ajoute une méthode : penser à `--array`.
- **Run simple ou balayage** : `ALPHAS=(1)` pour un run simple, `ALPHAS=(0 1 10)` pour un balayage.
- **Taille du job array** : si tu changes `DATASETS`, `N_RUNS`, `ALPHAS` ou `METHODS`, mets à jour `#SBATCH --array=0-<N_TOTAL - 1>` (le calcul est en commentaire en tête du script). Un `--array` trop court est signalé dans le `.out` de la tâche 0, et les indices en trop ne font rien.
- **Hyperparamètres de la recherche, automatiquement** (`USE_TUNED_PARAMS=1`, défaut) : pour chaque dataset, `collaboration.sh` cherche `results/hyperparameter_search/<dataset>/<ARCH>/tuned_collaboration.json`. S'il existe, ses paramètres (optimiseur, lr, weight decay, batch...) remplacent `OPTIMIZER`, `LEARNING_RATE`, `BATCH_SIZE` du script, et les résultats vont dans `..._tuned/`. Sinon, les valeurs du script sont utilisées. La ligne `Paramètres:` du `.out` dit lequel des deux. `USE_TUNED_PARAMS=0` force les valeurs du script. Les époques, K, la loss et les alphas restent ceux du script.
- **Fusion** : la 1re tâche de chaque (dataset, méthode) soumet `slurm/merge_results.sh`, qui démarre quand tout le job array est fini et écrit `results.json` (`AUTO_MERGE=0 sbatch ...` pour ne pas le faire). Chaque tâche met aussi `results.json` à jour en finissant : les résultats partiels sont lisibles pendant la campagne. À la main : `python -m experiments.collaboration [mêmes options] --merge-only`.
- **Sorties SLURM** dans `slurm_logs/<expérience>_<job>_<tâche>.out / .err` ; résultats dans `results/<expérience>/<dataset>/...`.
- **Relancer** le même script ne refait que les tâches non terminées ; une tâche coupée reprend à sa dernière époque. Une seule tâche : `sbatch --array=7 slurm/collaboration.sh`.
- **Données** : télécharger une fois depuis le nœud de connexion si les nœuds de calcul n'ont pas Internet (`python -c "from collaborative_learning.data import download_dataset; download_dataset('cifar10')"`). Les tâches qui démarrent ensemble lisent le cache l'une après l'autre (verrou), sans se gêner.

### Avec Git

```bash
git pull                     # sur le cluster comme en local, au lieu de copier un zip
python -m pytest             # avant de lancer une campagne : ~1 min sur CPU
```

Lancer les campagnes depuis un code commité : `config.json` enregistre alors le commit exact (`git_dirty: false`). `results/`, `slurm_logs/` et les poids (`*.npz`, `*.npy`) sont ignorés par Git (`.gitignore`).

## Tests

```bash
python -m pytest             # depuis la racine du dépôt
```

Ils vérifient : la CE quand K = 1 (et `L_k²` avec `single_agent_loss="form"`), la forme additive, le gradient des losses contre une différence finie, l'ajout d'une loss au registre, l'évaluation reproductible de `vote`, le weight decay hors BatchNorm et biais, la sauvegarde et la reprise d'un entraînement interrompu, la lecture et la vérification des configs. Lancer les tests après chaque modification du code.

## Points d'attention

- **Échelle des losses multiplicatives.** `L_k * (L_k + alpha * (L_loc + L_glob))` vaut environ 16 au départ sur 10 classes, puis tend vers 0 plus vite que la CE. Avec SGD à lr 0.1, le pas effectif est donc plus grand au début et plus petit à la fin que pour la référence. Le warmup (5 epochs) limite le premier effet ; si les agents décrochent en fin d'entraînement, c'est la piste à regarder.
- **Weight decay.** Le `train_step` n'ajoute pas les régularisations des couches (`self.losses`), d'où le weight decay porté par l'optimiseur. Il ne s'applique plus aux paramètres de BatchNorm ni aux biais (`wd_exclude_norm_bias: false` pour l'ancien comportement).
- **`ConvergenceTracker`** : sans `reference_weights`, il doit garder les poids de chaque epoch jusqu'à la fin pour mesurer la distance aux poids finaux. Pour les gros modèles, utiliser `ConvergenceTracker(snapshot_dir="auto")` (sur disque) ou fournir `reference_weights`.
- **Téléchargement des données.** Keras télécharge CIFAR-10 depuis cs.toronto.edu au premier appel. Sur un nœud de cluster sans internet, lancer `load_dataset` une fois sur le nœud de connexion (cache dans `~/.keras/datasets`).
