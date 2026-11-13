# Multi-agents avec ResNet / MobileNet (v2)

Apprentissage collaboratif de K agents (ResNet, MobileNet) avec les losses multi-agents de DALPHA, comparé à des agents seuls ou à un grand agent au même nombre de paramètres. TensorFlow / Keras, testé sous TF 2.15 (Keras 2, modules du cluster) et TF 2.21 (Keras 3).

## Organisation du code

```
.
├── dml/                        package Python (tout le code réutilisable)
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
│       └── log.py              journal console + <run_dir>/log.txt
├── experiments/                un script par expérience
│   ├── common.py               socle commun : tâches, dossier de run, reprise, fusion des résultats
│   ├── collab_vs_big.py        K petits agents vs 1 grand agent
│   ├── alpha_sweep.py          balayage de alpha
│   └── best_hyperparams.py     recherche des meilleurs hyperparamètres (Optuna)
├── configs/                    un JSON par expérience (collab_vs_big, alpha_sweep, best_hyperparams)
├── slurm/                      cluster (sbatch slurm/<expérience>.sh)
│   ├── collab_vs_big.sh        job array : dataset x run x (petits | grand)
│   ├── alpha_sweep.sh          job array : dataset x run x alpha
│   ├── best_hyperparams.sh     job array : dataset x worker Optuna (un seul modèle)
│   ├── merge.sh                fusion des résultats (soumise automatiquement)
│   └── common.sh               fonctions communes (vérif. de --array, soumission de la fusion)
├── notebooks/                  analyse : resultats.ipynb (un collab_vs_big), resultats_sweep.ipynb (un sweep),
│                               comparaison.ipynb (plusieurs méthodes côte à côte)
├── tests/                      tests pytest
├── runs/                       sorties des runs (ignoré par Git)
└── Logs/                       sorties SLURM .out / .err (ignoré par Git)
```

### Dossier d'un run

Chaque expérience écrit dans `runs/<expérience>_<arch>_<dataset>_x<K>[_<tag>]/`, par exemple `runs/collab_vs_big_resnet20_fashion_mnist_x3/` :

```
runs/collab_vs_big_resnet20_fashion_mnist_x3/
├── config.json          réglages, tailles des modèles, environnement (commit Git, versions)
├── results.json         résumé sur les runs (médiane [q05 ; q95]) et détail par run
├── probs/               probabilités de test de chaque run (.npy), lues par les notebooks
├── parts/<tâche>/       une tâche : result.json, probs.npy, history.csv, checkpoint/
│                        (collab_vs_big : small_r0, big_r0, small_r1... ; alpha_sweep : a0_r0, a1_r0...)
└── logs/                journal de chaque processus (all.txt, task_003.txt, merge.txt)
```

- **Un run = une graine.** `n_runs = 10` lance les graines 0 à 9 ; `results.json` donne la médiane et les quantiles 5 % / 95 % sur ces runs.
- **Même dossier = mêmes réglages.** Si le dossier existe avec d'autres réglages (lr, optimiseur, epochs...), le script s'arrête et le dit. Pour une variante, ajoute un suffixe : `--tag adamw` donne `runs/collab_vs_big_resnet20_fashion_mnist_x3_adamw/`. Seuls `n_runs` (et `alphas` pour le balayage) peuvent grandir : relancer avec `--n-runs 20` ajoute les graines 10 à 19 au même dossier.
- **Reprise.** Une tâche terminée est sautée, une tâche interrompue reprend à sa dernière époque.

**Toutes les commandes se lancent depuis la racine du dépôt** (le dossier qui contient `dml/`). Les scripts se lancent comme modules : `python -m experiments.alpha_sweep`. Pas besoin d'installer le package ; `pip install -e .` est possible si tu veux importer `dml` depuis n'importe quel dossier.

Où ajouter quoi :

| Ajouter | Où | Comment |
|---|---|---|
| une loss | `dml/core/losses.py` | `@register_loss("nom")` |
| une stratégie de consensus | `dml/core/consensus.py` | `@register_strategy("nom")` |
| une architecture | `dml/models/<famille>.py` | `@register_model("nom")`, puis importer le module dans `dml/models/__init__.py` |
| un optimiseur | `dml/training/optimizers.py` | `@register_optimizer("nom")` |
| un jeu de données | `dml/data/datasets.py` | `@register_dataset("nom")` |
| un réglage de config | `dml/utils/config.py` | un champ `opt(...)` dans la dataclass (option `--nom` créée automatiquement) |
| un hyperparamètre à chercher | `configs/best_hyperparams.json` | une entrée dans `search_space` (un réglage du modèle) |
| une expérience | `experiments/<nom>.py` + `configs/<nom>.json` | une sous-classe de `Experiment` (`experiments/common.py`) avec `tasks`, `run_task`, `merge` ; copier `alpha_sweep.py` comme modèle. plus `slurm/<nom>.sh` copié de `slurm/alpha_sweep.sh` |

Le nom enregistré est accepté partout ensuite : dans `compile(...)`, dans le JSON et sur la ligne de commande.

## Baseline DML (Zhang et al., 2018)

Loss `dml` : I_k = 1/(K−1) · Σ_{j≠k} KL(y^j ‖ y^k), les prédictions des autres agents étant des cibles fixes (`stop_gradient`).

| Variante | Réglages | Loss de l'agent k |
|---|---|---|
| DML classique (papier) | `--loss-type dml --loss-form additive`, α = 1 | CE(y, y^k) + 1/(K−1) Σ KL(y^j ‖ y^k) |
| DML dans la forme du projet | `--loss-type dml --loss-form multiplicative` | L_k · (L_k + α · KL) |

```bash
# DML classique (α = 1) et l'ensemble indépendant (α = 0, CE seule) dans le même sweep
python -m experiments.alpha_sweep --loss-type dml --loss-form additive --alphas 0 1 --tag dml
# variante multiplicative, plusieurs α
python -m experiments.alpha_sweep --loss-type dml --loss-form multiplicative --alphas 0 1 10 --tag dml_mult
# K petits agents DML vs grand agent
python -m experiments.collab_vs_big --loss-type dml --loss-form additive --alpha 1 --tag dml
```

Sur le cluster : la DML est déjà dans la liste `METHODS` de `slurm/alpha_sweep.sh` et `slurm/collab_vs_big.sh` ; elle tourne dans le même job array que ta méthode.

Deux différences avec le papier :
- les K agents sont mis à jour en même temps à partir du même forward, alors que le papier les met à jour l'un après l'autre dans un pas (la mise à jour simultanée est celle de la plupart des implémentations publiques) ;
- le papier évalue chaque réseau seul : nos résultats le donnent aussi (chaque agent, agent médian, meilleur agent, moyenne simple), en plus du consensus.

## Comparer les méthodes (`notebooks/comparaison.ipynb`)

Le notebook lit tous les dossiers `runs/alpha_sweep_<arch>_<dataset>_x<K>*` et `runs/collab_vs_big_<arch>_<dataset>_x<K>*` (avec ou sans tag) et compare ta méthode (chaque α), les agents indépendants (α = 0), la DML et le grand agent. Régler `DATASET`, `ARCH`, `K` en tête, ou donner la liste `RUN_DIRS`.

- **Tableau** : consensus, moyenne simple, agent médian, meilleur agent, en médiane [q05 ; q95] sur les graines (`comparaison.csv`).
- **Accuracy par méthode** : consensus (point plein) et agent médian (point creux), couleur par famille.
- **Différence appariée** : pour chaque graine, écart d'accuracy avec la référence (grand agent, sinon agents indépendants) à graine égale, et nombre de graines gagnées.
- **Courbes d'apprentissage** : accuracy de validation par époque, médiane sur les graines.

Sorties dans `runs/comparaison_<arch>_<dataset>_x<K>/`. Pour que la comparaison soit juste, toutes les variantes doivent avoir le même dataset, la même architecture, le même K, les mêmes époques et les mêmes graines (`N_RUNS`).

## Nouveautés : `runs/`, `n_runs`, job arrays

- `--n-runs N` remplace `--seeds` (graines 0 à N - 1).
- Les sorties vont dans `runs/<expérience>_<arch>_<dataset>_x<K>[_<tag>]/` au lieu de `results/<date>_<nom>/` (voir « Dossier d'un run »).
- Sur le cluster : `sbatch slurm/<expérience>.sh`. Chaque dataset et chaque run (graine) devient une tâche GPU du job array, et toutes tournent en parallèle. Les K agents d'un run restent dans la même tâche, car ils sont entraînés ensemble avec une loss commune. Un job de fusion écrit `results.json` à la fin. `merge_sweep.py` est supprimé.
- `steps_per_execution` (8 par défaut) : 8 pas d'entraînement par appel au GPU, moins d'aller-retours CPU ↔ GPU. Les résultats sont identiques (vérifié sous TF 2.15 et 2.21 : mêmes accuracies à chaque époque).

## Nouveautés du 2026-11-12 (arborescence)

**Nouvelle arborescence** (voir plus haut). Les commandes changent : `python run_experiment.py` devient `python -m experiments.collab_vs_big`, `python run_alpha_sweep.py` devient `python -m experiments.alpha_sweep`, `sbatch soumission_*.sh` devient `sbatch slurm/<expérience>.sh`. Les imports deviennent `from dml.core.multiagent import MultiAgentModel` (ou `from dml import MultiAgentModel`). Les noms des classes et des fonctions ne changent pas.

Avec les réglages par défaut, l'entraînement est **identique au bit près** à la version précédente (vérifié en mode déterministe sous TF 2.15 : mêmes losses à chaque époque et mêmes accuracies), sauf le weight decay, qui ne touche plus BatchNorm ni les biais. Pour retrouver exactement l'ancien comportement : `"wd_exclude_norm_bias": false`.

- **Loss de la référence au choix** (`reference_loss`). `"ce"` (défaut, comme avant) ou `"same_as_agents"` : le grand agent est entraîné avec la même forme de loss que les petits agents et I_k = 0 (`L_k²` en multiplicatif, comme les agents à α = 0). Les deux camps ont alors la même échelle de gradient, ce qui enlève l'effet « lr effectif » vu entre SGD et Adam.
- **Forme de la loss** (`loss_form`) : `"multiplicative"` (défaut) ou `"additive"` (`L_k + α I_k`, qui redonne la CE à α = 0).
- **Reprise après coupure.** L'état complet (poids, BatchNorm, optimiseurs, historique) est sauvegardé à chaque époque. Relancer la même commande saute ce qui est terminé et reprend l'entraînement en cours à sa dernière époque. Les poids finaux et ceux de la meilleure `val_accuracy` sont gardés (`parts/<tâche>/checkpoint/final_weights.npz`, `best_weights.npz`).
- **Meilleur modèle** : `"eval_on": "best"` évalue le test sur les poids de la meilleure époque de validation (défaut `"final"`, comme avant).
- **Config vérifiée.** Une clé inconnue dans le JSON arrête le script (`Clé(s) inconnue(s) dans la config : 'epoch' (voulais-tu dire 'epochs' ?)`), ainsi qu'un nom de loss, de stratégie ou d'optimiseur inconnu, ou une liste par agent de mauvaise longueur. Les clés qui commencent par `_` sont des commentaires.
- **Reproductibilité.** `config.json` contient une section `environment` : commit Git, branche, modifications non commitées (`git_dirty`), versions de Python / TF / Keras, machine, numéro de job SLURM, commande lancée.
- **Évaluation reproductible** : la stratégie `vote` tire l'agent avec un générateur remis à zéro à chaque `evaluate` / `predict`, donc la même accuracy à chaque évaluation.
- **Historique par époque** en CSV (`parts/<tâche>/history.csv`), journaux dans `logs/`, arrêt immédiat si la loss devient NaN (`TerminateOnNaN`).
- **Une seule passe arrière ?** Testé : détacher les autres agents et dériver la somme des losses donne exactement les mêmes gradients, mais n'est pas plus rapide (≈ 500 ms par pas dans les deux cas, 5 × ResNet-20, CPU). TensorFlow ne parcourt déjà que le réseau de l'agent k pour le gradient de sa loss. Le `train_step` garde donc un gradient par agent.

## Configuration

Les valeurs par défaut sont dans `dml/utils/config.py`, le fichier JSON les remplace, la ligne de commande remplace le JSON. Chaque champ a son option : `python -m experiments.alpha_sweep --help` donne la liste. Les deux écritures marchent : `--n-runs 3` ou `--n_runs 3`.

Ajouter un réglage : ajouter un champ dans la dataclass (`mon_reglage: float = opt(0.5, "explication")`). L'option `--mon-reglage` et la vérification sont créées automatiquement ; il reste à le transmettre là où il sert.

| Clé | Valeurs | Effet |
|---|---|---|
| `loss_type`, `strategy` | noms enregistrés | loss et consensus |
| `loss_form` | `multiplicative`, `additive` | forme de la loss |
| `reference_loss` (collab_vs_big) | `ce`, `same_as_agents` | loss du grand agent |
| `wd_exclude_norm_bias` | `true`, `false` | weight decay hors BatchNorm et biais |
| `eval_on` | `final`, `best` | poids évalués sur le test |
| `checkpoint_every` | entier | fréquence de sauvegarde de l'état de reprise |
| `optimizer_kwargs` | objet JSON | ex. `{"global_clipnorm": 1.0}` ou `{"use_ema": true}` |

## Ajouter une loss ou une stratégie

```python
# dans dml/core/losses.py (ou un module importé avant l'entraînement)
@register_loss("ma_loss")
def ma_loss(ctx, outputs, log_p, raw_log_p, y):
    # outputs (B, K, C) probabilités ; log_p log-probabilités bornées à log(1e-7) ; y (B,)
    w_k = confidence_weights(outputs, ctx.temperature, raw_log_p)
    return (1.0 - w_k) * ce_soft(simplicial_proba(outputs, w_k), log_p)   # I_k, (B, K)
```

```python
# dans dml/core/consensus.py
@register_strategy("meilleur_agent")
def meilleur_agent(outputs, ctx):
    w = confidence_weights(outputs, ctx.temperature)
    return pick_agent_labels(outputs, tf.argmax(w, axis=-1, output_type=tf.int32))   # labels (B,)
```

Le nom est alors accepté partout : `compile(loss_type="ma_loss")`, `"loss_type": "ma_loss"` dans le JSON, `--loss-type ma_loss`.

## Organisation des modèles

```
dml/models/
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
# dml/models/vgg.py
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

puis ajouter `vgg` à la ligne `from . import resnet, mobilenet` de `dml/models/__init__.py`. Le backbone renvoie des caractéristiques (B, H, W, C) ; la tête est ajoutée par `AgentNetwork`.

## Installation

| Fichier | Pour |
|---|---|
| `requirements_cpu.txt` | Windows natif, macOS, Linux sans GPU |
| `requirements_gpu.txt` | GPU NVIDIA sous Linux ou WSL2 (CUDA/cuDNN installés par pip) |

Python 3.10 à 3.13. Commandes détaillées : voir `INSTALL.md`.

## GPU ou CPU

Les scripts utilisent automatiquement le GPU s'il est visible par TensorFlow, sinon le CPU (`--cpu` pour forcer le CPU). Il affiche au démarrage le matériel utilisé. Dans un notebook, appeler une fois `from dml.utils.device import configure_device; configure_device()` avant de créer un modèle.

- **Windows** : TensorFlow >= 2.11 n'utilise pas le GPU nativement sous Windows. Il faut l'installer dans WSL2 (Ubuntu) : `pip install "tensorflow[and-cuda]"`, le pilote NVIDIA Windows suffit. Vérifier avec `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
- **RTX 4060 (8 Go)** : `--precision mixed_bfloat16` accélère l'entraînement et réduit la mémoire. La mémoire GPU est allouée au besoin, l'écran reste utilisable.

## Optimiseurs

Chaque optimiseur se choisit par son nom ; un nom inconnu lève une erreur qui liste les choix.

`adadelta`, `adafactor`, `adagrad`, `adam`, `adamax`, `adamw`, `ftrl`, `lamb`, `lion`, `muon`, `nadam`, `rmsprop`, `schedule_free_adamw`, `sgd` (ceux absents de la version de Keras installée sont ignorés ; `optimizers.available_optimizers()` donne la liste réelle). `sgd` utilise le momentum de Nesterov 0.9, comme avant.

```python
from dml.training.optimizers import choix_optimiseur, build_agent_optimizers
opt = choix_optimiseur("adam", learning_rate=1e-3)

# un optimiseur différent par agent (lr et weight_decay : valeur commune ou liste)
opts = build_agent_optimizers(["sgd", "adam", "lion"], num_agents=3, lr=[0.1, 1e-3, 1e-4],
                              epochs=200, steps_per_epoch=390)
model.compile(optimizers_config=opts)          # ou simplement compile(["sgd", "adam", "lion"])
```

Dans l'expérience, `optimizer` et `lr` du fichier de config acceptent un nom / une valeur, ou une liste (un par petit agent). Le grand agent prend le même optimiseur que les petits, sauf si `big_optimizer` / `big_lr` sont donnés (obligatoires quand les petits ont des optimiseurs différents). `optimizer_kwargs` passe des réglages supplémentaires (ex. `{"beta_1": 0.95}`).

```bash
python -m experiments.collab_vs_big --optimizer sgd adam adamw lion nadam --lr 0.1 1e-3 1e-3 1e-4 1e-3 \
    --big-optimizer sgd --big-lr 0.1
```

Ajouter un optimiseur : dans `dml/training/optimizers.py`,

```python
@register_optimizer("mon_sgd")
def mon_sgd(**kwargs):
    return tf.keras.optimizers.SGD(momentum=0.95, **kwargs)
```

## Utilisation

### Expérience : 5 petits agents vs 1 grand agent

```bash
python -m experiments.collab_vs_big --config configs/collab_vs_big.json
# surcharger un réglage sans toucher au fichier
python -m experiments.collab_vs_big --config configs/collab_vs_big.json --dataset fashion_mnist --epochs 50
# test rapide (quelques minutes sur CPU)
python -m experiments.collab_vs_big --cpu --train-subset 2000 --epochs 2 --n-runs 1
# 10 runs pour la médiane et les quantiles
python -m experiments.collab_vs_big --dataset fashion_mnist --num-agents 3 --optimizer sgd --lr 0.01 --epochs 20 --batch-size 128 --n-runs 10
```

- **Petits agents** : `num_agents` agents `small_arch` entraînés ensemble avec `loss_type` + `strategy` (par défaut 5 × resnet20, `global_interaction` + `simplicial`).
- **Grand agent** : la même architecture élargie. La largeur est cherchée automatiquement pour avoir autant de paramètres entraînables que les petits réunis (5 × ResNet-20 = 1,36 M ; grand ResNet-20 largeur ≈ 2,24 → 1,36 M, écart < 0,3 %). Il est entraîné seul, en cross-entropy (`reference_loss: "ce"`) ou avec la même forme de loss que les petits agents et I_k = 0 (`reference_loss: "same_as_agents"`, soit `L_k²` en multiplicatif), avec les mêmes données, augmentation, optimiseur, epochs et graine.
- **Tâches** : `small_r<i>` (les K petits agents du run i) et `big_r<i>` (le grand agent du run i), soit 2 × `n_runs` tâches indépendantes.
- **Résultats** dans `runs/collab_vs_big_<arch>_<dataset>_x<K>/` : `config.json` (config utilisée, matériel, nombres de paramètres, largeur du grand agent, commit Git et versions), `results.json` (par run : accuracy, NLL, ECE de chaque petit agent, du consensus et du grand agent, historiques par epoch ; résumé par médiane et quantiles 5 % / 95 % sur les runs), `probs/small_agents_s<graine>.npy` et `big_agent_s<graine>.npy`, `parts/` (historique CSV et poids de chaque tâche). `results.json` est réécrit après chaque run complet.
- Un tableau final compare, en médiane [quantile 5 % ; quantile 95 %] sur les graines : grand agent seul, consensus des petits, moyenne simple des petits, petit agent médian, meilleur petit agent. Le notebook `notebooks/resultats.ipynb` affiche les mêmes statistiques en tableaux et figures.

### Expérience : impact de la collaboration (balayage de alpha)

```bash
python -m experiments.alpha_sweep                                  # configs/alpha_sweep.json
python -m experiments.alpha_sweep --alphas 0 1 10 --epochs 50      # surcharger un réglage
python -m experiments.alpha_sweep --cpu --train-subset 2000 --epochs 2 --warmup-epochs 1 --alphas 0 1 --n-runs 1   # test rapide
```

- `num_agents` agents `arch` entraînés avec `loss_type` pour chaque valeur de `alphas` (par défaut 0, 0.5, 1, 10, 50, 100), sur chaque graine.
- **alpha = 0** : pas d'interaction (loss `L_k²`, agents indépendants), consensus par **vote majoritaire**. **alpha > 0** : consensus avec la `strategy` du fichier de config.
- **Tâches** : une par (alpha, run), nommées `a<alpha>_r<i>`.
- **Résultats** dans `runs/alpha_sweep_<arch>_<dataset>_x<K>/` : `config.json`, `results.json` (par alpha et par graine : accuracy du consensus, de la moyenne simple, de chaque agent, NLL/ECE, désaccord entre agents, historiques ; résumé par alpha en médiane et quantiles 5 % / 95 %), `probs/agents_alpha<a>_s<graine>.npy`.
- Le tableau final donne, pour chaque alpha : consensus, moyenne simple, agent médian, meilleur agent et désaccord moyen entre agents (diversité).

Depuis Python :

```python
from dml.data import load_dataset
from dml.training.runner import benchmark_architectures

data = load_dataset("cifar10")
out = benchmark_architectures(
    data, configs=["resnet20", "resnet56", ["resnet32", "mobilenetv2"]],
    num_agents=2, n_runs=3, epochs=200,
    loss_type="global_interaction", strategy="simplicial",
)
print(out["summary"])
```

### Recherche des meilleurs hyperparamètres d'un modèle

La recherche porte sur **un seul modèle** (une architecture `arch`) sur un jeu de données, sans agents ni alpha. Les meilleurs paramètres trouvés (lr, weight decay, optimiseur...) servent ensuite, si tu veux, à construire K agents de cette architecture.

```bash
pip install optuna                                             # en local, une fois (cluster : module Optuna/3.5.0-foss-2023a, déjà dans les scripts)
python -m experiments.best_hyperparams                         # configs/best_hyperparams.json
python -m experiments.best_hyperparams --dataset cifar10 --arch resnet32 --n-trials 50
python -m experiments.best_hyperparams --cpu --train-subset 2000 --epochs 3 --n-trials 2 --n-workers 1   # test rapide
sbatch slurm/best_hyperparams.sh                    # cluster : 8 GPU sur la même recherche

# ensuite : K agents avec les meilleurs paramètres du modèle seul
python -m experiments.collab_vs_big --config runs/best_hyperparams_resnet20_fashion_mnist/best_collab_vs_big.json --num-agents 3
python -m experiments.alpha_sweep   --config runs/best_hyperparams_resnet20_fashion_mnist/best_alpha_sweep.json   --num-agents 3
```

J'utilise [Optuna](https://optuna.org) plutôt que Keras Tuner :
- il ne demande pas de réécrire le modèle sous forme de `HyperModel` ;
- son sampler TPE (bayésien) trouve de bonnes régions plus vite qu'une grille ou que le hasard ;
- il arrête tôt les essais mauvais (pruning médian ou Hyperband) ;
- plusieurs jobs SLURM peuvent travailler sur la même étude, grâce au fichier journal, qui marche aussi sur un disque réseau ;
- il calcule l'importance de chaque hyperparamètre.

**Ce qui est cherché.** `search_space` dans `configs/best_hyperparams.json` contient une entrée par hyperparamètre du modèle (`lr`, `weight_decay`, `optimizer`, `batch_size`, `optimizer_kwargs`, `warmup_epochs`, `augment`...) :

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

**Parallélisation.** Une tâche = un worker qui lance `n_trials` essais ; les `n_workers` workers partagent la même étude (`study.journal`), soit `n_workers × n_trials` essais au total. En local, les workers tournent l'un après l'autre.

**Résultats** dans `runs/best_hyperparams_<arch>_<dataset>/` :
- `best.json` : meilleur essai, ses paramètres, importance de chaque hyperparamètre ;
- `trials.csv` : tous les essais, avec leur état (terminé, arrêté tôt, échoué) ;
- `best_collab_vs_big.json`, `best_alpha_sweep.json` : la config de l'expérience (`configs/<expérience>.json`) avec l'architecture, le jeu de données et les meilleurs paramètres du modèle. K se choisit au lancement (`--num-agents`), le nombre d'époques final reste celui de `configs/<expérience>.json`. Sortie dans `runs/<expérience>_<arch>_<dataset>_x<K>_best/`.

Relancer la même commande ajoute des essais à la recherche existante.

### Sur le cluster

Un script sbatch par expérience. Les réglages sont en tête du script (datasets, `N_RUNS`, hyperparamètres) ; ceux qui n'y sont pas viennent de `configs/<expérience>.json`. Depuis la racine du dépôt :

```bash
sbatch slurm/collab_vs_big.sh      # 2 datasets x 2 méthodes x 10 runs x (petits | grand) = 80 tâches -> --array=0-79
sbatch slurm/alpha_sweep.sh        # 2 datasets x 2 méthodes x 5 runs x 3 alphas = 60 tâches -> --array=0-59
sbatch slurm/best_hyperparams.sh   # 1 modèle, 1 dataset x 8 workers = 8 tâches -> --array=0-7 (module Optuna chargé par le script)
```

- **Une tâche = un dataset × un run** (× petits agents / grand agent, ou × alpha). L'indice `SLURM_ARRAY_TASK_ID` donne le dataset et la tâche, comme dans les autres projets. Toutes les tâches tournent en parallèle. Les K agents d'un run sont dans la même tâche.
- **Taille du job array** : si tu changes `DATASETS`, `N_RUNS` ou `ALPHAS`, mets à jour `#SBATCH --array=0-<N_TOTAL - 1>` (la valeur est en commentaire à côté). Un `--array` trop court est signalé dans le `.out` de la tâche 0, et les indices en trop ne font rien.
- **Plusieurs méthodes dans le même job array** : `METHODS` en tête de `alpha_sweep.sh` et `collab_vs_big.sh` liste les losses à comparer, une par ligne, au format `"loss_type:forme:tag"`. Par défaut : ta méthode (`global_interaction:multiplicative:`) et la DML classique (`dml:additive:dml`). Chaque méthode a son dossier (le tag est ajouté au nom), donc `notebooks/comparaison.ipynb` les retrouve toutes. Ajouter une ligne (ex. `"dml:multiplicative:dml_mult"`) ajoute une méthode : penser à `--array`.
- **Best model automatique** (`USE_BEST=1`, défaut) : pour chaque dataset, `alpha_sweep.sh` et `collab_vs_big.sh` cherchent `runs/best_hyperparams_<ARCH>_<dataset>/best_<expérience>.json`. S'il existe, ses paramètres (optimiseur, lr, weight decay, batch...) remplacent `OPTIMIZER`, `LEARNING_RATE`, `BATCH_SIZE` du script, et les résultats vont dans `..._x<K>_best/`. Sinon, les valeurs du script sont utilisées. La ligne `Paramètres:` du `.out` dit lequel des deux. `USE_BEST=0` force les valeurs du script. Les époques, K, la loss et α restent ceux du script.
- **Fusion** : la 1re tâche de chaque dataset soumet `slurm/merge.sh`, qui démarre quand tout le job array est fini et écrit `results.json` (`AUTO_MERGE=0 sbatch ...` pour ne pas le faire). Chaque tâche met aussi `results.json` à jour en finissant : les résultats partiels sont lisibles pendant la campagne. À la main : `python -m experiments.collab_vs_big [mêmes options] --merge-only`.
- **Sorties SLURM** dans `Logs/<expérience>_<job>_<tâche>.out / .err` ; résultats dans `runs/<expérience>_<arch>_<dataset>_x<K>/`.
- **Relancer** le même script ne refait que les tâches non terminées ; une tâche coupée reprend à sa dernière époque. Une seule tâche : `sbatch --array=7 slurm/collab_vs_big.sh`.
- **Données** : télécharger une fois depuis le nœud de connexion si les nœuds de calcul n'ont pas Internet (`python -c "from dml.data import download_dataset; download_dataset('cifar10')"`). Les tâches qui démarrent ensemble lisent le cache l'une après l'autre (verrou), sans se gêner.

### Avec Git

```bash
git pull                     # sur le cluster comme en local, au lieu de copier un zip
python -m pytest             # avant de lancer une campagne : ~1 min sur CPU
```

Lancer les campagnes depuis un code commité : `config.json` enregistre alors le commit exact (`git_dirty: false`). `runs/`, `Logs/` et les poids (`*.npz`, `*.npy`) sont ignorés par Git (`.gitignore`).

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
