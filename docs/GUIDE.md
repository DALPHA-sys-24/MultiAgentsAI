# Guide détaillé

Référence complète du code. Pour démarrer, voir le [README](../README.md).

Apprentissage collaboratif de K agents (ResNet, MobileNet) avec des losses multi-agents, comparé à des agents indépendants, à la DML (Zhang et al., 2018) et à un grand modèle seul au même nombre de paramètres. TensorFlow / Keras, testé sous TF 2.15 (Keras 2, modules du cluster) et TF 2.21 (Keras 3).

## Organisation du code

```
.
├── collaborative_learning/     package Python (tout le code réutilisable)
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
│   │   ├── reporting.py        médiane [q05 ; q95] sur les graines, tableaux
│   │   └── plots.py            figures PNG / PDF et summary.csv (sans Jupyter)
│   └── utils/
│       ├── config.py           configs des expériences (dataclasses), ligne de commande
│       ├── config_files.py     lecture des YAML / JSON, composition avec defaults:
│       ├── device.py           GPU si disponible, sinon CPU
│       └── log.py              journal console + fichiers de logs/
├── experiments/
│   ├── common.py               socle commun : tâches, dossier de résultats, reprise, fusion
│   ├── campaign.py             campagnes : liste numérotée des tâches, lancement, fusion, figures
│   ├── collaboration.py        K agents pour chaque alpha
│   ├── reference.py            grand modèle seul (autant de paramètres que les K agents)
│   ├── hyperparameter_search.py  meilleurs hyperparamètres d'UN modèle (Optuna)
│   └── plot_results.py         figures (PNG + PDF) d'une campagne, sans Jupyter
├── configs/
│   ├── defaults.yaml           réglages communs à toutes les expériences
│   ├── datasets/               fashion_mnist, cifar10, cifar100
│   ├── methods/                global_interaction, dml, independent
│   ├── collaboration.yaml, reference.yaml, hyperparameter_search.yaml
│   └── campaigns/              alpha_sweep, lr_sweep, hyperparameter_search, quick_test
├── slurm/
│   ├── campaign.sh             job array : sbatch --array=0-<N-1> slurm/campaign.sh <campagne>
│   ├── merge_campaign.sh       fusion + figures (soumise automatiquement)
│   ├── modules.sh              modules du cluster
│   └── helpers.sh              fonctions communes
├── notebooks/
│   ├── effect_of_alpha.ipynb       effet de alpha pour une méthode d'une campagne
│   ├── agents_vs_big_model.ipynb   K agents vs grand modèle, pour un point de la grille
│   ├── compare_methods.ipynb       plusieurs méthodes côte à côte (la loss proposée, DML, indépendants, grand modèle)
│   └── example_figures/
├── tests/                      tests pytest
├── results/                    résultats (ignoré par Git)
└── slurm_logs/                 sorties SLURM .out / .err (ignoré par Git)
```

### Dossier de résultats

```
results/
├── <campagne>/<dataset>/<arch>_<K>agents/
│   ├── independent/alpha0/                 agents indépendants (alpha = 0), partagés par les méthodes
│   ├── global_interaction/alpha10/         un dossier par méthode et par point de la grille
│   ├── dml/alpha1_lr0.1/
│   └── figures/                            figures + summary.csv de la campagne
├── reference/<dataset>/<arch>_x<K>params_<optimiseur>_lr<lr>[_tuned]/   grand modèle seul
└── hyperparameter_search/<dataset>/<arch>/ best.json, trials.csv, tuned_params.yaml
```

Contenu d'un dossier de run (agents ou grand modèle) :

```
├── config.json             réglages, tailles des modèles, environnement (commit Git, versions)
├── results.json            résumé (médiane [q05 ; q95]) et détail par graine
├── test_probabilities/     probabilités de test (.npy) : alpha<a>_seed<s>.npy (agents), seed<s>.npy (grand modèle)
├── tasks/<tâche>/          une tâche : result.json, test_probabilities.npy, history.csv, checkpoint/
│                           (alpha1_seed0, alpha1_seed1... ; seed0, seed1... pour le grand modèle)
└── logs/                   journal de chaque processus (all.txt, task_003.txt, merge.txt)
```

- **Un run = une graine.** `n_runs: 10` lance les graines 0 à 9 ; `results.json` donne la médiane et les quantiles 5 % / 95 % sur ces runs.
- **Même dossier = mêmes réglages.** Si le dossier existe avec d'autres réglages (lr, optimiseur, epochs...), le script s'arrête et le dit (`--list` d'une campagne le vérifie avant de soumettre). Peuvent s'ajouter à un dossier existant : des graines (`n_runs`) et, pour une expérience lancée seule, des alphas.
- **Reprise.** Une tâche terminée est sautée, une tâche interrompue reprend à sa dernière époque.
- **Fusion seule** : `python -m experiments.campaign <campagne> --merge`, ou pour un dossier `--run-dir <dossier> --merge-only` (relit `config.json`).

**Toutes les commandes se lancent depuis la racine du dépôt** (le dossier qui contient `collaborative_learning/`), comme modules : `python -m experiments.collaboration`. Pas besoin d'installer le package ; `pip install -e .` est possible pour importer `collaborative_learning` depuis n'importe quel dossier.

Où ajouter quoi :

| Ajouter | Où | Comment |
|---|---|---|
| une loss | `collaborative_learning/core/losses.py` | `@register_loss("nom")` |
| une stratégie de consensus | `collaborative_learning/core/consensus.py` | `@register_strategy("nom")` |
| une architecture | `collaborative_learning/models/<famille>.py` | `@register_model("nom")`, puis importer le module dans `collaborative_learning/models/__init__.py` |
| un optimiseur | `collaborative_learning/training/optimizers.py` | `@register_optimizer("nom")` |
| un jeu de données | `collaborative_learning/data/datasets.py` | `@register_dataset("nom")` |
| un réglage de config | `collaborative_learning/utils/config.py` | un champ `opt(...)` dans la dataclass (option `--nom` créée automatiquement) |
| un hyperparamètre à chercher | `configs/hyperparameter_search.yaml` | une entrée dans `search_space` (un réglage du modèle) |
| une méthode à comparer | `configs/methods/<nom>.yaml` | `method`, `loss_type`, `loss_form`, `strategy` ; puis le nom dans `methods:` d'une campagne |
| un jeu de réglages propre à un dataset | `configs/datasets/<nom>.yaml` | repris automatiquement par les campagnes |
| une étude (balayage, comparaison) | `configs/campaigns/<nom>.yaml` | `datasets`, `methods`, `settings`, `grid`, `reference` |
| une expérience | `experiments/<nom>.py` + `configs/<nom>.yaml` | une sous-classe de `Experiment` (`experiments/common.py`) avec `tasks`, `run_task`, `merge` ; copier `reference.py` comme modèle, et l'ajouter à `experiment_class` dans `campaign.py` |

Le nom enregistré est accepté partout ensuite : dans `compile(...)`, dans les YAML et sur la ligne de commande.

## Baseline DML (Zhang et al., 2018)

Loss `dml` : I_k = 1/(K−1) · Σ_{j≠k} KL(y^j ‖ y^k), les prédictions des autres agents étant des cibles fixes (`stop_gradient`).

| Variante | Réglages | Loss de l'agent k |
|---|---|---|
| DML classique (papier) | `configs/methods/dml.yaml` (additive), α = 1 | CE(y, y^k) + 1/(K−1) Σ KL(y^j ‖ y^k) |
| DML dans la forme du projet | `loss_type: dml`, `loss_form: multiplicative` | L_k · (L_k + α · KL) |

```bash
# DML classique (α = 1) et l'ensemble indépendant (α = 0, CE seule pour chaque agent)
python -m experiments.collaboration --config configs/methods/dml.yaml --alphas 0 1
```

Dans une campagne : `methods: [global_interaction, dml]` (c'est le cas de `alpha_sweep` et `lr_sweep`). Pour la variante multiplicative, créer `configs/methods/dml_multiplicative.yaml` (`defaults: [methods/dml]`, `method: dml_multiplicative`, `loss_form: multiplicative`) et l'ajouter à `methods:`.

Deux différences avec le papier :
- les K agents sont mis à jour en même temps à partir du même forward, alors que le papier les met à jour l'un après l'autre dans un pas (la mise à jour simultanée est celle de la plupart des implémentations publiques) ;
- le papier évalue chaque réseau seul : nos résultats le donnent aussi (chaque agent, agent médian, meilleur agent, moyenne simple), en plus du consensus.

## Figures sans Jupyter : `plot_results`

```bash
python -m experiments.plot_results lr_sweep                      # une campagne (refait aussi à chaque --merge)
python -m experiments.plot_results alpha_sweep --formats pdf     # PDF seulement (défaut : png pdf)
python -m experiments.plot_results --run-dirs results/alpha_sweep/cifar10/resnet20_3agents/dml/alpha10 \
    results/alpha_sweep/cifar10/resnet20_3agents/global_interaction/alpha10 \
    --reference results/reference/cifar10/resnet20_x3params_sgd_lr0.01_tuned --params alpha --out-dir figures_article
```

Pour chaque (dataset, arch, K) de la campagne, compare toutes les méthodes et tous les points de la grille. Partout : médiane et bande des quantiles 5 % / 95 % sur les graines. Sorties dans `results/<campagne>/<dataset>/<arch>_<K>agents/figures/` :

| Fichier | Contenu |
|---|---|
| `summary.csv` | une ligne par méthode et par point (+ grand modèle) : consensus, moyenne simple, agent médian, meilleur agent, désaccord, NLL, ECE en médiane / q05 / q95 |
| `accuracy_vs_<param>.png/.pdf` | pour chaque paramètre balayé (alpha, lr...) : consensus et agent médian, une courbe par méthode, grand modèle en pointillés ; échelle log pour lr et weight_decay |
| `diversity_vs_<param>.png/.pdf` | désaccord entre agents |
| `paired_difference.png/.pdf` | écart d'accuracy à graine égale avec le grand modèle (sinon avec les agents indépendants), et graines gagnées |
| `learning_curves.png/.pdf` | accuracy de validation du consensus par époque, un panneau par méthode, grand modèle en pointillés |

Les points alpha = 0 (agents indépendants) sont repris dans la courbe de chaque méthode. Les PDF gardent le texte en vrai texte (polices TrueType), pour LaTeX. Sur le cluster, le job de fusion fait les figures tout seul. Pas de GPU nécessaire, quelques secondes sur le nœud de connexion.

## Notebooks

Les notebooks lisent une campagne fusionnée : régler en tête `CAMPAIGN` (ex. `alpha_sweep`, `lr_sweep`), `RESULTS_ROOT` (`../results`), et au besoin `DATASET` et `K` (défaut : le premier groupe de la campagne). Ils retrouvent les dossiers des méthodes et le grand modèle de `results/reference/` comme `plot_results` (`campaign_groups` dans `collaborative_learning/evaluation/plots.py`). Figures dans `results/<campagne>/<dataset>/<arch>_<K>agents/figures/notebooks/`.

| Notebook | Réglages | Montre |
|---|---|---|
| `effect_of_alpha.ipynb` | `METHOD`, `FIXED` (ex. `{"lr": 0.01}`) ; demande au moins 2 alphas | accuracy, gain, diversité, courbes et calibration selon alpha ; grand modèle en référence |
| `agents_vs_big_model.ipynb` | `METHOD`, `POINT` (ex. `{"alpha": 1, "lr": 0.1}`) | K agents vs grand modèle pour un point : tableau, accuracy par agent, courbes, calibration, diversité, accuracy par classe, puis tous les points face au grand modèle |
| `compare_methods.ipynb` | `REFERENCE` | toutes les méthodes et tous les points côte à côte, différence appariée, courbes de validation |

`compare_methods.ipynb` :
- **Tableau** : consensus, moyenne simple, agent médian, meilleur agent, en médiane [q05 ; q95] sur les graines (`comparison.csv`).
- **Accuracy par méthode** : consensus (point plein) et agent médian (point creux), même couleur par méthode que les figures de `plot_results`.
- **Différence appariée** : pour chaque graine, écart d'accuracy avec la référence (grand modèle, sinon agents indépendants) à graine égale, et nombre de graines gagnées.
- **Courbes d'apprentissage** : accuracy de validation par époque, médiane sur les graines.

Pour la campagne de test : `RESULTS_ROOT = "../results_quick_test"`, `CAMPAIGN = "quick_test"`. Les notebooks importent le code (`CODE_DIR = ".."`) : TensorFlow et PyYAML doivent être installés.

## Fonctionnalités

- `--n-runs N` : graines 0 à N - 1. `steps_per_execution` (8 par défaut) : 8 pas d'entraînement par appel au GPU ; résultats identiques (vérifié sous TF 2.15 et 2.21).
- Le weight decay ne touche pas BatchNorm ni les biais (`"wd_exclude_norm_bias": false` pour l'ancien comportement).
- **Loss du grand modèle au choix** (`loss` de `configs/reference.yaml`). `ce` (défaut) ou `same_as_agents` : le grand modèle est entraîné avec la même forme de loss que les agents et I_k = 0 (`L_k²` en multiplicatif, `loss_form`). Les deux camps ont alors la même échelle de gradient, ce qui enlève l'effet « lr effectif » vu entre SGD et Adam.
- **Forme de la loss** (`loss_form`) : `"multiplicative"` (défaut) ou `"additive"` (`L_k + α I_k`, qui redonne la CE à α = 0).
- **Reprise après coupure.** L'état complet (poids, BatchNorm, optimiseurs, historique) est sauvegardé à chaque époque. Relancer la même commande saute ce qui est terminé et reprend l'entraînement en cours à sa dernière époque. Les poids finaux et ceux de la meilleure `val_accuracy` sont gardés (`tasks/<tâche>/checkpoint/final_weights.npz`, `best_weights.npz`).
- **Meilleur modèle** : `eval_on: best` évalue le test sur les poids de la meilleure époque de validation (défaut `final`, comme avant).
- **Config vérifiée.** Une clé inconnue dans un YAML (ou une campagne) arrête le script (`Clé(s) inconnue(s) dans la config : 'epoch' (voulais-tu dire 'epochs' ?)`), ainsi qu'un nom de loss, de stratégie ou d'optimiseur inconnu, ou une liste par agent de mauvaise longueur. Les clés qui commencent par `_` sont des commentaires.
- **Reproductibilité.** `config.json` contient une section `environment` : commit Git, branche, modifications non commitées (`git_dirty`), versions de Python / TF / Keras, machine, numéro de job SLURM, commande lancée.
- **Évaluation reproductible** : la stratégie `vote` tire l'agent avec un générateur remis à zéro à chaque `evaluate` / `predict`, donc la même accuracy à chaque évaluation.
- **Historique par époque** en CSV (`tasks/<tâche>/history.csv`), journaux dans `logs/`, arrêt immédiat si la loss devient NaN (`TerminateOnNaN`).

## Configuration

Ordre de lecture (la dernière valeur lue gagne) : valeurs par défaut des dataclasses (`collaborative_learning/utils/config.py`) < fichier YAML (`--config`, défaut `configs/<expérience>.yaml`) < ligne de commande. Chaque champ a son option : `python -m experiments.collaboration --help` donne la liste. Les deux écritures marchent : `--n-runs 3` ou `--n_runs 3`.

Un YAML peut reprendre d'autres fichiers de `configs/` avant ses propres valeurs :

```yaml
# configs/collaboration.yaml
defaults: [defaults, datasets/cifar10, methods/global_interaction]
arch: resnet20
num_agents: 5
alphas: [0, 1, 10]
```

Les anciens fichiers JSON (dont les `config.json` des dossiers de résultats) restent lisibles. Les clés qui commencent par `_` sont des commentaires.

Dans une campagne, la config d'un point est : `collaboration.yaml` < `datasets/<dataset>.yaml` < `methods/<méthode>.yaml` < `tuned_params.yaml` (si `use_tuned_params: true` ; seuls batch, optimiseur, lr, weight decay, warmup, augmentation, val_size, précision sont repris) < `settings` < valeurs du point de la grille. `--set clé=valeur` remplace un `settings` sans toucher au fichier.

Ajouter un réglage : ajouter un champ dans la dataclass (`mon_reglage: float = opt(0.5, "explication")`). L'option `--mon-reglage` et la vérification sont créées automatiquement ; il reste à le transmettre là où il sert. Il peut alors aller dans `settings` ou `grid` d'une campagne.

| Clé | Valeurs | Effet |
|---|---|---|
| `method` | nom libre | nom du dossier de la méthode (défaut : `<loss_type>_<loss_form>`) |
| `loss_type`, `strategy` | noms enregistrés | loss et consensus |
| `loss_form` | `multiplicative`, `additive` | forme de la loss |
| `alphas` | liste de nombres | une valeur = run simple, plusieurs = balayage (`grid: {alpha: [...]}` dans une campagne) |
| `loss` (reference) | `ce`, `same_as_agents` | loss du grand modèle |
| `wd_exclude_norm_bias` | `true`, `false` | weight decay hors BatchNorm et biais |
| `eval_on` | `final`, `best` | poids évalués sur le test |
| `checkpoint_every` | entier | fréquence de sauvegarde de l'état de reprise |
| `optimizer_kwargs` | dictionnaire | ex. `{global_clipnorm: 1.0}` ou `{use_ema: true}` |

## Campagnes

Une campagne (`configs/campaigns/<nom>.yaml`) décrit tout ce qu'on lance d'un coup :

| Clé | Rôle |
|---|---|
| `experiment` | `collaboration` (défaut) ou `hyperparameter_search` |
| `datasets` | liste de jeux de données |
| `methods` | fichiers de `configs/methods/` ; alpha = 0 donne toujours `independent`, entraîné une seule fois |
| `use_tuned_params` | reprend `results/hyperparameter_search/<dataset>/<arch>/tuned_params.yaml` s'il existe (ou l'ancien `tuned_collaboration.json`) |
| `settings` | réglages communs à toute la campagne (arch, num_agents, epochs, n_runs...) |
| `grid` | valeurs à balayer, produit cartésien : `alpha`, `lr`, `weight_decay`, `num_agents`... |
| `reference` | `true` : grand modèle seul pour chaque (dataset, arch, K), aux réglages de base (pas ceux de la grille) |

```bash
python -m experiments.campaign lr_sweep --list         # tâches numérotées, dossiers, commande sbatch ; vérifie les dossiers existants
python -m experiments.campaign lr_sweep --task 7       # une seule tâche (ce que fait chaque tâche du job array)
python -m experiments.campaign lr_sweep --merge        # results.json de chaque dossier, puis figures
python -m experiments.campaign quick_test --cpu        # toutes les tâches l'une après l'autre, puis fusion
python -m experiments.campaign lr_sweep --list --set epochs=2 n_runs=1
```

Le grand modèle est rangé dans `results/reference/`, hors de la campagne : deux campagnes avec le même dataset, la même architecture, le même K et les mêmes réglages de base (ex. `alpha_sweep` et `lr_sweep`) le partagent, et il n'est entraîné qu'une fois. Une valeur de `num_agents` dans la grille donne un grand modèle par K.

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

Dans l'expérience `collaboration`, `optimizer` et `lr` acceptent un nom / une valeur, ou une liste (un par agent). Le grand modèle (`experiments.reference`) a ses propres `optimizer`, `lr` et `optimizer_kwargs`, une seule valeur chacun. `optimizer_kwargs` passe des réglages supplémentaires (ex. `{beta_1: 0.95}`).

```bash
python -m experiments.collaboration --num-agents 5 --optimizer sgd adam adamw lion nadam --lr 0.1 1e-3 1e-3 1e-4 1e-3
python -m experiments.reference --num-agents 5 --optimizer sgd --lr 0.1
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
python -m experiments.collaboration                                   # configs/collaboration.yaml
# un run simple : un seul alpha
python -m experiments.collaboration --dataset fashion_mnist --num-agents 3 --alphas 1 --n-runs 5
# un balayage de alpha
python -m experiments.collaboration --alphas 0 1 10 --epochs 50
# test rapide (quelques minutes sur CPU)
python -m experiments.collaboration --cpu --train-subset 2000 --epochs 2 --warmup-epochs 1 --alphas 1 --n-runs 1
```

- **Agents** : `num_agents` agents `arch` entraînés ensemble avec `loss_type` pour chaque valeur de `alphas`, sur chaque graine. **alpha = 0** : agents **indépendants** : chaque agent est entraîné seul en **cross-entropy classique** (loss `independent`, forme additive), quels que soient `loss_type` et `loss_form`, et le consensus est un **vote majoritaire**. **alpha > 0** : `loss_type`, `loss_form` et `strategy` de la config (par défaut `global_interaction`, multiplicative, `simplicial`). Chaque run de `results.json` indique la loss, la forme et le consensus utilisés.
- **Tâches** : `alpha<a>_seed<s>`, indépendantes.
- **`results.json`** :
  - `method`, `agent_params` ;
  - `summary` : pour chaque alpha, consensus, moyenne simple, agent médian, meilleur agent, désaccord, NLL/ECE, en médiane [q05 ; q95] ;
  - `runs` : une entrée par (alpha, graine), avec les accuracies de chaque agent et les historiques.
- Sortie d'un lancement seul : `results/collaboration/<dataset>/<arch>_<K>agents/<méthode>[_<tag>]/`.

### Le grand modèle seul : `reference`

```bash
python -m experiments.reference --dataset cifar10 --num-agents 3 --n-runs 5
```

La même architecture élargie : la largeur est cherchée automatiquement pour avoir autant de paramètres entraînables que les K agents réunis (5 × ResNet-20 = 1,36 M ; grand ResNet-20 largeur ≈ 2,24, écart < 0,3 %), ou imposée avec `--width`. Entraîné seul, en cross-entropy (`loss: ce`) ou avec la même forme de loss que les agents et I_k = 0 (`same_as_agents`), avec les mêmes données, augmentation, epochs et graines. Il ne dépend ni de alpha ni de la méthode. Tâches `seed<s>` ; `results.json` contient `summary` (accuracy, NLL, ECE en médiane [q05 ; q95]), `runs` et `model_sizes`. Dossier : `results/reference/<dataset>/<arch>_x<K>params_<optimiseur>_lr<lr>[_<tag>]/`.

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
python -m experiments.hyperparameter_search              # configs/hyperparameter_search.yaml
python -m experiments.hyperparameter_search --dataset cifar10 --arch resnet32 --n-trials 50
python -m experiments.hyperparameter_search --cpu --train-subset 2000 --epochs 3 --n-trials 2 --n-workers 1   # test rapide
sbatch --array=0-15 slurm/campaign.sh hyperparameter_search   # cluster : 2 datasets x 8 workers

# ensuite : les campagnes avec use_tuned_params: true reprennent les meilleurs paramètres
python -m experiments.campaign alpha_sweep --list        # la ligne « paramètres <dataset> » dit quel fichier est utilisé
```

On utilise [Optuna](https://optuna.org) plutôt que Keras Tuner :
- il ne demande pas de réécrire le modèle sous forme de `HyperModel` ;
- son sampler TPE (bayésien) trouve de bonnes régions plus vite qu'une grille ou que le hasard ;
- il arrête tôt les essais mauvais (pruning médian ou Hyperband) ;
- plusieurs jobs SLURM peuvent travailler sur la même étude, grâce au fichier journal, qui marche aussi sur un disque réseau ;
- il calcule l'importance de chaque hyperparamètre.

**Ce qui est cherché.** `search_space` dans `configs/hyperparameter_search.yaml` contient une entrée par hyperparamètre du modèle (`lr`, `weight_decay`, `optimizer`, `batch_size`, `optimizer_kwargs`, `warmup_epochs`, `augment`...) :

```yaml
search_space:
  lr:           {type: float, low: 0.0001, high: 0.3, log: true}
  weight_decay: {type: float, low: 0.00001, high: 0.01, log: true}
  batch_size:   {type: int, low: 64, high: 256, step: 64}
  optimizer:    {type: categorical, choices: [sgd, adam, adamw]}
```

Les autres réglages (jeu de données, `arch`, `epochs` par essai...) sont fixés dans le même fichier.

**Loss du modèle** (`loss`) : `ce` (cross-entropy, défaut) ou `same_as_agents` (`L_k²` en multiplicatif). Avec SGD, la loss multiplicative des agents a un gradient environ `2 L_k` fois plus grand que la CE : un lr trouvé en CE peut être trop grand pour les K agents (lr 0.1 fait diverger la loss multiplicative). `same_as_agents` donne un lr directement transposable aux agents ; avec Adam / AdamW la différence est faible.

**Critère.** `metric` vaut `val_accuracy` (défaut) ou `val_loss` du modèle. Chaque essai est noté sur la meilleure époque de **validation**, et le test n'est jamais utilisé pendant la recherche.

**Parallélisation.** Une tâche = un worker (`worker0`, `worker1`...) qui lance `n_trials` essais ; les `n_workers` workers partagent la même étude (`study.journal`), soit `n_workers × n_trials` essais au total. En local, les workers tournent l'un après l'autre.

**Résultats** dans `results/hyperparameter_search/<dataset>/<arch>/` :
- `best.json` : meilleur essai, ses paramètres, importance de chaque hyperparamètre ;
- `trials.csv` : tous les essais, avec leur état (terminé, arrêté tôt, échoué) ;
- `tuned_params.yaml` : les meilleurs réglages d'entraînement (optimiseur, lr, weight decay, batch...), repris par les campagnes avec `use_tuned_params: true` ; K, epochs, la loss et alpha restent ceux de la campagne. Le grand modèle de ces campagnes est marqué `_tuned`. À la main : `python -m experiments.collaboration --config results/hyperparameter_search/cifar10/resnet20/tuned_params.yaml --num-agents 3`.

Relancer la même commande ajoute des essais à la recherche existante.

### Sur le cluster

Un seul script sbatch pour toutes les campagnes. Depuis la racine du dépôt, sur le nœud de connexion :

```bash
python -m experiments.campaign alpha_sweep --list         # N tâches, dossiers, données téléchargées
sbatch --array=0-59 slurm/campaign.sh alpha_sweep         # la commande exacte est affichée par --list
```

- **Une tâche du job array = une tâche de la campagne** : un dataset × une méthode × un point de la grille × une graine, ou une graine du grand modèle, ou un worker Optuna. Les K agents d'un run sont dans la même tâche (loss commune). L'ordre des tâches est celui de `--list`.
- **Taille du job array** : `--list` donne N. Un `--array` trop court est signalé dans le `.out` de la tâche 0 ; les indices en trop ne font rien.
- **Fusion** : la première tâche soumet `slurm/merge_campaign.sh` (job `merge_<campagne>`), qui démarre quand tout le job array est fini, écrit `results.json` de chaque dossier puis les figures. Chaque tâche met aussi `results.json` de son dossier à jour en finissant : les résultats partiels sont lisibles pendant la campagne. À la main : `python -m experiments.campaign <campagne> --merge`.
- **Modules** : `slurm/modules.sh` (TensorFlow 2.15, matplotlib, Optuna, PyYAML).
- **Sorties SLURM** dans `slurm_logs/campaign_<job>_<tâche>.out / .err`.
- **Relancer** la même commande ne refait que les tâches non terminées ; une tâche coupée reprend à sa dernière époque. Une seule tâche : `sbatch --array=7 slurm/campaign.sh alpha_sweep`.
- **Données** : `--list` les télécharge depuis le nœud de connexion (les nœuds de calcul n'ont souvent pas Internet). Les tâches qui démarrent ensemble lisent le cache l'une après l'autre (verrou), sans se gêner.

Campagnes fournies :

| Campagne | Contenu | Tâches |
|---|---|---|
| `hyperparameter_search` | Optuna sur un ResNet-20, Fashion-MNIST et CIFAR-10, 8 workers | 16 |
| `alpha_sweep` | interaction globale et DML, α ∈ {0, 10, 100}, K = 3, 5 graines, grand modèle | 60 |
| `lr_sweep` | interaction globale et DML, α = 1, lr ∈ {0.1, 0.01, 0.001, 0.0001}, K = 3, 5 graines, grand modèle au lr* | 90 |
| `quick_test` | test local sur CPU (2 époques, 500 images) | 8 |

### Balayage du learning rate (`lr_sweep`)

α fixé à 1, chaque méthode lancée pour chaque lr de la grille ; les autres hyperparamètres viennent de `tuned_params.yaml` du dataset. Référence : le grand modèle seul, entraîné une fois par graine avec le lr* de la recherche (le même que pour `alpha_sweep`, donc partagé). Dossiers `results/lr_sweep/<dataset>/resnet20_3agents/<méthode>/alpha1_lr<lr>/`. Figures `accuracy_vs_lr` et `diversity_vs_lr` (échelle log, grand modèle à lr* en pointillés).

### Avec Git

```bash
git pull                     # sur le cluster comme en local, au lieu de copier un zip
python -m pytest             # avant de lancer une campagne : quelques minutes sur CPU
```

Lancer les campagnes depuis un code commité : `config.json` enregistre alors le commit exact (`git_dirty: false`). `results/`, `slurm_logs/` et les poids (`*.npz`, `*.npy`) sont ignorés par Git (`.gitignore`).

## Tests

```bash
python -m pytest             # depuis la racine du dépôt
```

Ils vérifient : la CE quand K = 1 (et `L_k²` avec `single_agent_loss="form"`), la forme additive, le gradient des losses contre une différence finie, l'ajout d'une loss au registre, l'évaluation reproductible de `vote`, le weight decay hors BatchNorm et biais, la sauvegarde et la reprise d'un entraînement interrompu, la lecture, la composition (`defaults:`) et la vérification des configs, le découpage des campagnes (dossiers, alpha = 0 partagé, grand modèle, paramètres de la recherche), une campagne complète tâche par tâche jusqu'aux figures. Lancer les tests après chaque modification du code.

## Points d'attention

- **Échelle des losses multiplicatives.** `L_k * (L_k + alpha * (L_loc + L_glob))` vaut environ 16 au départ sur 10 classes, puis tend vers 0 plus vite que la CE. Avec SGD à lr 0.1, le pas effectif est donc plus grand au début et plus petit à la fin que pour la référence. Le warmup (5 epochs) limite le premier effet ; si les agents décrochent en fin d'entraînement, c'est la piste à regarder.
- **Weight decay.** Le `train_step` n'ajoute pas les régularisations des couches (`self.losses`), d'où le weight decay porté par l'optimiseur. Il ne s'applique plus aux paramètres de BatchNorm ni aux biais (`wd_exclude_norm_bias: false` pour l'ancien comportement).
- **`ConvergenceTracker`** : sans `reference_weights`, il doit garder les poids de chaque epoch jusqu'à la fin pour mesurer la distance aux poids finaux. Pour les gros modèles, utiliser `ConvergenceTracker(snapshot_dir="auto")` (sur disque) ou fournir `reference_weights`.
- **Téléchargement des données.** Keras télécharge CIFAR-10 depuis cs.toronto.edu au premier appel. Sur un nœud de cluster sans internet, lancer `load_dataset` une fois sur le nœud de connexion (cache dans `~/.keras/datasets`).
