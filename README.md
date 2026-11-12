# Multi-agents avec ResNet / MobileNet (v2)

Apprentissage collaboratif de K agents (ResNet, MobileNet) avec les losses multi-agents de DALPHA, comparé à des agents seuls ou à un grand agent au même nombre de paramètres. TensorFlow / Keras, testé sous TF 2.15 (Keras 2, modules du cluster) et TF 2.21 (Keras 3).

## Organisation du code

```
.
├── dml/                        package Python (tout le code réutilisable)
│   ├── core/                   le système multi-agents
│   │   ├── multiagent.py       MultiAgentModel : K agents, un optimiseur par agent, train/test step
│   │   ├── losses.py           les 5 losses, registre @register_loss
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
│   ├── evaluation/
│   │   ├── metrics.py          accuracy, NLL, ECE, désaccord
│   │   └── reporting.py        médiane [q05 ; q95] sur les graines, tableaux
│   └── utils/
│       ├── config.py           configs des expériences (dataclasses), JSON, ligne de commande
│       ├── device.py           GPU si disponible, sinon CPU
│       └── log.py              journal console + <run_dir>/log.txt
├── experiments/                un script par expérience (minces : config -> dml.training.runner)
│   ├── collab_vs_big.py        K petits agents vs 1 grand agent
│   ├── alpha_sweep.py          balayage de alpha
│   └── merge_sweep.py          rassemble les jobs SLURM d'un balayage
├── configs/                    un JSON par expérience (collab_vs_big.json, alpha_sweep.json)
├── slurm/                      scripts de soumission du cluster
├── notebooks/                  analyse des résultats (resultats.ipynb, resultats_sweep.ipynb)
├── tests/                      tests pytest
├── results/                    sorties des runs (ignoré par Git)
└── Logs/                       sorties SLURM (ignoré par Git)
```

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
| une expérience | `experiments/<nom>.py` + `configs/<nom>.json` + `slurm/soumission_<nom>.sh` | copier `alpha_sweep.py` comme modèle, avec une dataclass de config dans `dml/utils/config.py` |

Le nom enregistré est accepté partout ensuite : dans `compile(...)`, dans le JSON et sur la ligne de commande.

## Nouveautés du 2026-11-12

**Nouvelle arborescence** (voir plus haut). Les commandes changent : `python run_experiment.py` devient `python -m experiments.collab_vs_big`, `python run_alpha_sweep.py` devient `python -m experiments.alpha_sweep`, `sbatch soumission_*.sh` devient `sbatch slurm/soumission_*.sh`. Les imports deviennent `from dml.core.multiagent import MultiAgentModel` (ou `from dml import MultiAgentModel`). Les noms des classes et des fonctions ne changent pas.

Avec les réglages par défaut, l'entraînement est **identique au bit près** à la version précédente (vérifié en mode déterministe sous TF 2.15 : mêmes losses à chaque époque et mêmes accuracies), sauf le weight decay, qui ne touche plus BatchNorm ni les biais. Pour retrouver exactement l'ancien comportement : `"wd_exclude_norm_bias": false`.

- **Loss de la référence au choix** (`reference_loss`). `"ce"` (défaut, comme avant) ou `"same_as_agents"` : le grand agent est entraîné avec la même forme de loss que les petits agents et I_k = 0 (`L_k²` en multiplicatif, comme les agents à α = 0). Les deux camps ont alors la même échelle de gradient, ce qui enlève l'effet « lr effectif » vu entre SGD et Adam.
- **Forme de la loss** (`loss_form`) : `"multiplicative"` (défaut) ou `"additive"` (`L_k + α I_k`, qui redonne la CE à α = 0).
- **Reprise après coupure.** L'état complet (poids, BatchNorm, optimiseurs, historique) est sauvegardé à chaque époque. Relancer la même commande (avec `--run-dir`, ou `--resume <dossier>`) saute ce qui est terminé et reprend l'entraînement en cours à sa dernière époque. Les poids finaux et ceux de la meilleure `val_accuracy` sont gardés (`checkpoints/<run>/final_weights.npz`, `best_weights.npz`).
- **Meilleur modèle** : `"eval_on": "best"` évalue le test sur les poids de la meilleure époque de validation (défaut `"final"`, comme avant).
- **Config vérifiée.** Une clé inconnue dans le JSON arrête le script (`Clé(s) inconnue(s) dans la config : 'epoch' (voulais-tu dire 'epochs' ?)`), ainsi qu'un nom de loss, de stratégie ou d'optimiseur inconnu, ou une liste par agent de mauvaise longueur. Les clés qui commencent par `_` sont des commentaires.
- **Reproductibilité.** `config.json` contient une section `environment` : commit Git, branche, modifications non commitées (`git_dirty`), versions de Python / TF / Keras, machine, numéro de job SLURM, commande lancée.
- **Évaluation reproductible** : la stratégie `vote` tire l'agent avec un générateur remis à zéro à chaque `evaluate` / `predict`, donc la même accuracy à chaque évaluation.
- **Historique par époque** en CSV (`history/`), `log.txt` dans chaque dossier de résultats, arrêt immédiat si la loss devient NaN (`TerminateOnNaN`).
- **Une seule passe arrière ?** Testé : détacher les autres agents et dériver la somme des losses donne exactement les mêmes gradients, mais n'est pas plus rapide (≈ 500 ms par pas dans les deux cas, 5 × ResNet-20, CPU). TensorFlow ne parcourt déjà que le réseau de l'agent k pour le gradient de sa loss. Le `train_step` garde donc un gradient par agent.

## Configuration

Les valeurs par défaut sont dans `dml/utils/config.py`, le fichier JSON les remplace, la ligne de commande remplace le JSON. Chaque champ a son option : `python -m experiments.alpha_sweep --help` donne la liste.

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
python -m experiments.collab_vs_big --cpu --train-subset 2000 --epochs 2 --seeds 0
```

- **Petits agents** : `num_agents` agents `small_arch` entraînés ensemble avec `loss_type` + `strategy` (par défaut 5 × resnet20, `global_interaction` + `simplicial`).
- **Grand agent** : la même architecture élargie. La largeur est cherchée automatiquement pour avoir autant de paramètres entraînables que les petits réunis (5 × ResNet-20 = 1,36 M ; grand ResNet-20 largeur ≈ 2,24 → 1,36 M, écart < 0,3 %). Il est entraîné seul, en cross-entropy (`reference_loss: "ce"`) ou avec la même forme de loss que les petits agents et I_k = 0 (`reference_loss: "same_as_agents"`, soit `L_k²` en multiplicatif), avec les mêmes données, augmentation, optimiseur, epochs et graine.
- **Résultats** dans `results/<date>_<nom>/` (ou `--run-dir`) : `config.json` (config utilisée, matériel, nombres de paramètres, largeur du grand agent, commit Git et versions), `results.json` (par graine : accuracy, NLL, ECE de chaque petit agent, du consensus et du grand agent, historiques par epoch ; résumé par médiane et quantiles 5 % / 95 % sur les graines, moyenne et écart-type gardés pour info), `probs/*.npy` (probabilités de test), `history/*.csv` (historique par époque), `checkpoints/` (poids), `log.txt`. `results.json` est réécrit après chaque graine.
- Un tableau final compare, en médiane [quantile 5 % ; quantile 95 %] sur les graines : grand agent seul, consensus des petits, moyenne simple des petits, petit agent médian, meilleur petit agent. Le notebook `notebooks/resultats.ipynb` affiche les mêmes statistiques en tableaux et figures.

### Expérience : impact de la collaboration (balayage de alpha)

```bash
python -m experiments.alpha_sweep                                  # configs/alpha_sweep.json
python -m experiments.alpha_sweep --alphas 0 1 10 --epochs 50      # surcharger un réglage
python -m experiments.alpha_sweep --cpu --train-subset 2000 --epochs 2 --warmup-epochs 1 --alphas 0 1 --seeds 0   # test rapide
```

- `num_agents` agents `arch` entraînés avec `loss_type` pour chaque valeur de `alphas` (par défaut 0, 0.5, 1, 10, 50, 100), sur chaque graine.
- **alpha = 0** : pas d'interaction (loss `L_k²`, agents indépendants), consensus par **vote majoritaire**. **alpha > 0** : consensus avec la `strategy` du fichier de config.
- **Résultats** dans `results/<date>_alpha_sweep/` : `config.json`, `results.json` (par alpha et par graine : accuracy du consensus, de la moyenne simple, de chaque agent, NLL/ECE, désaccord entre agents, historiques ; résumé par alpha en médiane et quantiles 5 % / 95 %), `probs/agents_alpha<a>_s<graine>.npy`.
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

### Sur le cluster

```bash
sbatch slurm/soumission_alpha_sweep.sh            # un job par (alpha, graine)
python -m experiments.merge_sweep results/alpha_sweep   # rassemble les jobs pour notebooks/resultats_sweep.ipynb
sbatch slurm/soumission_collab_vs_big.sh
```

Un job coupé (limite de temps, nœud perdu) se relance avec la même commande (`sbatch --array=7 slurm/soumission_alpha_sweep.sh` pour un seul job du sweep) : il reprend à la dernière époque sauvegardée.

### Avec Git

```bash
git pull                     # sur le cluster comme en local, au lieu de copier un zip
python -m pytest             # avant de lancer une campagne : ~1 min sur CPU
```

Lancer les campagnes depuis un code commité : `config.json` enregistre alors le commit exact (`git_dirty: false`). `results/`, `Logs/` et les poids (`*.npz`, `*.npy`) sont ignorés par Git (`.gitignore`).

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
