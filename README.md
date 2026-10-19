# Multi-agents avec ResNet / MobileNet (v2)

Version modifiée du code partagé par DALPHA (`multiagent.py`, `models.py`, `tools.py`). Les fichiers d'origine sont intacts dans `/mnt/project-files/uploads/hearth/`.

## Ce qui ne change pas

- Les 5 fonctions de coût (`leave_one_out`, `global_interaction`, `vote_random`, `majority_hard`, `majority_loo`) et les 4 stratégies de consensus sont reprises telles quelles.
- Les poids de confiance `w_k = softmax(-H_k / tau)`, les callbacks `AgentHistoryCallback` et `ConvergenceTracker`, `predict` et `summary` aussi.
- Un optimiseur par agent, chaque agent n'est mis à jour qu'avec sa propre loss.

## Ce qui change

| Fichier | Changement |
|---|---|
| `models/` | Package remplaçant `models.py`. `AgentNetwork(arch="resnet32")` garde son nom de classe mais charge désormais un modèle enregistré par son nom, suivi d'une tête GAP + Dense + Softmax (float32). L'ancien MLP / petit CNN est supprimé. Modèles disponibles (`available_models()`) : resnet20/32/44/56/110, resnet18/34, mobilenetv1, mobilenetv2, mobilenetv2_0.5. |
| `multiagent.py` | Mêmes noms de classes et de méthodes. `MultiAgentModel(num_agents, archs=...)` remplace `hidden_sizes / use_conv / conv_*`. `archs` est un nom (agents identiques) ou une liste (agents hétérogènes). L'entrée peut être `(B, H, W, C)` (même image pour tous) ou `(B, K, H, W, C)` (format d'origine). Les 5 losses partagent une structure commune (`L_k * (L_k + alpha * interaction)`, calculée pour tous les agents à la fois ; alpha pondère l'interaction dans les 5 losses) et sont choisies via le dictionnaire `LOSS_FUNCTIONS` ; `train_step` et `test_step` partagent la mise à jour des métriques. Les cross-entropies sont calculées depuis les logits, avec la même borne à 1e-7 que Keras (résultats identiques). `optimizers_config` accepte un nom d'optimiseur, une liste (un par agent) ou une fonction qui crée un optimiseur par agent. XLA se règle par `compile(jit_compile=True)`. |
| `callbacks.py` (nouveau) | `AgentHistoryCallback` et `ConvergenceTracker`, sortis de `multiagent.py` (toujours importables depuis `multiagent`). `ConvergenceTracker` ne garde plus tous les poids en RAM : avec `reference_weights` il calcule les distances au fil de l'eau, sinon `snapshot_dir="auto"` écrit les poids sur disque. `keep_snapshots=True` retrouve l'ancien comportement. |
| `data.py` (nouveau) | Chargement Fashion-MNIST / CIFAR-10 / CIFAR-100, normalisation par canal, validation tirée du train, pipeline `tf.data` avec crop + flip. Option `per_agent_views` : une augmentation différente par agent. |
| `optimizers.py` (nouveau) | Registre des optimiseurs par nom, `choix_optimiseur(nom, learning_rate, weight_decay, **kwargs)` et `build_agent_optimizers` (un optimiseur par agent, éventuellement différents). Voir « Optimiseurs » plus bas. |
| `tools.py` | Warmup + cosinus avec weight decay dans l'optimiseur. La référence est un `MultiAgentModel` à 1 agent (même boucle, même optimiseur). Évaluation avec accuracy, NLL et ECE par agent et pour le consensus. `benchmark_architectures` compare référence, multi-agents et ensemble de K références indépendantes. |
| `run_alpha_sweep.py` (nouveau) | Expérience « impact de la collaboration » : mêmes K agents pour plusieurs valeurs de alpha, réglée par `configs/alpha_sweep.json`. |
| `run_experiment.py` (nouveau) | Expérience « K petits agents vs 1 grand agent » à nombre de paramètres égal, réglée par `configs/collab_vs_big.json`. |
| `models/` : `width` | `AgentNetwork(arch, width=...)` multiplie le nombre de filtres de chaque couche ; `tools.match_width` trouve la largeur qui donne un nombre de paramètres cible. |

## Organisation des modèles

```
models/
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
# models/vgg.py
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

puis ajouter `vgg` à la ligne `from . import resnet, mobilenet` de `models/__init__.py`. Le backbone renvoie des caractéristiques (B, H, W, C) ; la tête est ajoutée par `AgentNetwork`.

## Installation

| Fichier | Pour |
|---|---|
| `requirements_cpu.txt` | Windows natif, macOS, Linux sans GPU |
| `requirements_gpu.txt` | GPU NVIDIA sous Linux ou WSL2 (CUDA/cuDNN installés par pip) |

Python 3.10 à 3.13. Commandes détaillées : voir `INSTALL.md`.

## GPU ou CPU

`run_experiment.py` utilise automatiquement le GPU s'il est visible par TensorFlow, sinon le CPU (`--cpu` pour forcer le CPU). Il affiche au démarrage le matériel utilisé. Dans un notebook, appeler une fois `from device import configure_device; configure_device()` avant de créer un modèle.

- **Windows** : TensorFlow >= 2.11 n'utilise pas le GPU nativement sous Windows. Il faut l'installer dans WSL2 (Ubuntu) : `pip install "tensorflow[and-cuda]"`, le pilote NVIDIA Windows suffit. Vérifier avec `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
- **RTX 4060 (8 Go)** : `--precision mixed_bfloat16` accélère l'entraînement et réduit la mémoire. La mémoire GPU est allouée au besoin, l'écran reste utilisable.

## Optimiseurs

Chaque optimiseur se choisit par son nom ; un nom inconnu lève une erreur qui liste les choix.

`adadelta`, `adafactor`, `adagrad`, `adam`, `adamax`, `adamw`, `ftrl`, `lamb`, `lion`, `muon`, `nadam`, `rmsprop`, `schedule_free_adamw`, `sgd` (ceux absents de la version de Keras installée sont ignorés ; `optimizers.available_optimizers()` donne la liste réelle). `sgd` utilise le momentum de Nesterov 0.9, comme avant.

```python
from optimizers import choix_optimiseur, build_agent_optimizers
opt = choix_optimiseur("adam", learning_rate=1e-3)

# un optimiseur différent par agent (lr et weight_decay : valeur commune ou liste)
opts = build_agent_optimizers(["sgd", "adam", "lion"], num_agents=3, lr=[0.1, 1e-3, 1e-4],
                              epochs=200, steps_per_epoch=390)
model.compile(optimizers_config=opts)          # ou simplement compile(["sgd", "adam", "lion"])
```

Dans l'expérience, `optimizer` et `lr` du fichier de config acceptent un nom / une valeur, ou une liste (un par petit agent). Le grand agent prend le même optimiseur que les petits, sauf si `big_optimizer` / `big_lr` sont donnés (obligatoires quand les petits ont des optimiseurs différents). `optimizer_kwargs` passe des réglages supplémentaires (ex. `{"beta_1": 0.95}`).

```bash
python run_experiment.py --optimizer sgd adam adamw lion nadam --lr 0.1 1e-3 1e-3 1e-4 1e-3 \
    --big-optimizer sgd --big-lr 0.1
```

Ajouter un optimiseur : dans `optimizers.py`,

```python
@register_optimizer("mon_sgd")
def mon_sgd(**kwargs):
    return tf.keras.optimizers.SGD(momentum=0.95, **kwargs)
```

## Utilisation

### Expérience : 5 petits agents vs 1 grand agent

```bash
python run_experiment.py --config configs/collab_vs_big.json
# surcharger un réglage sans toucher au fichier
python run_experiment.py --config configs/collab_vs_big.json --dataset fashion_mnist --epochs 50
# test rapide (quelques minutes sur CPU)
python run_experiment.py --cpu --train-subset 2000 --epochs 2 --seeds 0
```

- **Petits agents** : `num_agents` agents `small_arch` entraînés ensemble avec `loss_type` + `strategy` (par défaut 5 × resnet20, `global_interaction` + `simplicial`).
- **Grand agent** : la même architecture élargie. La largeur est cherchée automatiquement pour avoir autant de paramètres entraînables que les petits réunis (5 × ResNet-20 = 1,36 M ; grand ResNet-20 largeur ≈ 2,24 → 1,36 M, écart < 0,3 %). Il est entraîné seul en cross-entropy, avec les mêmes données, augmentation, optimiseur, epochs et graine.
- **Résultats** dans `results/<date>_<nom>/` : `config.json` (config utilisée, matériel, nombres de paramètres, largeur du grand agent), `results.json` (par graine : accuracy, NLL, ECE de chaque petit agent, du consensus et du grand agent, historiques par epoch ; résumé par médiane et quantiles 5 % / 95 % sur les graines, moyenne et écart-type gardés pour info), `probs/*.npy` (probabilités de test). `results.json` est réécrit après chaque graine.
- Un tableau final compare, en médiane [quantile 5 % ; quantile 95 %] sur les graines : grand agent seul, consensus des petits, moyenne simple des petits, petit agent médian, meilleur petit agent. Le notebook `resultats.ipynb` affiche les mêmes statistiques en tableaux et figures.

### Expérience : impact de la collaboration (balayage de alpha)

```bash
python run_alpha_sweep.py                                  # configs/alpha_sweep.json
python run_alpha_sweep.py --alphas 0 1 10 --epochs 50      # surcharger un réglage
python run_alpha_sweep.py --cpu --train-subset 2000 --epochs 2 --warmup-epochs 1 --alphas 0 1 --seeds 0   # test rapide
```

- `num_agents` agents `arch` entraînés avec `loss_type` pour chaque valeur de `alphas` (par défaut 0, 0.5, 1, 10, 50, 100), sur chaque graine.
- **alpha = 0** : pas d'interaction (loss `L_k²`, agents indépendants), consensus par **vote majoritaire**. **alpha > 0** : consensus avec la `strategy` du fichier de config.
- **Résultats** dans `results/<date>_alpha_sweep/` : `config.json`, `results.json` (par alpha et par graine : accuracy du consensus, de la moyenne simple, de chaque agent, NLL/ECE, désaccord entre agents, historiques ; résumé par alpha en médiane et quantiles 5 % / 95 %), `probs/agents_alpha<a>_s<graine>.npy`.
- Le tableau final donne, pour chaque alpha : consensus, moyenne simple, agent médian, meilleur agent et désaccord moyen entre agents (diversité).

Depuis Python :

```python
from data import load_dataset
from tools import benchmark_architectures

data = load_dataset("cifar10")
out = benchmark_architectures(
    data, configs=["resnet20", "resnet56", ["resnet32", "mobilenetv2"]],
    num_agents=2, n_runs=3, epochs=200,
    loss_type="global_interaction", strategy="simplicial",
)
print(out["summary"])
```

## Équivalence avec la version précédente

Vérifiée numériquement : les 5 losses et leurs gradients sont identiques à l'ancienne version à 3e-7 près (précision float32), y compris quand un agent donne une probabilité < 1e-7 à la bonne classe. Sur 80 pas d'entraînement réels (Fashion-MNIST, sans augmentation), les deux versions donnent la même loss au départ et à l'arrivée (33,04).

La borne à 1e-7 sur les probabilités est indispensable : sans elle, la forme multiplicative `L_k * (L_k + ...)` produit des gradients énormes sur les exemples très mal classés et l'entraînement diverge (NaN dès le 4e pas à lr 0,1 dans un test).

## Points d'attention

- **Échelle des losses multiplicatives.** `L_k * (L_k + alpha * (L_loc + L_glob))` vaut environ 16 au départ sur 10 classes, puis tend vers 0 plus vite que la CE. Avec SGD à lr 0.1, le pas effectif est donc plus grand au début et plus petit à la fin que pour la référence. Le warmup (5 epochs) limite le premier effet ; si les agents décrochent en fin d'entraînement, c'est la piste à regarder.
- **Weight decay.** Le `train_step` n'ajoute pas les régularisations des couches (`self.losses`), d'où le weight decay porté par l'optimiseur.
- **`ConvergenceTracker`** : sans `reference_weights`, il doit garder les poids de chaque epoch jusqu'à la fin pour mesurer la distance aux poids finaux. Pour les gros modèles, utiliser `ConvergenceTracker(snapshot_dir="auto")` (sur disque) ou fournir `reference_weights`.
- **Téléchargement des données.** Keras télécharge CIFAR-10 depuis cs.toronto.edu au premier appel. Sur un nœud de cluster sans internet, lancer `load_dataset` une fois sur le nœud de connexion (cache dans `~/.keras/datasets`).
