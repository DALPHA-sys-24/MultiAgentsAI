"""
dml : apprentissage collaboratif multi-agents (Deep Mutual Learning et variantes).

    dml/
    ├── core/         le modèle multi-agents, les losses et les consensus
    ├── models/       les architectures des agents (ResNet, MobileNet...)
    ├── data/         jeux de données et pipelines tf.data
    ├── training/     optimiseurs, checkpoints, boucle d'expérience
    ├── evaluation/   métriques et résumés sur les graines
    └── utils/        config des expériences, choix du GPU, journal

Chaque famille (losses, consensus, modèles, optimiseurs, jeux de données) a un registre :
on ajoute un élément avec un décorateur (@register_loss, @register_strategy,
@register_model, @register_optimizer, @register_dataset), sans toucher au reste du code.

Raccourcis :
    from dml import MultiAgentModel, AgentNetwork, load_dataset
"""
from dml.core.multiagent import MultiAgentModel
from dml.data import load_dataset
from dml.models import AgentNetwork

__all__ = ["MultiAgentModel", "AgentNetwork", "load_dataset"]
