"""
Architectures des agents.

    from models import AgentNetwork, available_models
    agent = AgentNetwork(arch="resnet32", num_classes=10)

Chaque famille d'architectures a son module (resnet.py, mobilenet.py). Pour en
ajouter une, créer un module qui enregistre ses backbones avec @register_model,
puis l'importer ci-dessous.
"""
from .registry import available_models, get_backbone, register_model
from . import resnet, mobilenet  # noqa: F401  (enregistre les architectures)
from .agent import AgentNetwork

__all__ = ["AgentNetwork", "available_models", "get_backbone", "register_model"]
