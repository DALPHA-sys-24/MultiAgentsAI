"""
Multi-Agent Model pour la classification collaborative d'images.

Structure:
- loss_type: "leave_one_out", "global_interaction", "vote_random", "majority_hard", "majority_loo"
- strategy: "vote", "majority", "simplicial", "k_fold"

Les callbacks (AgentHistoryCallback, ConvergenceTracker) sont dans callbacks.py ;
ils restent importables depuis ce module.
"""
from typing import Callable, Dict, List, Union, Optional

import numpy as np
import tensorflow as tf

from callbacks import AgentHistoryCallback, ConvergenceTracker  # noqa: F401  (compatibilité)
from models import AgentNetwork
from optimizers import choix_optimiseur

# Nom du loss_type -> nom de la méthode qui le calcule.
# Ajouter une loss : écrire une méthode compute_loss_<nom>(outputs, y, log_outputs) et l'ajouter ici.
LOSS_FUNCTIONS = {
    "leave_one_out": "compute_loss_leave_one_out",
    "global_interaction": "compute_loss_global_interaction",
    "vote_random": "compute_loss_vote_random",
    "majority_hard": "compute_loss_majority_hard",
    "majority_loo": "compute_loss_majority_loo",
}
STRATEGIES = ("vote", "majority", "simplicial", "k_fold")

_EPS = 1e-7  # même seuil que Keras : log-probabilités bornées à log(_EPS) dans les CE
_LOG_EPS = float(np.log(_EPS))


def _make_optimizer(cfg):
    """Nom enregistré -> choix_optimiseur ; dict Keras ou instance acceptés aussi."""
    if isinstance(cfg, str):
        return choix_optimiseur(cfg)
    if isinstance(cfg, tf.keras.optimizers.Optimizer):
        return cfg
    return tf.keras.optimizers.get(cfg)


class MultiAgentModel(tf.keras.Model):
    """
    Modele multi-agents pour la classification collaborative.

    Loss types (L_k = CE(y, y^k)), toutes de la forme L_k * (L_k + alpha * I_k) :
        - "leave_one_out": I_k = (1-w_k) * CE(y^cons_{-k}, y^k)
        - "global_interaction": I_k = CE(y^cons, y^k) + CE(y, y^cons)
        - "vote_random": I_k = (1-w_k) * CE(y^{vote}, y^k)
        - "majority_hard": I_k = (1-w_k) * CE(y^{majority}, y^k)
        - "majority_loo":  I_k = (1-w_k) * CE(y^{majority_{-k}}, y^k)
    Avec un seul agent, toutes se réduisent à la cross-entropy L_k.

    Les cross-entropies sont calculées à partir des log-probabilités (log_softmax des
    logits), bornées à log(1e-7) comme le fait Keras sur les probabilités. Cette borne
    est nécessaire : sans elle, la forme multiplicative L_k * (L_k + ...) donne des
    gradients énormes sur les exemples très mal classés et l'entraînement diverge.

    Strategies:
        - "vote": Selection aleatoire d'un agent
        - "majority": Vote majoritaire
        - "simplicial": Moyenne ponderee par confiance
        - "k_fold": Vote majoritaire sur les consensus leave-one-out

    Args:
        num_agents: Nombre d'agents K.
        archs: Nom(s) de modèles enregistrés (voir models.available_models()). Un str = même
               architecture pour tous ; une liste de K str = agents hétérogènes,
               ex. ["resnet32", "mobilenetv2"].
        num_classes: Nombre de classes.
        dropout_rate: Dropout avant la couche de sortie de chaque agent.
        widths: Multiplicateur de largeur des agents (float pour tous, ou liste de K float).

    Entrées acceptées par call/fit :
        - (batch, H, W, C) : la même image est donnée à tous les agents (DML classique) ;
        - (batch, K, H, W, C) : une vue par agent (format d'origine, inputs[:, i]).
    """

    def __init__(
        self,
        num_agents: int,
        archs: Union[str, List[str]] = "resnet20",
        num_classes: int = 10,
        dropout_rate: float = 0.0,
        widths: Union[float, List[float]] = 1.0,
        name: str = "multi_agent",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        archs_list = [archs] * num_agents if isinstance(archs, str) else list(archs)
        if len(archs_list) != num_agents:
            raise ValueError(f"archs a {len(archs_list)} éléments mais num_agents={num_agents}.")
        widths_list = [float(widths)] * num_agents if isinstance(widths, (int, float)) else list(widths)
        if len(widths_list) != num_agents:
            raise ValueError(f"widths a {len(widths_list)} éléments mais num_agents={num_agents}.")

        self._num_agents = num_agents
        self._archs = archs_list
        self._widths = widths_list
        self._num_classes = num_classes
        self._dropout_rate = dropout_rate
        self._agents: List = []
        self._optimizers: List = []

        # Parametres (initialisés dans compile)
        self._loss_type: Optional[str] = None
        self._strategy: Optional[str] = None
        self._alpha: Optional[float] = None
        self._temperature: Optional[float] = None

        # Trackers internes Keras
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._acc_tracker = tf.keras.metrics.Mean(name="accuracy")
        self._agent_loss_trackers = [
            tf.keras.metrics.Mean(name=f"loss_agent{i+1}") for i in range(num_agents)
        ]
        self._agent_acc_trackers = [
            tf.keras.metrics.Mean(name=f"accuracy_agent{i+1}") for i in range(num_agents)
        ]

    def build(self, input_shape):
        # Les 3 dernières dimensions sont (H, W, C), que l'entrée soit partagée ou par agent
        image_shape = tuple(input_shape[-3:])
        self._agents = []
        for i, (arch, width) in enumerate(zip(self._archs, self._widths)):
            agent = AgentNetwork(
                arch=arch,
                num_classes=self._num_classes,
                dropout_rate=self._dropout_rate,
                width=width,
                name=f"{self.name}_agent{i+1}",
            )
            agent.build((None,) + image_shape)
            self._agents.append(agent)
        super().build(input_shape)

    def forward_log_probs(self, inputs, training=False):
        """Log-probabilités de chaque agent : (batch, K, num_classes)."""
        if not self.built:  # train_step n'appelle pas self(x), donc construire ici au besoin
            self.build(inputs.shape)
        if len(inputs.shape) == 4:  # (batch, H, W, C) : même image pour tous les agents
            logits = [agent(inputs, training=training, return_logits=True) for agent in self._agents]
        else:                        # (batch, K, H, W, C) : une vue par agent
            logits = [agent(inputs[:, i], training=training, return_logits=True)
                      for i, agent in enumerate(self._agents)]
        return tf.nn.log_softmax(tf.stack(logits, axis=1), axis=-1)

    def call(self, inputs, training=False):
        """Probabilités de chaque agent : (batch, K, num_classes)."""
        return tf.exp(self.forward_log_probs(inputs, training=training))

    def compile(
        self,
        optimizers_config: Union[str, Dict, List, Callable] = "adam",
        loss_type: str = "global_interaction",
        strategy: str = "simplicial",
        alpha: float = 1.0,
        temperature: float = 1.0,
        **kwargs
    ):
        """
        Configure le modele.

        Args:
            optimizers_config: Optimiseur(s) des agents :
                - un nom (voir optimizers.available_optimizers()), le même pour tous ;
                - une liste, un élément par agent : nom, dict Keras ou instance ;
                - une fonction sans argument renvoyant un nouvel optimiseur (appelée
                  une fois par agent).
                Ne pas passer une instance unique : elle serait partagée par tous les agents.
                Avec schedule et weight decay : optimizers.build_agent_optimizers.
            loss_type: "leave_one_out", "global_interaction", "vote_random", "majority_hard", "majority_loo"
            strategy: "vote", "majority", "simplicial", "k_fold"
            alpha: Poids du terme d'interaction I_k, pour toutes les losses.
            temperature: Température tau pour le softmax des w_k.
            kwargs: Transmis à Model.compile (ex. jit_compile=True pour XLA).
        """
        if loss_type not in LOSS_FUNCTIONS:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choix : {list(LOSS_FUNCTIONS)}")
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Choix : {list(STRATEGIES)}")

        self._loss_type = loss_type
        self._strategy = strategy
        self._alpha = alpha
        self._temperature = temperature

        if isinstance(optimizers_config, (list, tuple)):
            if len(optimizers_config) != self._num_agents:
                raise ValueError("optimizers_config doit contenir un optimiseur par agent.")
            self._optimizers = [_make_optimizer(cfg) for cfg in optimizers_config]
        elif callable(optimizers_config) and not isinstance(optimizers_config, tf.keras.optimizers.Optimizer):
            self._optimizers = [optimizers_config() for _ in range(self._num_agents)]
        else:
            self._optimizers = [_make_optimizer(optimizers_config) for _ in range(self._num_agents)]

        super().compile(**kwargs)

    # =========================================================================
    # CONFIDENCE WEIGHTS
    # =========================================================================

    @staticmethod
    def _log_probs(outputs, log_outputs=None):
        """
        Log-probabilités utilisées dans les cross-entropies, bornées à log(1e-7).

        Identique à la CE Keras sur probabilités tronquées (valeur et gradient), mais
        calculé depuis les logits quand ils sont disponibles.
        """
        if log_outputs is None:
            return tf.math.log(tf.clip_by_value(tf.cast(outputs, tf.float32), _EPS, 1.0))
        return tf.maximum(tf.cast(log_outputs, tf.float32), _LOG_EPS)

    def compute_log_confidence_weights(self, outputs, log_outputs=None):
        """log w_k = log_softmax(-H_k / tau), (batch, k)."""
        p = tf.cast(outputs, tf.float32)
        if log_outputs is None:
            # xlogy(p, p) = p * log(p), retourne 0 si p=0 (évite les NaN)
            H_k = -tf.reduce_sum(tf.math.xlogy(p, p), axis=-1)
        else:
            H_k = -tf.reduce_sum(p * tf.cast(log_outputs, tf.float32), axis=-1)
        return tf.nn.log_softmax(-H_k / self._temperature, axis=-1)

    def compute_confidence_weights(self, outputs, log_outputs=None):
        """
        w_k = softmax(-H_k / tau)  où  H_k = entropie de y^k.

        Args:
            outputs: (batch, k, num_classes)
            log_outputs: log-probabilités correspondantes (optionnel, plus précis).
        Returns:
            weights: (batch, k)
        """
        return tf.exp(self.compute_log_confidence_weights(outputs, log_outputs))

    # =========================================================================
    # CONSENSUS FUNCTIONS
    # =========================================================================

    def consensus_vote(self, outputs):
        """Selection aleatoire d'un agent pour chaque exemple du batch."""
        preds = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        batch_size = tf.shape(preds)[0]
        idx = tf.random.uniform([batch_size], 0, self._num_agents, dtype=tf.int32)
        return tf.gather_nd(preds, tf.stack([tf.range(batch_size), idx], axis=1))

    def consensus_majority(self, outputs):
        """Vote majoritaire des K agents."""
        preds = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        votes = tf.one_hot(preds, depth=self._num_classes)
        return tf.argmax(tf.reduce_sum(votes, axis=1), axis=-1, output_type=tf.int32)

    def consensus_simplicial(self, outputs, weights, return_proba=False):
        """Moyenne pondérée : y^cons = sum_k w_k * y^k."""
        proba = tf.reduce_sum(outputs * tf.expand_dims(weights, -1), axis=1)
        return proba if return_proba else tf.argmax(proba, axis=-1, output_type=tf.int32)

    def consensus_k_fold(self, outputs, weights):
        """Vote majoritaire sur les K consensus leave-one-out."""
        loo = self.compute_leave_one_out_consensus(outputs, weights)
        preds = tf.argmax(loo, axis=-1, output_type=tf.int32)
        votes = tf.one_hot(preds, depth=self._num_classes)
        return tf.argmax(tf.reduce_sum(votes, axis=1), axis=-1, output_type=tf.int32)

    def compute_consensus(self, outputs):
        """Dispatch vers la stratégie configurée."""
        if self._strategy == "vote":
            return self.consensus_vote(outputs)
        if self._strategy == "majority":
            return self.consensus_majority(outputs)
        w = self.compute_confidence_weights(outputs)
        if self._strategy == "simplicial":
            return self.consensus_simplicial(outputs, w, return_proba=True)
        return self.consensus_k_fold(outputs, w)

    def compute_consensus_predictions(self, outputs):
        """Retourne un tenseur de classes (batch,) pour le calcul d'accuracy."""
        cons = self.compute_consensus(outputs)
        return tf.argmax(cons, axis=-1, output_type=tf.int32) if len(cons.shape) == 2 else cons

    def compute_leave_one_out_consensus(self, outputs, weights):
        """
        y^cons_{-k} = sum_{j≠k} w_j * y^j / sum_{j≠k} w_j

        Returns: (batch, k, num_classes)
        """
        w_exp = tf.expand_dims(weights, -1)
        weighted = outputs * w_exp
        total_w = tf.reduce_sum(weighted, axis=1, keepdims=True)
        total_s = tf.reduce_sum(weights, axis=1, keepdims=True)
        loo_w = total_w - weighted
        loo_s = total_s - weights
        return tf.math.divide_no_nan(loo_w, tf.expand_dims(loo_s, -1))

    # =========================================================================
    # LOSS FUNCTIONS
    # Toutes renvoient une liste de K scalaires (une loss par agent). Les CE sont
    # calculées pour tous les agents à la fois, sous forme (batch, K).
    # =========================================================================

    def _ce_labels(self, labels, log_p):
        """CE(label, y^k) pour chaque agent. labels : (batch,) ou (batch, K). Renvoie (batch, K)."""
        one_hot = tf.one_hot(labels, depth=self._num_classes)
        if len(one_hot.shape) == 2:
            one_hot = one_hot[:, None, :]
        return -tf.reduce_sum(one_hot * log_p, axis=-1)

    @staticmethod
    def _ce_soft(targets, log_p):
        """CE(cible douce, y^k) pour chaque agent. targets : (batch, C) ou (batch, K, C)."""
        if len(targets.shape) == 2:
            targets = targets[:, None, :]
        return -tf.reduce_sum(targets * log_p, axis=-1)

    def _interaction_losses(self, l_k, interaction):
        """L^k = mean_batch( L_k * (L_k + alpha * interaction_k) ), renvoyée comme liste de K scalaires."""
        return tf.unstack(tf.reduce_mean(l_k * (l_k + self._alpha * interaction), axis=0))

    def compute_loss_leave_one_out(self, outputs, y_true, log_outputs=None):
        """L^k = L_k * (L_k + alpha * (1 - w_k) * CE(y^cons_{-k},y^k))"""
        log_p = self._log_probs(outputs, log_outputs)
        w_k = self.compute_confidence_weights(outputs, log_outputs)
        loo = self.compute_leave_one_out_consensus(outputs, w_k)
        l_k = self._ce_labels(y_true, log_p)
        return self._interaction_losses(l_k, (1.0 - w_k) * self._ce_soft(loo, log_p))

    def compute_loss_global_interaction(self, outputs, y_true, log_outputs=None):
        """L^k = L_k * (L_k + alpha * (CE(y^cons, y^k) + CE(y, y^cons)))"""
        log_p = self._log_probs(outputs, log_outputs)
        log_w = self.compute_log_confidence_weights(outputs, log_outputs)
        cons = self.consensus_simplicial(outputs, tf.exp(log_w), return_proba=True)
        # log y^cons = logsumexp_k(log w_k + log y^k), borné comme les autres CE
        raw_log_p = log_p if log_outputs is None else tf.cast(log_outputs, tf.float32)
        log_cons = tf.maximum(tf.reduce_logsumexp(log_w[..., None] + raw_log_p, axis=1), _LOG_EPS)
        l_glob = self._ce_labels(y_true, log_cons[:, None, :])   # (batch, 1)
        l_loc = self._ce_soft(cons, log_p)                        # (batch, K)
        l_k = self._ce_labels(y_true, log_p)
        return self._interaction_losses(l_k, l_loc + l_glob)

    def compute_loss_vote_random(self, outputs, y_true, log_outputs=None):
        """L^k = L_k * (L_k + alpha * (1 - w_k) * CE(y^{vote}, y^k))"""
        log_p = self._log_probs(outputs, log_outputs)
        w_k = self.compute_confidence_weights(outputs, log_outputs)
        batch_size = tf.shape(outputs)[0]
        idx = tf.random.uniform([batch_size], 0, self._num_agents, dtype=tf.int32)
        # Classe prédite par l'agent tiré au sort — label dur comme y_true
        agent_probs = tf.gather_nd(outputs, tf.stack([tf.range(batch_size), idx], axis=1))
        vote = tf.argmax(agent_probs, axis=-1, output_type=tf.int32)
        l_k = self._ce_labels(y_true, log_p)
        return self._interaction_losses(l_k, (1.0 - w_k) * self._ce_labels(vote, log_p))

    def compute_loss_majority_hard(self, outputs, y_true, log_outputs=None):
        """L^k = L_k * (L_k + alpha * (1 - w_k) * CE(y^{majority}, y^k))"""
        log_p = self._log_probs(outputs, log_outputs)
        w_k = self.compute_confidence_weights(outputs, log_outputs)
        majority = self.consensus_majority(outputs)  # label dur (batch,)
        l_k = self._ce_labels(y_true, log_p)
        return self._interaction_losses(l_k, (1.0 - w_k) * self._ce_labels(majority, log_p))

    def compute_loss_majority_loo(self, outputs, y_true, log_outputs=None):
        """L^k = L_k * (L_k + alpha * (1 - w_k) * CE(y^majority_{-k}, y^k))

        y^majority_{-k} = vote majoritaire des K-1 agents (agent k exclu).
        """
        log_p = self._log_probs(outputs, log_outputs)
        w_k = self.compute_confidence_weights(outputs, log_outputs)
        preds = tf.argmax(outputs, axis=-1, output_type=tf.int32)          # (batch, K)
        votes = tf.one_hot(preds, depth=self._num_classes)                  # (batch, K, C)
        votes_loo = tf.reduce_sum(votes, axis=1, keepdims=True) - votes     # (batch, K, C)
        majority_loo = tf.argmax(votes_loo, axis=-1, output_type=tf.int32)  # (batch, K)
        l_k = self._ce_labels(y_true, log_p)
        return self._interaction_losses(l_k, (1.0 - w_k) * self._ce_labels(majority_loo, log_p))

    def compute_losses(self, outputs, y_true, log_outputs=None):
        """Dispatch vers le type de loss configuré. Renvoie une loss par agent."""
        y = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        if self._num_agents == 1:
            log_p = self._log_probs(outputs, log_outputs)
            return [tf.reduce_mean(self._ce_labels(y, log_p)[:, 0])]
        if self._loss_type not in LOSS_FUNCTIONS:
            raise ValueError(f"Unknown loss_type: {self._loss_type}")
        return getattr(self, LOSS_FUNCTIONS[self._loss_type])(outputs, y, log_outputs)

    # =========================================================================
    # TRAIN / TEST STEPS
    # =========================================================================

    @property
    def metrics(self):
        return (
            [self._loss_tracker, self._acc_tracker]
            + self._agent_loss_trackers
            + self._agent_acc_trackers
        )

    def _update_metrics(self, outputs, y, losses):
        """Met à jour les trackers (loss et accuracy par agent et du consensus)."""
        y_i = tf.cast(tf.reshape(y, [-1]), tf.int32)
        preds = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        correct = tf.cast(tf.equal(preds, y_i[:, None]), tf.float32)
        for i, loss in enumerate(losses):
            self._agent_loss_trackers[i].update_state(loss)
            self._agent_acc_trackers[i].update_state(tf.reduce_mean(correct[:, i]))

        cons_preds = self.compute_consensus_predictions(outputs)
        self._loss_tracker.update_state(tf.reduce_mean(tf.stack(losses)))
        self._acc_tracker.update_state(
            tf.reduce_mean(tf.cast(tf.equal(cons_preds, y_i), tf.float32))
        )
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            log_outputs = self.forward_log_probs(x, training=True)
            outputs = tf.exp(log_outputs)
            losses = self.compute_losses(outputs, y, log_outputs)

        # Chaque agent n'est mis à jour qu'avec sa propre loss
        for agent, opt, loss in zip(self._agents, self._optimizers, losses):
            grads = tape.gradient(loss, agent.trainable_variables)
            opt.apply_gradients(zip(grads, agent.trainable_variables))
        del tape
        return self._update_metrics(outputs, y, losses)

    def test_step(self, data):
        x, y = data
        log_outputs = self.forward_log_probs(x, training=False)
        outputs = tf.exp(log_outputs)
        losses = self.compute_losses(outputs, y, log_outputs)
        return self._update_metrics(outputs, y, losses)

    # =========================================================================
    # EVALUATE / PREDICT / SUMMARY
    # =========================================================================

    def evaluate(self, *args, return_dict=True, **kwargs):
        """Wrapper Keras-compatible autour de Model.evaluate."""
        return super().evaluate(*args, return_dict=return_dict, **kwargs)

    def predict(self, x, return_agents=False, return_probabilities=False, **kwargs):
        """
        Effectue une inférence et retourne les prédictions du consensus.

        Args:
            x: Données d'entrée.
            return_agents: Si True, inclut les prédictions par agent.
            return_probabilities: Si True, inclut les probabilités brutes.

        Returns:
            dict avec 'consensus' et optionnellement 'agents',
            'consensus_probabilities', 'agents_probabilities'.
        """
        outputs = super().predict(x, **kwargs)  # (n_samples, num_agents, num_classes)
        consensus = self.compute_consensus(outputs)

        if len(consensus.shape) == 2:  # probabilités
            consensus_probs = np.asarray(consensus)
            consensus_preds = np.argmax(consensus_probs, axis=-1)
        else:
            consensus_preds = np.asarray(consensus)
            consensus_probs = None

        result = {"consensus": consensus_preds}
        if return_probabilities and consensus_probs is not None:
            result["consensus_probabilities"] = consensus_probs
        if return_agents:
            result["agents"] = np.argmax(outputs, axis=-1)
        if return_probabilities:
            result["agents_probabilities"] = outputs
        return result

    def summary(self, **kwargs):
        """Affiche un résumé de l'architecture du modele multi-agents."""
        print("=" * 60)
        print(f"MultiAgentModel: {self.name}")
        print("=" * 60)
        print(f"Nombre d'agents  : {self._num_agents}")
        print(f"Nombre de classes: {self._num_classes}")
        print(f"Loss type        : {self._loss_type}")
        print(f"Strategy         : {self._strategy}")
        print(f"Alpha            : {self._alpha}")
        print(f"Temperature      : {self._temperature}")
        print("-" * 60)

        total_params, trainable_params = 0, 0
        for i, agent in enumerate(self._agents):
            print(f"\n[Agent {i+1}] - {agent.name}")
            print(f"  arch        : {self._archs[i]} (width={self._widths[i]:g})")
            if agent.built:
                a_total = sum(np.prod(w.shape) for w in agent.weights)
                a_train = sum(np.prod(w.shape) for w in agent.trainable_weights)
                print(f"  Parameters  : {a_total:,} (trainable: {a_train:,})")
                total_params += a_total
                trainable_params += a_train
            else:
                print("  (not built yet)")

        print("\n" + "-" * 60)
        print(f"Total parameters     : {total_params:,}")
        print(f"Trainable parameters : {trainable_params:,}")
        print("=" * 60)

    # =========================================================================
    # PROPERTIES / CONFIG
    # =========================================================================

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def agents(self):
        return self._agents

    @property
    def archs(self):
        return self._archs

    def get_config(self):
        return {
            "num_agents": self._num_agents,
            "archs": self._archs,
            "widths": self._widths,
            "num_classes": self._num_classes,
            "dropout_rate": self._dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
