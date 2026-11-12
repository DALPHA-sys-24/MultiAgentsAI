"""
Multi-Agent Model pour la classification collaborative d'images.

- loss_type : une loss enregistrée dans losses.py ("leave_one_out", "global_interaction",
  "vote_random", "majority_hard", "majority_loo", ...)
- strategy  : une stratégie enregistrée dans consensus.py ("vote", "majority",
  "simplicial", "k_fold", ...)

Les losses et les consensus sont des fonctions enregistrées par leur nom : en ajouter
une ne demande pas de modifier cette classe (voir losses.py et consensus.py).

Les callbacks (AgentHistoryCallback, ConvergenceTracker) sont dans callbacks.py ;
ils restent importables depuis ce module.
"""
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from dml.core import consensus as cons_lib
from dml.core import losses as loss_lib
from dml.core.callbacks import AgentHistoryCallback, ConvergenceTracker  # noqa: F401  (compatibilité)
from dml.models import AgentNetwork
from dml.training.optimizers import choix_optimiseur

# Registres (nom -> fonction). Itérer dessus donne les noms disponibles.
LOSS_FUNCTIONS = loss_lib._LOSSES
STRATEGIES = tuple(cons_lib.available_strategies())


def _make_optimizer(cfg):
    """Nom enregistré -> choix_optimiseur ; dict Keras ou instance acceptés aussi."""
    if isinstance(cfg, str):
        return choix_optimiseur(cfg)
    if isinstance(cfg, tf.keras.optimizers.Optimizer):
        return cfg
    return tf.keras.optimizers.get(cfg)


class _EvalRNG:
    """
    Générateur aléatoire de l'évaluation, remis à zéro au début de chaque evaluate/predict :
    la stratégie "vote" donne alors la même accuracy à chaque évaluation.

    Objet Python simple (pas un attribut tf.Variable du modèle), pour que son état ne
    soit pas compté dans les poids du modèle.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.gen = tf.random.Generator.from_seed(seed)

    def reset(self):
        self.gen.reset_from_seed(self.seed)

    def randint(self, shape, maxval):
        return self.gen.uniform(shape, minval=0, maxval=maxval, dtype=tf.int32)


class MultiAgentModel(tf.keras.Model):
    """
    Modele multi-agents pour la classification collaborative.

    Loss types (L_k = CE(y, y^k)), toutes de la forme L_k * (L_k + alpha * I_k)
    (ou L_k + alpha * I_k avec form="additive") :
        - "leave_one_out": I_k = (1-w_k) * CE(y^cons_{-k}, y^k)
        - "global_interaction": I_k = CE(y^cons, y^k) + CE(y, y^cons)
        - "vote_random": I_k = (1-w_k) * CE(y^{vote}, y^k)
        - "majority_hard": I_k = (1-w_k) * CE(y^{majority}, y^k)
        - "majority_loo":  I_k = (1-w_k) * CE(y^{majority_{-k}}, y^k)
    Avec un seul agent, la loss est la cross-entropy (single_agent_loss="ce") ou la
    même forme avec I_k = 0 (single_agent_loss="form").

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
        self._form: str = "multiplicative"
        self._single_agent_loss: str = "ce"
        self._eval_rng: Optional[_EvalRNG] = None

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
        # Pas super().build() : sous Keras 2 (TF <= 2.15), il rappelle call() et boucle à l'infini
        self.built = True

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
        form: str = "multiplicative",
        single_agent_loss: str = "ce",
        eval_seed: int = 0,
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
            loss_type: nom d'une loss de losses.py.
            strategy: nom d'une stratégie de consensus.py.
            alpha: Poids du terme d'interaction I_k, pour toutes les losses.
            temperature: Température tau pour le softmax des w_k.
            form: "multiplicative" (L_k * (L_k + alpha I_k), défaut) ou "additive" (L_k + alpha I_k).
            single_agent_loss: loss quand K = 1 : "ce" (défaut) ou "form" (même forme, I_k = 0).
            eval_seed: graine du tirage aléatoire de la stratégie "vote" à l'évaluation.
            kwargs: Transmis à Model.compile (ex. jit_compile=True pour XLA).
        """
        loss_lib.get_loss(loss_type)          # erreur claire si le nom est inconnu
        cons_lib.get_strategy(strategy)
        if form not in loss_lib.FORMS:
            raise ValueError(f"form inconnue : {form}. Choix : {loss_lib.FORMS}")
        if single_agent_loss not in loss_lib.SINGLE_AGENT_LOSSES:
            raise ValueError(f"single_agent_loss inconnue : {single_agent_loss}. "
                             f"Choix : {loss_lib.SINGLE_AGENT_LOSSES}")

        self._loss_type = loss_type
        self._strategy = strategy
        self._alpha = alpha
        self._temperature = temperature
        self._form = form
        self._single_agent_loss = single_agent_loss
        self._eval_rng = _EvalRNG(eval_seed)

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
    # CONTEXTES
    # =========================================================================

    def _randint(self, training: bool):
        """Tirage aléatoire : libre à l'entraînement, reproductible à l'évaluation."""
        if training or self._eval_rng is None:
            return None
        return self._eval_rng.randint

    def _consensus_context(self, training: bool = False) -> cons_lib.ConsensusContext:
        return cons_lib.ConsensusContext(
            num_agents=self._num_agents, num_classes=self._num_classes,
            temperature=self._temperature, randint=self._randint(training),
        )

    def _loss_context(self, batch_size, training: bool = False) -> loss_lib.LossContext:
        ctx = loss_lib.LossContext(
            num_agents=self._num_agents, num_classes=self._num_classes,
            temperature=self._temperature, randint=self._randint(training),
            alpha=self._alpha, form=self._form,
        )
        if self._loss_type == "vote_random":
            ctx.vote_idx = ctx.random_agent(batch_size)  # un seul tirage partagé par les K losses
        return ctx

    # =========================================================================
    # CONFIDENCE WEIGHTS / CONSENSUS (raccourcis vers consensus.py)
    # =========================================================================

    def compute_log_confidence_weights(self, outputs, log_outputs=None):
        """log w_k = log_softmax(-H_k / tau), (batch, k)."""
        return cons_lib.log_confidence_weights(outputs, self._temperature, log_outputs)

    def compute_confidence_weights(self, outputs, log_outputs=None):
        """w_k = softmax(-H_k / tau) où H_k = entropie de y^k. (batch, k)."""
        return cons_lib.confidence_weights(outputs, self._temperature, log_outputs)

    def consensus_vote(self, outputs):
        return cons_lib.vote(outputs, self._consensus_context())

    def consensus_majority(self, outputs):
        return cons_lib.majority_labels(outputs, self._num_classes)

    def consensus_simplicial(self, outputs, weights, return_proba=False):
        proba = cons_lib.simplicial_proba(outputs, weights)
        return proba if return_proba else tf.argmax(proba, axis=-1, output_type=tf.int32)

    def consensus_k_fold(self, outputs, weights):
        return cons_lib.majority_labels(cons_lib.leave_one_out_consensus(outputs, weights), self._num_classes)

    def compute_leave_one_out_consensus(self, outputs, weights):
        return cons_lib.leave_one_out_consensus(outputs, weights)

    def compute_consensus(self, outputs, training=False):
        """Consensus de la stratégie configurée : probabilités (batch, C) ou labels (batch,)."""
        outputs = tf.convert_to_tensor(outputs, tf.float32)
        return cons_lib.get_strategy(self._strategy)(outputs, self._consensus_context(training))

    def compute_consensus_predictions(self, outputs, training=False):
        """Retourne un tenseur de classes (batch,) pour le calcul d'accuracy."""
        return cons_lib.consensus_to_labels(self.compute_consensus(outputs, training))

    # =========================================================================
    # LOSS FUNCTIONS (raccourcis vers losses.py)
    # =========================================================================

    def compute_losses(self, outputs, y_true, log_outputs=None, training=False, ctx=None):
        """Loss de chaque agent pour le loss_type configuré : liste de K scalaires."""
        y = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        if ctx is None:
            ctx = self._loss_context(tf.shape(y)[0], training)
        per_agent = loss_lib.agent_losses(self._loss_type, ctx, outputs, y, log_outputs,
                                          single_agent_loss=self._single_agent_loss)
        return tf.unstack(per_agent, num=self._num_agents)

    def _compute_named_loss(self, loss_type, outputs, y_true, log_outputs=None):
        saved, self._loss_type = self._loss_type, loss_type
        try:
            return self.compute_losses(outputs, y_true, log_outputs)
        finally:
            self._loss_type = saved

    def compute_loss_leave_one_out(self, outputs, y_true, log_outputs=None):
        return self._compute_named_loss("leave_one_out", outputs, y_true, log_outputs)

    def compute_loss_global_interaction(self, outputs, y_true, log_outputs=None):
        return self._compute_named_loss("global_interaction", outputs, y_true, log_outputs)

    def compute_loss_vote_random(self, outputs, y_true, log_outputs=None):
        return self._compute_named_loss("vote_random", outputs, y_true, log_outputs)

    def compute_loss_majority_hard(self, outputs, y_true, log_outputs=None):
        return self._compute_named_loss("majority_hard", outputs, y_true, log_outputs)

    def compute_loss_majority_loo(self, outputs, y_true, log_outputs=None):
        return self._compute_named_loss("majority_loo", outputs, y_true, log_outputs)

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

    def _update_metrics(self, outputs, y, losses, training=False):
        """Met à jour les trackers (loss et accuracy par agent et du consensus)."""
        y_i = tf.cast(tf.reshape(y, [-1]), tf.int32)
        preds = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        correct = tf.cast(tf.equal(preds, y_i[:, None]), tf.float32)
        for i, loss in enumerate(losses):
            self._agent_loss_trackers[i].update_state(loss)
            self._agent_acc_trackers[i].update_state(tf.reduce_mean(correct[:, i]))

        cons_preds = self.compute_consensus_predictions(outputs, training)
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
            losses = self.compute_losses(outputs, y, log_outputs, training=True)

        # Chaque agent n'est mis à jour qu'avec sa propre loss. TensorFlow ne parcourt que
        # les opérations entre la loss k et les poids de l'agent k : les réseaux des autres
        # agents ne sont pas redérivés (vérifié : une passe arrière unique n'est pas plus rapide).
        for agent, opt, loss in zip(self._agents, self._optimizers, losses):
            grads = tape.gradient(loss, agent.trainable_variables)
            opt.apply_gradients(zip(grads, agent.trainable_variables))
        del tape
        return self._update_metrics(outputs, y, losses, training=True)

    def test_step(self, data):
        x, y = data
        log_outputs = self.forward_log_probs(x, training=False)
        outputs = tf.exp(log_outputs)
        losses = self.compute_losses(outputs, y, log_outputs, training=False)
        return self._update_metrics(outputs, y, losses, training=False)

    # =========================================================================
    # EVALUATE / PREDICT / SUMMARY
    # =========================================================================

    def _reset_eval_rng(self):
        if self._eval_rng is not None:
            self._eval_rng.reset()

    def evaluate(self, *args, return_dict=True, **kwargs):
        """Model.evaluate, avec le tirage de la stratégie "vote" remis à zéro (résultat reproductible)."""
        self._reset_eval_rng()
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
        self._reset_eval_rng()
        consensus = np.asarray(self.compute_consensus(outputs))

        if consensus.ndim == 2:  # probabilités
            consensus_probs = consensus
            consensus_preds = np.argmax(consensus_probs, axis=-1)
        else:
            consensus_preds = consensus
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
        print(f"Loss type        : {self._loss_type} ({self._form})")
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
