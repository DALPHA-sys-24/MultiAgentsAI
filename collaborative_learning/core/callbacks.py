"""
Callbacks de suivi de l'entraînement multi-agents.

- ConvergenceTracker : distance des poids de chaque agent à des poids de référence w*.
- AgentHistoryCallback : historique des losses, accuracies, poids de confiance et entropies.
- ConfidenceWeightsCallback : poids de confiance w_k à chaque époque, sur tout un ensemble
  (validation, train), au total et par classe, enregistrés dans un fichier .npz.
"""
import os
import pickle
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class ConvergenceTracker(tf.keras.callbacks.Callback):
    """
    Mesure la vitesse de convergence des poids de chaque agent vers w*.

    Args:
        reference_weights: Poids w* de chaque agent (liste de listes de np.ndarray).
            None = poids finaux, les distances sont alors calculées en fin d'entraînement.
        snapshot_dir: Utilisé seulement si reference_weights est None. Les poids de chaque
            epoch sont écrits dans ce dossier au lieu d'être gardés en RAM (indispensable
            pour les gros modèles : ResNet-18 x 4 agents x 200 epochs ≈ 35 Go en RAM).
            "auto" = dossier temporaire.
        keep_snapshots: Garde aussi les poids de chaque epoch en RAM dans
            history["weights_snapshots"] (comportement de l'ancienne version).
    """

    def __init__(
        self,
        reference_weights: Optional[List[List[np.ndarray]]] = None,
        snapshot_dir: Optional[str] = None,
        keep_snapshots: bool = False,
    ):
        super().__init__()
        self.reference_weights = reference_weights
        self.snapshot_dir = snapshot_dir
        self.keep_snapshots = keep_snapshots
        self.history = {"epoch": [], "distances": [], "rates": [], "weights_snapshots": []}
        self._pending = []  # snapshots (en RAM ou chemins de fichiers) en attente de w*

    def _get_agent_weights(self):
        return [[w.copy() for w in agent.get_weights()] for agent in self.model.agents]

    @staticmethod
    def _compute_distance(w_curr, w_ref):
        return np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(w_curr, w_ref)))

    def _append_distances(self, weights):
        distances = [self._compute_distance(c, r) for c, r in zip(weights, self.reference_weights)]
        prev = self.history["distances"][-1] if self.history["distances"] else None
        self.history["distances"].append(distances)
        if prev:
            self.history["rates"].append([d / p if p > 1e-10 else 0 for d, p in zip(distances, prev)])

    def _store(self, weights):
        if self.snapshot_dir is None:
            self._pending.append(weights)
            return
        if self.snapshot_dir == "auto":
            self.snapshot_dir = tempfile.mkdtemp(prefix="convergence_")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(self.snapshot_dir, f"epoch_{len(self._pending) + 1:04d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(weights, f)
        self._pending.append(path)

    def _load(self, item):
        if isinstance(item, str):
            with open(item, "rb") as f:
                return pickle.load(f)
        return item

    def on_epoch_end(self, epoch, logs=None):
        current = self._get_agent_weights()
        self.history["epoch"].append(epoch + 1)
        if self.keep_snapshots:
            self.history["weights_snapshots"].append(current)
        if self.reference_weights is not None:
            self._append_distances(current)
        else:
            self._store(current)

    def on_train_end(self, logs=None):
        if self.reference_weights is not None or not self._pending:
            return
        self.reference_weights = self._load(self._pending[-1])
        self.history["distances"], self.history["rates"] = [], []
        for item in self._pending:
            self._append_distances(self._load(item))
        self._pending = []

    def get_convergence_summary(self):
        if not self.history["distances"]:
            return {}
        avg_rates = []
        if self.history["rates"]:
            avg_rates = np.mean(self.history["rates"], axis=0).tolist()
        return {
            "final_distances": self.history["distances"][-1],
            "avg_convergence_rates": avg_rates,
            "num_epochs": len(self.history["epoch"]),
        }

    def plot_convergence(self, save_path=None, filename="convergence.png"):
        if not self.history["distances"]:
            print("No convergence data to plot")
            return None

        distances = np.array(self.history["distances"])
        epochs = self.history["epoch"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for i in range(distances.shape[1]):
            axes[0].plot(epochs, distances[:, i], label=f"Agent {i+1}", marker='o', markersize=3)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Distance to w*")
        axes[0].set_title("Convergence des poids")
        axes[0].legend()
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)

        if self.history["rates"]:
            rates = np.array(self.history["rates"])
            for i in range(rates.shape[1]):
                axes[1].plot(epochs[1:], rates[:, i], label=f"Agent {i+1}", marker='o', markersize=3)
            axes[1].axhline(y=1.0, color='r', linestyle='--', label='Seuil')
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Rate")
            axes[1].set_title("Taux de convergence")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            out_path = os.path.join(save_path, filename)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return out_path
        plt.show()
        return None


class AgentHistoryCallback(tf.keras.callbacks.Callback):
    """
    Callback unifié pour enregistrer l'historique détaillé des agents.

    Regroupe le suivi des métriques de performance (loss, accuracy, seuils)
    et des métriques propres aux agents (poids de confiance, entropies).

    Enregistre par epoch :
    - Loss et accuracy du consensus (train et val)
    - Loss et accuracy de chaque agent (train et val)
    - Poids de confiance et entropie de chaque agent (si validation_data fourni)
    - Epoch de franchissement de seuils d'accuracy

    Args:
        validation_data: Tuple (x_val, y_val) pour calculer les poids de confiance.
        thresholds: Seuils d'accuracy à surveiller (ex: [0.8, 0.9, 0.95]).

    Usage:
        tracker = AgentHistoryCallback(validation_data=(x_val, y_val))
        model.fit(x_train, y_train, callbacks=[tracker])

        tracker.summary()
        tracker.plot_losses()
        tracker.plot_accuracies()
        tracker.plot_weights()
        tracker.plot_overview()
        df = tracker.to_dataframe()
    """

    def __init__(
        self,
        validation_data=None,
        thresholds: Optional[List[float]] = None,
    ):
        super().__init__()
        self.validation_data = validation_data
        self.thresholds = thresholds if thresholds is not None else [0.8, 0.9, 0.95]
        self.history: Dict = {}
        self._num_agents: Optional[int] = None
        self._epochs_to_threshold: Dict = {}

    # -------------------------------------------------------------------------
    # Keras hooks
    # -------------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        """Initialise toutes les listes d'historique avant l'entraînement."""
        self._num_agents = self.model.num_agents
        self._epochs_to_threshold = {t: None for t in self.thresholds}

        self.history = {
            'epoch': [],
            'loss': [],
            'accuracy_consensus': [],
            'val_loss': [],
            'val_accuracy_consensus': [],
            'weight_sum': [],
        }
        for i in range(self._num_agents):
            self.history[f'loss_agent{i+1}'] = []
            self.history[f'accuracy_agent{i+1}'] = []
            self.history[f'val_loss_agent{i+1}'] = []
            self.history[f'val_accuracy_agent{i+1}'] = []
            self.history[f'weight_agent{i+1}'] = []
            self.history[f'entropy_agent{i+1}'] = []

    def on_epoch_end(self, epoch, logs=None):
        """Enregistre les métriques à la fin de chaque epoch."""
        logs = logs or {}

        self.history['epoch'].append(epoch + 1)
        self.history['loss'].append(logs.get('loss', 0))
        # Keras expose 'accuracy' (pas 'accuracy_consensus')
        self.history['accuracy_consensus'].append(logs.get('accuracy', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        val_acc = logs.get('val_accuracy', 0)
        self.history['val_accuracy_consensus'].append(val_acc)

        # Métriques par agent issues des logs Keras
        for i in range(self._num_agents):
            self.history[f'loss_agent{i+1}'].append(logs.get(f'loss_agent{i+1}', 0))
            self.history[f'accuracy_agent{i+1}'].append(logs.get(f'accuracy_agent{i+1}', 0))
            self.history[f'val_loss_agent{i+1}'].append(logs.get(f'val_loss_agent{i+1}', 0))
            self.history[f'val_accuracy_agent{i+1}'].append(logs.get(f'val_accuracy_agent{i+1}', 0))

        # Franchissement de seuils
        for t in self.thresholds:
            if self._epochs_to_threshold[t] is None and val_acc >= t:
                self._epochs_to_threshold[t] = epoch + 1

        # Poids de confiance et entropies (nécessitent validation_data)
        if self.validation_data is not None:
            x_val, _ = self.validation_data
            sample_size = min(1000, len(x_val))
            indices = np.random.choice(len(x_val), sample_size, replace=False)
            outputs = self.model(x_val[indices], training=False)

            p = tf.cast(outputs, tf.float32)
            H_k = -tf.reduce_sum(tf.math.xlogy(p, p), axis=-1)  # (batch, k)
            mean_entropy = tf.reduce_mean(H_k, axis=0).numpy()

            weights = self.model.compute_confidence_weights(outputs)
            mean_weights = tf.reduce_mean(weights, axis=0).numpy()

            self.history['weight_sum'].append(float(np.sum(mean_weights)))
            for i in range(self._num_agents):
                self.history[f'weight_agent{i+1}'].append(float(mean_weights[i]))
                self.history[f'entropy_agent{i+1}'].append(float(mean_entropy[i]))

    # -------------------------------------------------------------------------
    # Données
    # -------------------------------------------------------------------------

    def get_summary(self) -> Dict:
        """Retourne un dictionnaire de synthèse des métriques finales."""
        if not self.history.get('epoch'):
            return {}
        val_accs = self.history['val_accuracy_consensus']
        return {
            "final_train_accuracy": self.history['accuracy_consensus'][-1],
            "final_val_accuracy": val_accs[-1],
            "final_train_loss": self.history['loss'][-1],
            "final_val_loss": self.history['val_loss'][-1],
            "best_val_accuracy": max(val_accs),
            "best_epoch": int(np.argmax(val_accs) + 1),
            "epochs_to_threshold": self._epochs_to_threshold.copy(),
            "num_epochs": len(self.history['epoch']),
        }

    def compare_to_baseline(self, baseline_summary: Dict) -> Dict:
        """Compare les performances au résultat d'un baseline (issu de get_summary)."""
        current = self.get_summary()
        comparison = {}

        if "final_val_accuracy" in baseline_summary:
            comparison["accuracy_diff"] = (
                current.get("final_val_accuracy", 0) - baseline_summary["final_val_accuracy"]
            )

        for t in self.thresholds:
            baseline_epoch = baseline_summary.get("epochs_to_threshold", {}).get(t)
            current_epoch = current.get("epochs_to_threshold", {}).get(t)

            if baseline_epoch and current_epoch:
                comparison[f"speedup_{int(t * 100)}pct"] = baseline_epoch / current_epoch
            elif current_epoch and not baseline_epoch:
                comparison[f"speedup_{int(t * 100)}pct"] = float('inf')
            else:
                comparison[f"speedup_{int(t * 100)}pct"] = None

        return comparison

    def to_dataframe(self):
        """Convertit l'historique en DataFrame pandas."""
        import pandas as pd
        return pd.DataFrame(self.history)

    # -------------------------------------------------------------------------
    # Visualisations
    # -------------------------------------------------------------------------

    def _save_or_show(self, fig, save_path: Optional[str]):
        """Sauvegarde la figure (PDF + PNG) ou l'affiche."""
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_losses(self, save_path=None, show_agents=True, figsize=(12, 5)):
        """Plot les courbes de loss (train et val, consensus et agents)."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        epochs = self.history['epoch']
        colors = plt.cm.tab10(np.linspace(0, 1, self._num_agents))

        for ax, key, title in [
            (axes[0], 'loss', 'Training Loss'),
            (axes[1], 'val_loss', 'Validation Loss'),
        ]:
            ax.plot(epochs, self.history[key], 'k-', linewidth=2, label='Consensus')
            if show_agents:
                prefix = '' if key == 'loss' else 'val_'
                for i in range(self._num_agents):
                    ax.plot(epochs, self.history[f'{prefix}loss_agent{i+1}'], '--',
                            color=colors[i], alpha=0.7, label=f'Agent {i+1}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_or_show(fig, save_path)
        return fig

    def plot_accuracies(self, save_path=None, show_agents=True, figsize=(12, 5)):
        """Plot les courbes d'accuracy (train et val, consensus et agents)."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        epochs = self.history['epoch']
        colors = plt.cm.tab10(np.linspace(0, 1, self._num_agents))

        for ax, cons_key, agent_prefix, title in [
            (axes[0], 'accuracy_consensus', 'accuracy_agent', 'Training Accuracy'),
            (axes[1], 'val_accuracy_consensus', 'val_accuracy_agent', 'Validation Accuracy'),
        ]:
            ax.plot(epochs, self.history[cons_key], 'k-', linewidth=2, label='Consensus')
            if show_agents:
                for i in range(self._num_agents):
                    ax.plot(epochs, self.history[f'{agent_prefix}{i+1}'], '--',
                            color=colors[i], alpha=0.7, label=f'Agent {i+1}')
            for t in self.thresholds:
                ax.axhline(y=t, linestyle=':', alpha=0.4, label=f'{int(t * 100)}%')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_or_show(fig, save_path)
        return fig

    def plot_weights(self, save_path=None, figsize=(12, 5)):
        """Plot l'évolution des poids de confiance et des entropies par agent."""
        if not self.history.get('weight_agent1'):
            print("No weight data available. Pass validation_data to the callback.")
            return None

        has_entropy = bool(self.history.get('entropy_agent1'))
        fig, axes = plt.subplots(1, 2 if has_entropy else 1, figsize=figsize)
        if not has_entropy:
            axes = [axes]

        epochs = self.history['epoch']
        colors = plt.cm.tab10(np.linspace(0, 1, self._num_agents))

        ax1 = axes[0]
        for i in range(self._num_agents):
            ax1.plot(epochs, self.history[f'weight_agent{i+1}'], '-o',
                     color=colors[i], markersize=4, label=f'Agent {i+1}')
        ax1.axhline(y=1 / self._num_agents, color='gray', linestyle='--', alpha=0.5, label='Uniform')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Weight')
        ax1.set_title('Confidence Weights Evolution')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, min(1, max(
            max(self.history[f'weight_agent{i+1}']) for i in range(self._num_agents)
        ) * 1.2))

        if has_entropy:
            ax2 = axes[1]
            for i in range(self._num_agents):
                ax2.plot(epochs, self.history[f'entropy_agent{i+1}'], '-o',
                         color=colors[i], markersize=4, label=f'Agent {i+1}')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Entropy')
            ax2.set_title('Entropy Evolution (lower = more confident)')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_or_show(fig, save_path)
        return fig

    def plot_overview(self, save_path=None, filename="overview.png"):
        """
        Vue d'ensemble 2×2 : consensus accuracy (avec seuils), accuracy par agent,
        loss train/val, loss de validation.
        """
        if not self.history.get('epoch'):
            print("No data to plot")
            return None

        epochs = self.history['epoch']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Consensus accuracy + seuils
        axes[0, 0].plot(epochs, self.history['accuracy_consensus'], label="Train", marker='o', markersize=3)
        axes[0, 0].plot(epochs, self.history['val_accuracy_consensus'], label="Val", marker='s', markersize=3)
        for t in self.thresholds:
            axes[0, 0].axhline(y=t, linestyle='--', alpha=0.5, label=f'{int(t * 100)}%')
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_title("Consensus Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy par agent (validation)
        agent_val_accs = np.array([
            self.history[f'val_accuracy_agent{i+1}'] for i in range(self._num_agents)
        ]).T  # (epochs, agents)
        for i in range(self._num_agents):
            axes[0, 1].plot(epochs, agent_val_accs[:, i], label=f"Agent {i+1}", marker='o', markersize=3)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Validation Accuracy par Agent")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Loss train + val
        axes[1, 0].plot(epochs, self.history['loss'], label="Train", marker='o', markersize=3)
        axes[1, 0].plot(epochs, self.history['val_loss'], label="Val", marker='s', markersize=3)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Loss de validation seule
        axes[1, 1].plot(epochs, self.history['val_loss'], label="Val Loss",
                        marker='o', markersize=3, color='orange')
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_title("Validation Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            out_path = os.path.join(save_path, filename)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return out_path
        plt.show()
        return None

    def plot_all(self, save_dir=None, prefix="", figsize=(12, 5)):
        """
        Génère toutes les figures.

        Args:
            save_dir: Dossier de sauvegarde (None = afficher).
            prefix: Préfixe pour les noms de fichiers.
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.plot_losses(os.path.join(save_dir, f"{prefix}losses.pdf"), figsize=figsize)
            self.plot_accuracies(os.path.join(save_dir, f"{prefix}accuracies.pdf"), figsize=figsize)
            self.plot_weights(os.path.join(save_dir, f"{prefix}weights.pdf"), figsize=(10, 5))
            self.plot_overview(save_dir, filename=f"{prefix}overview.png")
        else:
            self.plot_losses(figsize=figsize)
            self.plot_accuracies(figsize=figsize)
            self.plot_weights(figsize=(10, 5))
            self.plot_overview()

    def summary(self):
        """Affiche un résumé textuel des métriques finales."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        s = self.get_summary()
        print(f"Epochs       : {s['num_epochs']}")
        print(f"Best val acc : {s['best_val_accuracy']:.4f} (epoch {s['best_epoch']})")
        print(f"\nSeuils atteints :")
        for t, ep in s['epochs_to_threshold'].items():
            print(f"  {int(t * 100)}% → epoch {ep if ep is not None else 'non atteint'}")

        print(f"\nFinal Consensus:")
        print(f"  Train Loss: {s['final_train_loss']:.4f}  |  Train Acc: {s['final_train_accuracy']:.4f}")
        print(f"  Val Loss:   {s['final_val_loss']:.4f}  |  Val Acc:   {s['final_val_accuracy']:.4f}")

        print(f"\nFinal Agents (Validation):")
        for i in range(self._num_agents):
            acc = self.history[f'val_accuracy_agent{i+1}'][-1]
            loss = self.history[f'val_loss_agent{i+1}'][-1]
            w_list = self.history[f'weight_agent{i+1}']
            e_list = self.history[f'entropy_agent{i+1}']
            w_str = f"{w_list[-1]:.3f}" if w_list else "N/A"
            e_str = f"{e_list[-1]:.3f}" if e_list else "N/A"
            print(f"  Agent {i+1}: Acc={acc:.4f}, Loss={loss:.4f}, Weight={w_str}, Entropy={e_str}")

        if self.history.get('weight_agent1'):
            weight_sum = sum(self.history[f'weight_agent{i+1}'][-1] for i in range(self._num_agents))
            print(f"\n  Somme des poids : {weight_sum:.6f} (attendu : 1.0)")
            final_epoch = s['num_epochs']
            print(f"\nÉvolution des poids (epoch 1 → {final_epoch}):")
            for i in range(self._num_agents):
                w0 = self.history[f'weight_agent{i+1}'][0]
                w1 = self.history[f'weight_agent{i+1}'][-1]
                print(f"  Agent {i+1}: {w0:.3f} → {w1:.3f} ({w1 - w0:+.3f})")

        print("=" * 60)


class ConfidenceWeightsCallback(tf.keras.callbacks.Callback):
    """
    Poids de confiance w_k(x) = softmax(-H_k(x) / tau) à la fin de chaque époque, sur TOUTES les
    images de chaque ensemble donné (sans augmentation, avec les poids de fin d'époque).

    Moyennes exactes (image par image), regroupées selon la vraie classe y :
        overall[époque, k]             moyenne de w_k sur l'ensemble
        per_class[époque, c, k]        moyenne de w_k sur les images de classe c
        w_max[époque]                  moyenne de max_k w_k (1/K : uniforme, 1 : un seul agent décide)
        per_class_w_max[époque, c]     la même, par classe
        w_on_correct[époque]           moyenne de sum_k w_k 1[agent k juste]
        class_counts[c]                nombre d'images de classe c

    Les clés <ensemble>_w_agent<k>, <ensemble>_w_max et <ensemble>_w_on_correct sont aussi ajoutées
    aux logs Keras (donc à history.csv et à l'historique) : ce callback doit passer avant CSVLogger
    et le checkpoint. Rien ne change dans l'entraînement (passe d'inférence à part).

    Args:
        datasets: {"val": (x, y), "train": (x, y)} (un ou plusieurs ensembles).
        num_classes: nombre de classes.
        path: fichier .npz réécrit à chaque époque (None = pas de fichier). S'il existe déjà
            (reprise après interruption), les époques déjà calculées sont gardées.
        batch_size: taille des batchs de la passe d'inférence.
    """

    def __init__(self, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]], num_classes: int,
                 path: Optional[str] = None, batch_size: int = 512):
        super().__init__()
        self.datasets = {name: xy for name, xy in datasets.items() if xy is not None and len(xy[0])}
        self.num_classes = num_classes
        self.path = path
        self.batch_size = batch_size
        self.records: Dict[int, Dict[str, np.ndarray]] = {}   # époque (1, 2, ...) -> valeurs
        self.class_counts = {}
        self._step = None

    # --- calcul ----------------------------------------------------------------
    def _batch_fn(self):
        model = self.model

        @tf.function
        def step(x, y):
            log_p = model.forward_log_probs(x, training=False)
            p = tf.exp(log_p)
            w = tf.cast(model.compute_confidence_weights(p, log_p), tf.float32)       # (batch, K)
            preds = tf.argmax(p, axis=-1, output_type=tf.int32)                       # (batch, K)
            y = tf.cast(tf.reshape(y, [-1]), tf.int32)
            correct = tf.cast(tf.equal(preds, y[:, None]), tf.float32)
            onehot = tf.one_hot(y, self.num_classes, dtype=tf.float32)                # (batch, C)
            return (tf.matmul(onehot, w, transpose_a=True),                           # sommes par classe (C, K)
                    tf.reduce_sum(onehot * tf.reduce_max(w, axis=-1)[:, None], 0),     # (C,)
                    tf.reduce_sum(w * correct))
        return step

    def compute(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Moyennes de w_k sur (x, y) avec les poids actuels du modèle."""
        if self._step is None:
            self._step = self._batch_fn()
        k = self.model.num_agents
        sums, max_sums, on_correct = np.zeros((self.num_classes, k)), np.zeros(self.num_classes), 0.0
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        for xb, yb in ds:
            s, m, c = self._step(xb, yb)
            sums += s.numpy()
            max_sums += m.numpy()
            on_correct += float(c)
        counts = np.bincount(np.asarray(y).reshape(-1).astype(int), minlength=self.num_classes)
        n = max(int(counts.sum()), 1)
        safe = np.maximum(counts, 1)
        return {
            "overall": sums.sum(0) / n,
            "per_class": sums / safe[:, None],
            "w_max": np.float64(max_sums.sum() / n),
            "per_class_w_max": max_sums / safe,
            "w_on_correct": np.float64(on_correct / n),
            "class_counts": counts,
        }

    # --- Keras -----------------------------------------------------------------
    def on_train_begin(self, logs=None):
        if self.path and os.path.exists(self.path):     # reprise : époques déjà calculées
            with np.load(self.path) as f:
                saved = {key: f[key] for key in f.files}
            for i, epoch in enumerate(saved.get("epochs", [])):
                self.records[int(epoch)] = {
                    key: saved[key][i] for key in saved
                    if key not in ("epochs", "temperature") and not key.endswith("class_counts")}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        record = {}
        for name, (x, y) in self.datasets.items():
            res = self.compute(x, y)
            self.class_counts[name] = res.pop("class_counts")
            for key, value in res.items():
                record[f"{name}_{key}"] = value
            for i, w in enumerate(res["overall"]):
                logs[f"{name}_w_agent{i + 1}"] = float(w)
            logs[f"{name}_w_max"] = float(res["w_max"])
            logs[f"{name}_w_on_correct"] = float(res["w_on_correct"])
        self.records[epoch + 1] = record
        if self.path:
            self.save(self.path)

    def arrays(self) -> Dict[str, np.ndarray]:
        epochs = sorted(self.records)
        out = {"epochs": np.array(epochs, dtype=np.int32),
               "temperature": np.float64(getattr(self.model, "_temperature", np.nan))}
        if epochs:
            for key in self.records[epochs[-1]]:
                out[key] = np.stack([self.records[e][key] for e in epochs if key in self.records[e]])
        for name, counts in self.class_counts.items():
            out[f"{name}_class_counts"] = counts
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **self.arrays())
        os.replace(tmp, path)
