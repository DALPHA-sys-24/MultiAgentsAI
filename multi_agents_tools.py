import tensorflow as tf
import numpy as np
from typing import List,Tuple
import matplotlib.pyplot as plt

# Global Configuration
gpu_devices = tf.config.list_physical_devices('GPU')
on_gpu = len(gpu_devices) > 0
tf.keras.mixed_precision.set_global_policy('float32')

def show_images(images, titles=None):
    """
    Affiche une liste d'images sur une même figure.
    
    Parameters:
    - images : liste de tableaux numpy ou d'images (ex: PIL.Image)
    - titles : liste de titres (optionnel), même longueur que images
    """
    k = len(images)
    plt.figure(figsize=(4*k, 4))  # largeur proportionnelle au nombre d'images
    
    for i, img in enumerate(images):
        plt.subplot(1, k, i+1)     # 1 ligne, k colonnes
        plt.imshow(img, cmap="gray")  # cmap="gray" si images en niveaux de gris
        if titles:
            plt.title(titles[i])
        plt.axis("off")
    
    plt.show()
    
    
def get_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge et normalise les données pour MNIST, Fashion MNIST ou CIFAR-10.
    
    Les images sont normalisées entre [0, 1] et les labels sont aplatis 
    en vecteurs 1D de type int32.

    Args:
        dataset_name: Nom du dataset ('mnist', 'fashion_mnist', 'cifar10')

    Returns:
        Tuple contenant (x_train, y_train, x_test, y_test)
        - x_train, x_test: Images normalisées (float32)
        - y_train, y_test: Labels aplatis (int32)
        
    Raises:
        ValueError: Si le nom du dataset n'est pas reconnu
    """
    dataset_name = dataset_name.lower()

    # Chargement du dataset
    if dataset_name in ['mnist', 'fashion_mnist']:
        loader = (tf.keras.datasets.mnist if dataset_name == 'mnist' 
                  else tf.keras.datasets.fashion_mnist)
        (x_train, y_train), (x_test, y_test) = loader.load_data()
        # Ajout de la dimension canal pour grayscale: (28,28) → (28,28,1)
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # CIFAR-10 a déjà 3 canaux: (32,32,3)
        
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' non reconnu. "
            f"Utilisez 'mnist', 'fashion_mnist' ou 'cifar10'."
        )

    # Normalisation des images [0, 255] → [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Aplatissement des labels en vecteur 1D
    y_train = y_train.flatten().astype(np.int32)
    y_test = y_test.flatten().astype(np.int32)

    return x_train, y_train, x_test, y_test


def split_dataset_fully_synchronized(x, y, num_groups=3, seed=None):
    """
    Divise le dataset en groupes synchronisés par position, avec classes mélangées.
    
    Garantit qu'à chaque index i, tous les groupes ont le même label mais des images différentes.

    Args:
        x (ndarray): Données (ex: images)
        y (ndarray): Labels
        num_groups (int): Nombre de groupes
        seed (int, optional): Pour reproductibilité

    Returns:
        groups_x (list of ndarray): Données par groupe (images différentes)
        groups_y (list of ndarray): Labels par groupe (labels identiques à chaque ligne)
    """
    if seed is not None:
        np.random.seed(seed)

    unique_labels = np.unique(y)
    np.random.shuffle(unique_labels)

    groups_indices = [[] for _ in range(num_groups)]

    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        np.random.shuffle(label_indices)

        # Tronquer pour divisibilité parfaite
        n = len(label_indices)
        n_div = (n // num_groups) * num_groups
        label_indices = label_indices[:n_div]

        # Répartition synchronisée
        splits = np.array_split(label_indices, num_groups)
        for g in range(num_groups):
            groups_indices[g].extend(splits[g])

    # Convertir en tableaux numpy
    groups_x = [x[inds] for inds in groups_indices]
    groups_y = [y[inds] for inds in groups_indices]

    # Mélange synchronisé (optionnel mais recommandé)
    # Permet d'avoir les classes mélangées au sein de chaque groupe
    if seed is not None:
        np.random.seed(seed + 1)  # Seed différent pour ce mélange
    
    perm = np.random.permutation(len(groups_y[0]))
    groups_x = [gx[perm] for gx in groups_x]
    groups_y = [gy[perm] for gy in groups_y]

    # Vérification stricte de la synchronisation des labels
    reference = groups_y[0]
    for i in range(1, num_groups):
        if not np.array_equal(reference, groups_y[i]):
            raise ValueError(f"Les labels ne sont pas synchronisés entre les groupes {0} et {i}")

    return groups_x, groups_y

def dataset_statistics(groups_x, groups_y, show_comparison=True, verbose=True):
    """
    Affiche des statistiques détaillées sur un ou plusieurs groupes de données.

    Args:
        groups_x (list of ndarray): Liste des données par groupe
        groups_y (list of ndarray): Liste des labels par groupe
        show_comparison (bool): Si True, compare la synchronisation entre groupes
        verbose (bool): Si True, affiche les détails des distributions
    """
    num_groups = len(groups_x)
    
    print(f"{'='*60}")
    print(f"STATISTIQUES GLOBALES - {num_groups} groupe(s)")
    print(f"{'='*60}")
    
    for g in range(num_groups):
        x = groups_x[g]
        y = groups_y[g]
        num_samples = len(x)
        
        print(f"\n{'-'*60}")
        print(f"GROUPE {g+1}/{num_groups}")
        print(f"{'-'*60}")
        
        # Informations sur les données
        print(f"Nombre d'exemples    : {num_samples}")
        print(f"Dimension des images : {x.shape[1:]}")
        print(f"Plage de valeurs     : [{x.min():.3f}, {x.max():.3f}]")
        print(f"Type de données      : {x.dtype}")
        
        # Distribution des classes
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"\nNombre de classes    : {len(unique_labels)}")
        
        if verbose:
            print(f"\nDistribution des classes :")
            for label, count in zip(unique_labels, counts):
                bar = '#' * int(count / num_samples * 50)  # Barre visuelle
                print(f"  Classe {label:2d} : {count:5d} ({count/num_samples*100:5.2f}%) {bar}")
        
        # Stats équilibre
        balance_ratio = counts.min() / counts.max() if len(counts) > 0 else 1.0
        balance_status = "Equilibre" if balance_ratio > 0.8 else "Desequilibre"
        print(f"\nEquilibre            : {balance_status} (ratio: {balance_ratio:.2f})")
    
    # Comparaison entre groupes
    if num_groups > 1 and show_comparison:
        print(f"\n{'='*60}")
        print("COMPARAISON DES GROUPES")
        print(f"{'='*60}")
        
        # Vérifier la synchronisation des labels
        all_synced = True
        reference = groups_y[0]
        
        for g in range(1, num_groups):
            is_synced = np.array_equal(reference, groups_y[g])
            status = "Synchronises" if is_synced else "NON synchronises"
            print(f"Groupe 1 <-> Groupe {g+1} : {status}")
            all_synced = all_synced and is_synced
        
        if all_synced:
            print(f"\nTous les groupes sont parfaitement synchronises !")
        else:
            print(f"\nAttention : desynchronisation detectee !")
        
        # Vérifier les dimensions
        ref_shape = groups_x[0].shape
        same_shape = all(gx.shape == ref_shape for gx in groups_x)
        print(f"\nDimensions identiques  : {'Oui' if same_shape else 'Non'}")
        
        # Vérifier le nombre d'exemples
        ref_size = len(groups_x[0])
        same_size = all(len(gx) == ref_size for gx in groups_x)
        print(f"Taille identique       : {'Oui' if same_size else 'Non'}")
    
    print(f"\n{'='*60}\n")


# DataLoader
def make_tf_datasets_from_groups(groups_x: List[np.ndarray], 
                                groups_y: List[np.ndarray], 
                                batch_size: int = 100, 
                                shuffle: bool = False,
                                drop_remainder: bool = False
                            ) -> List[tf.data.Dataset]:
    """..."""
    if len(groups_x) != len(groups_y):
        raise ValueError(
            f"Nombre de groupes incompatible: {len(groups_x)} vs {len(groups_y)}"
        )
    
    datasets = []
    for idx, (x_group, y_group) in enumerate(zip(groups_x, groups_y)):
        if len(x_group) != len(y_group):
            raise ValueError(
                f"Groupe {idx}: taille incompatible"
            )
        
        # ✓ FORCER int32 ici
        y_group_int32 = y_group.astype(np.int32)
        
        ds = tf.data.Dataset.from_tensor_slices((x_group, y_group_int32))
        
        if shuffle:
            ds = ds.shuffle(buffer_size=len(x_group), reshuffle_each_iteration=True)
        
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        datasets.append(ds)
    
    return datasets

def verify_batch_synchronization(datasets):
    """Vérifie que les batches restent synchronisés entre groupes."""
    iterators = [iter(ds) for ds in datasets]
    
    batch_idx = 0
    try:
        while True:
            batches = [next(it) for it in iterators]
            labels = [batch[1].numpy() for batch in batches]
            
            # Vérifier que tous les groupes ont les mêmes labels
            reference = labels[0]
            for i, label_batch in enumerate(labels[1:], 1):
                if not np.array_equal(reference, label_batch):
                    print(f"Batch {batch_idx}: Groupe 0 et {i} non synchronisés")
                    return False
            
            batch_idx += 1
    except StopIteration:
        return True

# Détection GPU
ON_GPU = len(tf.config.list_physical_devices('GPU')) > 0


class MultiAgentAI:
    """Système multi-agents pour l'apprentissage collaboratif."""
    # ==================== INITIALISATION ====================
    
    def __init__(self, k=3, hidden_size=150, lr=1e-3, input_shape=None, num_classes=10, 
                 optimizer_type='adamw', optimizer_kwargs=None, batch_size=100):
        self.k = k
        self.batch_size = batch_size
        self.input_shape = input_shape or (28, 28)
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Création des composants
        self.sous_reseaux = self._create_all_agents()
        self.optims = self._create_optimizers(optimizer_type, lr, optimizer_kwargs)
        self.loss_fn, self.kl_fn = self._create_losses()
        self.training_history = self._init_history()
        
        print(f"✓ {'GPU' if ON_GPU else 'CPU'} - {k} agents - Optimiseur: {optimizer_type.upper()}")

    def _create_agent(self):
        """Crée un seul agent."""
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        if len(self.input_shape) == 3:
            num_filters = 16 if self.input_shape[-1] == 1 else 32
            x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu',
                                        kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = tf.keras.layers.Flatten()(x)
        else:
            x = tf.keras.layers.Flatten()(inputs)
        
        x = tf.keras.layers.Dense(80, activation='relu', 
                                   kernel_initializer=tf.keras.initializers.HeNormal())(x)
        x = tf.keras.layers.Dense(self.hidden_size, activation='relu',
                                   kernel_initializer=tf.keras.initializers.HeNormal())(x)
        x = tf.keras.layers.Dense(self.num_classes, 
                                   kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
        outputs = tf.keras.layers.Softmax()(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='agent')

    def _create_all_agents(self):
        """Crée et initialise tous les agents."""
        agents = [self._create_agent() for _ in range(self.k)]
        
        # Build explicite
        dummy = tf.zeros([1] + list(self.input_shape))
        for i, agent in enumerate(agents):
            _ = agent(dummy)
            print(f"  Agent {i}: {len(agent.trainable_variables)} variables")
        
        return agents

    def _create_optimizers(self, optimizer_type, lr, optimizer_kwargs):
        """Crée les optimiseurs pour chaque agent."""
        defaults = {
            'adamw': {'beta_1': 0.9, 'beta_2': 0.96, 'epsilon': 1e-9, 'weight_decay': 1e-5, 'amsgrad': True},
            'adam': {'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7},
            'sgd': {'momentum': 0.987, 'nesterov': True},
            'rmsprop': {'rho': 0.9, 'momentum': 0.0, 'epsilon': 1e-7},
            'adagrad': {'initial_accumulator_value': 0.1, 'epsilon': 1e-7}
        }
        
        opt_type = optimizer_type.lower()
        kwargs = {**defaults.get(opt_type, {}), **(optimizer_kwargs or {})}
        
        optimizer_classes = {
            'adamw': tf.keras.optimizers.AdamW,
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'adagrad': tf.keras.optimizers.Adagrad
        }
        
        if opt_type not in optimizer_classes:
            raise ValueError(f"Optimisateur '{optimizer_type}' non reconnu.")
        
        return [optimizer_classes[opt_type](learning_rate=lr, **kwargs) for _ in range(self.k)]

    def _create_losses(self):
        """Crée les fonctions de loss."""
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, 
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        kl_fn = tf.keras.losses.KLDivergence(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        return loss_fn, kl_fn

    def _init_history(self):
        """Initialise l'historique d'entraînement."""
        return {
            "losses": [[] for _ in range(self.k)],
            "accuracies": [[] for _ in range(self.k)],
            "ensemble_accuracy": []
        }

    # ==================== FORWARD / CALL ====================

    def __call__(self, x_batch_list, training=False):
        """Forward pass pour tous les agents."""
        return [self.sous_reseaux[i](x_batch_list[i], training=training) for i in range(self.k)]

    def call_no_grad(self, x_batch_list):
        """Forward pass sans gradient pour tous les agents."""
        return [tf.stop_gradient(self.sous_reseaux[i](x_batch_list[i], training=False)) for i in range(self.k)]

    def predict(self, x_batch_list):
        """Prédit les classes pour chaque agent."""
        outputs = self(x_batch_list, training=False)
        return [tf.argmax(o, axis=-1, output_type=tf.int32) for o in outputs]

    # ==================== CONSENSUS ====================

    def _hard_majority(self, outputs):
        """Vote majoritaire hard : one-hot sur la classe la plus votée."""
        # Prédictions de chaque agent [K, batch]
        preds = tf.stack([tf.argmax(o, axis=-1) for o in outputs], axis=0)
        
        # Comptage des votes [batch, num_classes]
        votes = tf.reduce_sum(tf.one_hot(preds, depth=self.num_classes), axis=0)
        
        # Classe majoritaire -> one-hot
        majority_class = tf.argmax(votes, axis=-1)
        return tf.one_hot(majority_class, depth=self.num_classes)

    def _soft_majority(self, outputs, temperature):
        """Vote majoritaire soft avec Gumbel-Softmax."""
        avg_probs = tf.reduce_mean(tf.stack(outputs, axis=0), axis=0)
        return self._gumbel_softmax(avg_probs, temperature=temperature, hard=True)

    def _random_vote(self, outputs, batch_size, temperature):
        """Sélection aléatoire d'un agent."""
        probs = tf.stack(outputs, axis=0)
        rand_idx = tf.random.uniform([batch_size], 0, self.k, tf.int32)
        indices = tf.stack([rand_idx, tf.range(batch_size)], axis=1)
        selected_probs = tf.gather_nd(probs, indices)
        return self._gumbel_softmax(selected_probs, temperature=temperature, hard=True)

    @staticmethod
    def _gumbel_softmax(probs, temperature=0.1, hard=True):
        """Gumbel-Softmax : approximation différentiable de argmax."""
        logits = tf.math.log(probs + 1e-8)
        uniform = tf.random.uniform(tf.shape(logits), minval=1e-8, maxval=1.0)
        gumbel_noise = -tf.math.log(-tf.math.log(uniform))
        y_soft = tf.nn.softmax((logits + gumbel_noise) / temperature)
        
        if hard:
            y_hard = tf.one_hot(tf.argmax(y_soft, axis=-1), depth=tf.shape(logits)[-1])
            y_hard = tf.cast(y_hard, y_soft.dtype)
            return y_hard - tf.stop_gradient(y_soft) + y_soft
        
        return y_soft

    def compute_consensus(self, outputs, batch_size, strategy="majority_hard", temperature=0.1):
        """
        Calcule la cible de consensus selon la stratégie.
        
        Strategies:
            - "majority_hard": Vote majoritaire dur (défaut)
            - "majority_soft": Moyenne des probas + Gumbel-Softmax
            - "vote": Sélection aléatoire d'un agent
        """
        if strategy == "majority_hard":
            consensus = self._hard_majority(outputs)
        elif strategy == "majority_soft":
            consensus = self._soft_majority(outputs, temperature)
        elif strategy == "vote":
            consensus = self._random_vote(outputs, batch_size, temperature)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'majority_hard', 'majority_soft' or 'vote'.")
        
        return tf.stop_gradient(consensus)

    # ==================== LOSSES ====================

    def soft_cross_entropy(self, soft_targets, probs):
        """Cross-entropy avec cibles soft."""
        log_probs = tf.math.log(probs + 1e-8)
        return -tf.reduce_mean(tf.reduce_sum(soft_targets * log_probs, axis=-1))

    def compute_divergence(self, output_i, other_outputs):
        """Divergence KL entre un agent et les autres."""
        kl_sum = sum(self.kl_fn(output_i, other) for other in other_outputs)
        return kl_sum / float(len(other_outputs))

    def compute_agent_loss(self, output_i, targets, consensus_target, other_outputs, alpha, beta):
        """Calcule la loss totale pour un agent."""
        L_sup = self.loss_fn(targets, output_i)
        L_cons = self.soft_cross_entropy(consensus_target, output_i) if alpha > 0 and self.k > 1 else 0.0
        L_div = self.compute_divergence(output_i, other_outputs) if beta > 0 and self.k > 1 else 0.0
        
        return L_sup * (L_sup + alpha * L_cons + beta * L_div)

    # ==================== TRAINING ====================

    def _train_single_agent(self, i, x_input, targets, consensus_target, other_outputs, alpha, beta):
        """Entraîne un seul agent."""
        with tf.GradientTape() as tape:
            output_i = self.sous_reseaux[i](x_input, training=True)
            loss = self.compute_agent_loss(output_i, targets, consensus_target, other_outputs, alpha, beta)
        
        grads = tape.gradient(loss, self.sous_reseaux[i].trainable_variables)
        
        if grads and not all(g is None for g in grads):
            self.optims[i].apply_gradients(zip(grads, self.sous_reseaux[i].trainable_variables))
        
        return loss, output_i

    def train_step_batch(self, x_batch_list, y_batch_list, strategy="majority_hard", alpha=0.5, beta=0.1, temperature=0.1):
        """Un pas d'entraînement pour tous les agents."""
        targets = tf.cast(y_batch_list[0], tf.int32)
        batch_size = tf.shape(x_batch_list[0])[0]
        
        # Forward sans gradient pour consensus
        all_outputs_no_grad = self.call_no_grad(x_batch_list)
        consensus_target = self.compute_consensus(all_outputs_no_grad, batch_size, strategy, temperature)
        
        # Entraîner chaque agent
        all_losses, all_outputs = [], []
        for i in range(self.k):
            other_outputs = [all_outputs_no_grad[j] for j in range(self.k) if j != i]
            loss, output = self._train_single_agent(
                i, x_batch_list[i], targets, consensus_target, other_outputs, alpha, beta
            )
            all_losses.append(loss)
            all_outputs.append(output)
        
        return all_losses, all_outputs

    # ==================== METRICS ====================

    def compute_accuracy(self, outputs, targets):
        """Calcule l'accuracy pour chaque agent."""
        targets = tf.cast(targets, tf.int32)
        return [
            float(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(o, axis=-1, output_type=tf.int32), targets), tf.float32)))
            for o in outputs
        ]

    def _update_history(self, epoch_losses, epoch_accs, test_acc=None):
        """Met à jour l'historique d'entraînement."""
        for i in range(self.k):
            self.training_history["losses"][i].append(np.mean(epoch_losses[i]))
            self.training_history["accuracies"][i].append(np.mean(epoch_accs[i]))
        
        if test_acc is not None:
            self.training_history["ensemble_accuracy"].append(test_acc)

    def _print_epoch(self, epoch, epochs, mean_losses, mean_accs, test_acc, strategy):
        """Affiche les résultats d'une epoch."""
        print(f"\nEpoch {epoch+1}/{epochs}")
        for i in range(self.k):
            print(f"  Agent {i+1} - Loss: {mean_losses[i]:.4f}, Acc: {mean_accs[i]:.4f}")
        if test_acc is not None:
            print(f"  Ensemble Test Accuracy ({strategy}): {test_acc:.4f}")

    # ==================== MAIN TRAINING LOOP ====================

    def train_step(self, train_datasets, epochs=10, test_datasets=None, strategy="majority_hard", 
                   verbose=True, alpha=0.5, beta=0.1, temperature=0.1):
        """Boucle d'entraînement principale."""
        for epoch in range(epochs):
            epoch_losses = [[] for _ in range(self.k)]
            epoch_accs = [[] for _ in range(self.k)]
            
            for batches in zip(*train_datasets):
                x_batch_list = [b[0] for b in batches]
                y_batch_list = [b[1] for b in batches]
                
                losses, outputs = self.train_step_batch(
                    x_batch_list, y_batch_list, strategy, alpha, beta, temperature
                )
                
                accs = self.compute_accuracy(outputs, y_batch_list[0])
                for i in range(self.k):
                    epoch_losses[i].append(float(losses[i]))
                    epoch_accs[i].append(accs[i])
            
            mean_losses = [np.mean(epoch_losses[i]) for i in range(self.k)]
            mean_accs = [np.mean(epoch_accs[i]) for i in range(self.k)]
            
            test_acc = self.evaluate_ensemble(test_datasets, strategy, verbose=False) if test_datasets else None
            self._update_history(epoch_losses, epoch_accs, test_acc)
            
            if verbose:
                self._print_epoch(epoch, epochs, mean_losses, mean_accs, test_acc, strategy)
        
        return self.training_history

    # ==================== EVALUATION ====================

    def _ensemble_predict_majority(self, preds):
        """Vote majoritaire."""
        votes = tf.one_hot(tf.stack(preds), depth=self.num_classes)
        return tf.argmax(tf.reduce_sum(votes, axis=0), axis=-1, output_type=tf.int32)

    def _ensemble_predict_vote(self, preds, batch_size):
        """Sélection aléatoire."""
        rand_idx = tf.random.uniform([batch_size], 0, self.k, tf.int32)
        preds_stack = tf.stack(preds, axis=1)
        return tf.gather_nd(preds_stack, tf.stack([tf.range(batch_size), rand_idx], axis=1))

    def ensemble_predict(self, preds, strategy, batch_size):
        """Prédiction d'ensemble selon la stratégie."""
        if strategy in ["majority_hard", "majority_soft"]:
            return self._ensemble_predict_majority(preds)
        elif strategy == "vote":
            return self._ensemble_predict_vote(preds, batch_size)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")

    def evaluate_ensemble(self, test_datasets, strategy="majority_hard", verbose=True):
        """Évaluation de l'ensemble."""
        all_preds_ensemble, all_trues = [], []
        all_preds_agents = [[] for _ in range(self.k)]
        
        for batches in zip(*test_datasets):
            x_batches = [b[0] for b in batches]
            y_batch = tf.cast(batches[0][1], tf.int32)
            batch_size = tf.shape(y_batch)[0]
            
            preds = self.predict(x_batches)
            ensemble_pred = self.ensemble_predict(preds, strategy, batch_size)
            
            for i in range(self.k):
                all_preds_agents[i].append(preds[i])
            
            all_preds_ensemble.append(ensemble_pred)
            all_trues.append(y_batch)
        
        # Calcul accuracy
        all_trues = tf.concat(all_trues, axis=0)
        all_preds_ensemble = tf.concat(all_preds_ensemble, axis=0)
        acc_ensemble = float(tf.reduce_mean(tf.cast(tf.equal(all_preds_ensemble, all_trues), tf.float32)))
        
        if verbose:
            self._print_evaluation(all_preds_agents, all_trues, acc_ensemble, strategy)
        
        return acc_ensemble

    def _print_evaluation(self, all_preds_agents, all_trues, acc_ensemble, strategy):
        """Affiche les résultats d'évaluation."""
        print("\nEvaluation des agents :")
        for i in range(self.k):
            preds_i = tf.concat(all_preds_agents[i], axis=0)
            acc_i = float(tf.reduce_mean(tf.cast(tf.equal(preds_i, all_trues), tf.float32)))
            print(f"  Agent {i+1} - Acc: {acc_i:.4f}")
        print(f"  Ensemble ({strategy}) - Acc: {acc_ensemble:.4f}")

    # ==================== ENTROPY ====================

    def _compute_single_entropy(self, pred):
        """Calcule l'entropie d'une seule prédiction."""
        _, _, counts = tf.unique_with_counts(pred)
        probs = tf.cast(counts, tf.float32) / tf.cast(tf.reduce_sum(counts), tf.float32)
        return -tf.reduce_sum(probs * tf.math.log(probs + 1e-8)) / tf.math.log(tf.cast(self.k, tf.float32))

    def entropy_ensemble(self, test_datasets):
        """Calcule l'entropie des prédictions de l'ensemble."""
        all_preds = []
        
        for batches in zip(*test_datasets):
            x_batches = [b[0] for b in batches]
            preds = self.predict(x_batches)
            all_preds.append(tf.stack(preds, axis=1))
        
        all_preds = tf.concat(all_preds, axis=0)
        return tf.stack([self._compute_single_entropy(all_preds[i]) for i in range(tf.shape(all_preds)[0])])


# ==================== TEST ====================

if __name__ == "__main__":
    pass  # Tests unitaires peuvent être ajoutés ici