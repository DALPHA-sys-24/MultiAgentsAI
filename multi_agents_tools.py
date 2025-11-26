import tensorflow as tf
import numpy as np
from typing import List,Tuple

# Global Configuration
gpu_devices = tf.config.list_physical_devices('GPU')
on_gpu = len(gpu_devices) > 0
tf.keras.mixed_precision.set_global_policy('float32')


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


def create_agent(hidden_size: int = 50, num_classes: int = 10, input_shape: tuple = None):
    """
    Crée un modèle pour apprentissage personnalisé avec GradientTape.
    
    Architecture:
    - Si input_shape fourni: Input → Conv2D → MaxPooling → Flatten → Dense → Dense → Softmax
    - Si input_shape = None: Flatten → Dense → Dense → Softmax
    
    Args:
        hidden_size: Nombre de neurones dans la couche cachée
        num_classes: Nombre de classes de sortie (10 pour MNIST/CIFAR-10)
        input_shape: Forme d'entrée optionnelle (ex: (28, 28, 1) pour MNIST)
                    Si None, sera inférée lors du premier appel (pas de conv)
    
    Returns:
        tf.keras.Model non compilé, qui retourne des probabilités
    
    Note:
        Le modèle retourne directement des probabilités (softmax appliqué).
    """
    layers = []
    
    # Ajouter Input layer et couches convolutives si shape spécifiée
    if input_shape is not None:
        layers.append(tf.keras.layers.Input(shape=input_shape))
        
        # Déterminer le nombre de filtres selon le nombre de canaux d'entrée
        num_channels = input_shape[-1] if len(input_shape) == 3 else 1
        num_filters = 16 if num_channels == 1 else 32  # Plus de filtres pour RGB
        
        layers.extend([
            tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu',
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer='zeros',
                name='conv1'
            ),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid',
                name='maxpool1'
            )
        ])
    
    layers.extend([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            hidden_size, 
            activation='relu',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            bias_initializer='zeros',
            name='hidden'
        ),
        tf.keras.layers.Dense(
            num_classes,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer='zeros',
            name='logits'
        ),
        tf.keras.layers.Softmax(name='probabilities')
    ])
    
    model = tf.keras.Sequential(layers, name='agent')
    return model


class MultiAgentAI:
    def __init__(self, k=3, hidden_size=150, lr=1e-3,input_shape=None, num_classes=10, 
                 optimizer_type='adamw', optimizer_kwargs=None,batch_size=100):
        """
        Args:
            k: Nombre de sous-réseaux
            hidden_size: Taille de la couche cachée
            lr: Learning rate
            num_classes: Nombre de classes
            optimizer_type: Type d'optimisateur ('adamw', 'sgd', 'adam', 'rmsprop', 'adagrad')
            optimizer_kwargs: Dict avec paramètres supplémentaires pour l'optimisateur
        """
        self.k = k
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.sous_reseaux = [create_agent(hidden_size, num_classes,input_shape) for _ in range(k)]
        
        # Paramètres par défaut selon le type d'optimisateur
        default_kwargs = {
            'adamw': {'beta_1': 0.9, 'beta_2': 0.96, 'epsilon': 1e-9, 
                      'weight_decay': 1e-5, 'amsgrad': True},
            'adam': {'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7},
            'sgd': {'momentum': 0.987, 'nesterov': True},
            'rmsprop': {'rho': 0.9, 'momentum': 0.0, 'epsilon': 1e-7},
            'adagrad': {'initial_accumulator_value': 0.1, 'epsilon': 1e-7}
        }
        
        # Fusion des kwargs par défaut avec ceux fournis
        opt_kwargs = default_kwargs.get(optimizer_type.lower(), {})
        if optimizer_kwargs:
            opt_kwargs.update(optimizer_kwargs)
        
        # Création de l'optimisateur de base
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'adamw':
            base_optim = lambda: tf.keras.optimizers.AdamW(learning_rate=lr, **opt_kwargs)
        elif optimizer_type == 'adam':
            base_optim = lambda: tf.keras.optimizers.Adam(learning_rate=lr, **opt_kwargs)
        elif optimizer_type == 'sgd':
            base_optim = lambda: tf.keras.optimizers.SGD(learning_rate=lr, **opt_kwargs)
        elif optimizer_type == 'rmsprop':
            base_optim = lambda: tf.keras.optimizers.RMSprop(learning_rate=lr, **opt_kwargs)
        elif optimizer_type == 'adagrad':
            base_optim = lambda: tf.keras.optimizers.Adagrad(learning_rate=lr, **opt_kwargs)
        else:
            raise ValueError(
                f"Optimisateur '{optimizer_type}' non reconnu. "
                f"Choisissez parmi: 'adamw', 'adam', 'sgd', 'rmsprop', 'adagrad'"
            )
        
        # Application du LossScaleOptimizer si GPU
        if on_gpu:
            print("Pas de precision mixte - Pas de LossScaleOptimizer.")
            self.optims = [base_optim() for _ in range(k)]
            print(f"✓ GPU détecté - Optimisateur: {optimizer_type.upper()}")
        else:
            self.optims = [base_optim() for _ in range(k)]
            print(f"✓ CPU détecté - Optimisateur: {optimizer_type.upper()}")
            
            
        
        # Loss avec from_logits=False car le modèle retourne des probas
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,reduction="sum_over_batch_size")
        self.kl_fn = tf.keras.losses.KLDivergence(reduction="sum_over_batch_size")
        self.training_history = {
            "losses": [[] for _ in range(k)],
            "accuracies": [[] for _ in range(k)],
            "ensemble_accuracy": []
        }

    @tf.function
    def cross_information(self, outputs, targets, strategy="majority", alpha=0.5, beta=0.0):

        targets = tf.cast(targets, tf.int32)
        batch_size = tf.shape(outputs[0])[0]
        self.batch_size = batch_size
        
        if self.k == 1:
            return tf.stack([self.loss_fn(targets, outputs[0])])

        # 1. prédictions
        preds = tf.stack([tf.argmax(o, axis=-1, output_type=tf.int32) for o in outputs], axis=0)

        if strategy == "majority":
            onehot = tf.one_hot(preds, depth=self.num_classes)
            votes = tf.reduce_sum(onehot, axis=0)
            maj_targets = tf.argmax(votes, axis=-1, output_type=tf.int32)
        elif strategy == "vote":
            rand_idx = tf.random.uniform([self.batch_size ], 0, self.k, tf.int32)
            preds_t = tf.transpose(preds, [1, 0])
            maj_targets = tf.gather_nd(preds_t, tf.stack([tf.range(self.batch_size ), rand_idx], axis=1))
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

        # 2. losses locales
        L_i = tf.stack([self.loss_fn(targets, o) for o in outputs])

        # 3. consensus
        L_loc = tf.stack([self.loss_fn(maj_targets, o) for o in outputs])

        # 4. divergence (seulement si k>1)
        # KL déjà scalaire par reduction="sum_over_batch_size"
        loss_div_list = []
        for i in range(self.k):
            div_i = tf.reduce_mean([self.kl_fn(outputs[i], outputs[j]) for j in range(self.k) if j != i])
            div_i = tf.math.multiply_no_nan(div_i, 1.0)
            loss_div_list.append(div_i)
        loss_div = tf.stack(loss_div_list)

        # 5. final
        losses = L_i * (L_i + alpha * L_loc + beta * loss_div)
        return losses

    @tf.function
    def train_step_batch(self, x_batch_list, y_batch_list, strategy="majority", alpha=0.5, beta=0.1):
        targets = tf.cast(y_batch_list[0], tf.int32)
        gradients_list = []
    
        # --- Calcul des gradients pour chaque sous-réseau ---
        for i in range(self.k):
            with tf.GradientTape() as tape:
                # Forward partagé : un seul calcul pour tous les agents
                outputs = [
                    self.sous_reseaux[j](x_batch_list[j], training=True)
                    for j in range(self.k)
                ]
                # Loss vectorielle par agent
                losses = self.cross_information(
                    outputs, targets, strategy=strategy,
                    alpha=alpha, beta=beta)
    
                # Loss du modèle i (scalaire)
                loss_i = losses[i]
                grads = tape.gradient(loss_i, self.sous_reseaux[i].trainable_variables)
            gradients_list.append(grads)
    
        # --- Application des gradients pour chaque agent ---
        for i in range(self.k):
            self.optims[i].apply_gradients(
                zip(gradients_list[i], self.sous_reseaux[i].trainable_variables))
    
        # --- Monitoring : forward en mode inference ---
        final_outputs = [
            self.sous_reseaux[i](x_batch_list[i], training=False)
            for i in range(self.k)]
    
        per_model_losses = self.cross_information(
            final_outputs, targets, strategy=strategy,
            alpha=alpha, beta=beta)
    
        return per_model_losses, final_outputs


    def train_step(self, train_datasets, epochs=10, test_datasets=None, strategy="majority", verbose=True,alpha=0.5, beta=0.1):
        for epoch in range(epochs):
            epoch_losses, epoch_accs = [[] for _ in range(self.k)], [[] for _ in range(self.k)]
            for batches in zip(*train_datasets):
                x_batch_list = [b[0] for b in batches]
                y_batch_list = [b[1] for b in batches]
                losses, outputs = self.train_step_batch(x_batch_list, y_batch_list, strategy=strategy,alpha=alpha, beta=beta)

                targets = tf.cast(y_batch_list[0], tf.int32)
                for i in range(self.k):
                    preds = tf.argmax(outputs[i], axis=-1, output_type=tf.int32)
                    acc = tf.reduce_mean(tf.cast(tf.equal(targets, preds), tf.float32))
                    epoch_losses[i].append(float(losses[i]))
                    epoch_accs[i].append(float(acc))

            mean_losses = [np.mean(epoch_losses[i]) for i in range(self.k)]
            mean_accs = [np.mean(epoch_accs[i]) for i in range(self.k)]
            for i in range(self.k):
                self.training_history["losses"][i].append(mean_losses[i])
                self.training_history["accuracies"][i].append(mean_accs[i])

            test_acc = None
            if test_datasets is not None:
                test_acc = self.evaluate_ensemble(test_datasets, strategy=strategy, verbose=False)
                self.training_history["ensemble_accuracy"].append(test_acc)

            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs}")
                for i in range(self.k):
                    print(f"  Agent {i+1} - Loss: {mean_losses[i]:.4f}, Acc: {mean_accs[i]:.4f}")
                if test_acc is not None:
                    print(f"  Ensemble Test Accuracy: {test_acc:.4f}")

        return self.training_history


    def evaluate_ensemble(self, test_datasets, strategy="majority", verbose=True):

        all_preds_ensemble = []
        all_trues = []
        all_preds_agents = [[] for _ in range(self.k)]

        for batches in zip(*test_datasets):
            x_batches = [b[0] for b in batches]
            y_batch = tf.cast(batches[0][1], tf.int32)

            # Forward pour tous les agents
            outputs = [self.sous_reseaux[i](x_batches[i], training=False) for i in range(self.k)]
            preds = [tf.argmax(o, axis=-1, output_type=tf.int32) for o in outputs]

            # Stocke les prédictions par agent
            for i in range(self.k):
                all_preds_agents[i].append(preds[i])

            # Vote pour l'ensemble
            if strategy == "majority":
                votes = tf.one_hot(tf.stack(preds), depth=self.num_classes)
                votes_sum = tf.reduce_sum(votes, axis=0)
                ensemble_pred = tf.argmax(votes_sum, axis=-1, output_type=tf.int32)
            elif strategy == "vote":
                batch_size = tf.shape(preds[0])[0]
                rand_idx = tf.random.uniform([batch_size], 0, self.k, tf.int32)
                preds_stack = tf.stack(preds, axis=1)
                ensemble_pred = tf.gather_nd(preds_stack, tf.stack([tf.range(batch_size), rand_idx], axis=1))

            all_preds_ensemble.append(ensemble_pred)
            all_trues.append(y_batch)

        # Concaténation
        all_trues = tf.concat(all_trues, axis=0)
        all_preds_ensemble = tf.concat(all_preds_ensemble, axis=0)
        all_preds_agents = [tf.concat(all_preds_agents[i], axis=0) for i in range(self.k)]

        # Accuracies
        acc_ensemble = tf.reduce_mean(tf.cast(tf.equal(all_preds_ensemble, all_trues), tf.float32))
        if verbose:
            print("\nEvaluation des agents :")
            for i in range(self.k):
                acc_i = tf.reduce_mean(tf.cast(tf.equal(all_preds_agents[i], all_trues), tf.float32))
                print(f"  Agent {i+1} - Acc: {float(acc_i):.4f}")
            print(f"  Ensemble ({strategy}) - Acc: {float(acc_ensemble):.4f}")

        return float(acc_ensemble)

    
    def entropy_ensemble(self, test_datasets):
        all_preds = []

        for batches in zip(*test_datasets):
            x_batches = [b[0] for b in batches]
            outputs = [self.sous_reseaux[i](x_batches[i], training=False) for i in range(self.k)]
            preds = [tf.argmax(o, axis=-1, output_type=tf.int32) for o in outputs]
            preds = tf.stack(preds, axis=1)  # shape (batch, k)
            all_preds.append(preds)

        all_preds = tf.concat(all_preds, axis=0)  # shape (N, k)
        N = tf.shape(all_preds)[0]

        entropies = []
        for i in range(N):
            pred_i = all_preds[i]  # shape (k,)
            values, _, counts = tf.unique_with_counts(pred_i)
            probs = tf.cast(counts, tf.float32) / tf.cast(tf.reduce_sum(counts), tf.float32)
            ent = -tf.reduce_sum(probs * tf.math.log(probs)) / tf.math.log(tf.cast(self.k, tf.float32))
            entropies.append(ent)

        entropies = tf.stack(entropies)  # shape (N,)
        return entropies







    
if __name__ == '__main__':
    pass