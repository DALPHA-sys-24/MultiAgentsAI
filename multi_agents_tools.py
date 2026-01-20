import tensorflow as tf
import numpy as np
from typing import List, Tuple, Optional, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------
gpu_devices = tf.config.list_physical_devices('GPU')
on_gpu = len(gpu_devices) > 0
tf.keras.mixed_precision.set_global_policy('float32')




# ---------------------------------------------------------------------
# Utilities: affichage
# ---------------------------------------------------------------------
def show_images(images, titles=None):
    """
    Affiche une liste d'images sur une même figure.

    Parameters:
    - images : liste de tableaux numpy ou d'images (ex: PIL.Image)
    - titles : liste de titres (optionnel), même longueur que images
    """
    k = len(images)
    plt.figure(figsize=(4 * k, 4))  # largeur proportionnelle au nombre d'images

    for i, img in enumerate(images):
        plt.subplot(1, k, i + 1)  # 1 ligne, k colonnes
        plt.imshow(img, cmap="gray")
        if titles:
            plt.title(titles[i])
        plt.axis("off")

    plt.show()

def inverse_transform(
    x: np.ndarray,
    input_shape: Tuple[int, ...],
    c: int = 10
) -> np.ndarray:
    """
    Inverse de la transformation appliquée dans get_data:
    - Retire les c derniers éléments (vecteur one-hot)
    - Reshape en input_shape original

    Args:
        x: Données aplaties avec vecteur one-hot (shape: (n_samples, flattened_size + c))
        input_shape: Shape original des images (ex: (28, 28) pour MNIST, (32, 32, 3) pour CIFAR)
        c: Nombre d'éléments à retirer à la fin (défaut: 10 pour le one-hot)

    Returns:
        Images reshapées en input_shape (shape: (n_samples, *input_shape))
    """
    # Retirer les c derniers éléments
    x_without_onehot = x[:, :-c]
    
    # Reshape en input_shape
    x_reshaped = x_without_onehot.reshape(-1, *input_shape)
    
    return x_reshaped


def inverse_transform_single(
    x: np.ndarray,
    input_shape: Tuple[int, ...],
    c: int = 10
) -> np.ndarray:
    """
    Version pour une seule image.

    Args:
        x: Image aplatie avec vecteur one-hot (shape: (flattened_size + c,))
        input_shape: Shape original de l'image
        c: Nombre d'éléments à retirer

    Returns:
        Image reshapée en input_shape
    """
    # Retirer les c derniers éléments
    x_without_onehot = x[:-c]
    
    # Reshape en input_shape
    x_reshaped = x_without_onehot.reshape(input_shape)
    
    return x_reshaped
# ---------------------------------------------------------------------
# Data loading / preprocessing
# ---------------------------------------------------------------------
# def get_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Charge et normalise les données pour MNIST, Fashion MNIST ou CIFAR-10.

#     Returns:
#         (x_train, y_train, x_test, y_test)
#     """
#     dataset_name = dataset_name.lower()

#     if dataset_name in ['mnist', 'fashion_mnist']:
#         loader = tf.keras.datasets.mnist if dataset_name == 'mnist' else tf.keras.datasets.fashion_mnist
#         (x_train, y_train), (x_test, y_test) = loader.load_data()
#         # Ajouter la dimension canal pour grayscale
#         x_train = x_train[..., np.newaxis]
#         x_test = x_test[..., np.newaxis]

#     elif dataset_name == 'cifar10':
#         (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#         # CIFAR-10 a déjà (32,32,3)

#     else:
#         raise ValueError(f"Dataset '{dataset_name}' non reconnu. Utilisez 'mnist', 'fashion_mnist' ou 'cifar10'.")

#     # Normalisation [0,255] -> [0,1]
#     x_train = x_train.astype(np.float32) / 255.0
#     x_test = x_test.astype(np.float32) / 255.0

#     # Aplatir les labels
#     y_train = y_train.flatten().astype(np.int32)
#     y_test = y_test.flatten().astype(np.int32)

#     return x_train, y_train, x_test, y_test

def get_data(
    dataset_name: str, 
    j: Optional[int] = None, 
    num_classes: int = 10,
    filter_class: Optional[Union[int, List[int]]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge, normalise et aplati les données pour MNIST, Fashion MNIST ou CIFAR-10.
    Peut filtrer par classe et ajouter un vecteur one-hot.

    Args:
        dataset_name: Nom du dataset ('mnist', 'fashion_mnist', 'cifar10')
        j: Position du 1 dans le vecteur one-hot (0 à num_classes-1). 
           Si None, aucun vecteur n'est ajouté.
        num_classes: Nombre de classes (taille du vecteur one-hot)
        filter_class: Classe(s) à extraire. Peut être:
           - Un entier (ex: 3) pour extraire une seule classe
           - Une liste (ex: [0, 1, 5]) pour extraire plusieurs classes
           - None pour garder toutes les classes

    Returns:
        (x_train, y_train, x_test, y_test)
    """
    dataset_name = dataset_name.lower()

    if dataset_name in ['mnist', 'fashion_mnist']:
        loader = tf.keras.datasets.mnist if dataset_name == 'mnist' else tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = loader.load_data()

    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    else:
        raise ValueError(f"Dataset '{dataset_name}' non reconnu. Utilisez 'mnist', 'fashion_mnist' ou 'cifar10'.")

    # Aplatir les labels
    y_train = y_train.flatten().astype(np.int32)
    y_test = y_test.flatten().astype(np.int32)

    # Filtrer par classe si spécifié
    if filter_class is not None:
        # Convertir en liste si c'est un entier
        if isinstance(filter_class, int):
            filter_class = [filter_class]
        
        # Créer les masques pour train et test
        train_mask = np.isin(y_train, filter_class)
        test_mask = np.isin(y_test, filter_class)
        
        # Appliquer les masques
        x_train = x_train[train_mask]
        y_train = y_train[train_mask]
        x_test = x_test[test_mask]
        y_test = y_test[test_mask]

    # Normalisation [0,255] -> [0,1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Aplatir les images en vecteurs
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Ajouter le vecteur one-hot si j est spécifié
    if j is not None:
        if not (0 <= j < num_classes):
            raise ValueError(f"j doit être entre 0 et {num_classes - 1}, reçu: {j}")
        
        # Créer le vecteur one-hot
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[j] = 1.0
        
        # Répliquer pour chaque échantillon et concaténer
        one_hot_train = np.tile(one_hot, (x_train.shape[0], 1))
        one_hot_test = np.tile(one_hot, (x_test.shape[0], 1))
        
        x_train = np.concatenate([x_train, one_hot_train], axis=1)
        x_test = np.concatenate([x_test, one_hot_test], axis=1)

    return x_train, y_train, x_test, y_test

# ---------------------------------------------------------------------
# Split dataset : fully synchronized groups
# ---------------------------------------------------------------------
def split_dataset_fully_synchronized(x: np.ndarray, y: np.ndarray, num_groups: int = 3, seed: int = None):
    """
    Divise le dataset en groupes synchronisés par position.
    Garantit qu'à chaque index i, tous les groupes ont le même label mais des images différentes.
    """
    if seed is not None:
        np.random.seed(seed)

    unique_labels = np.unique(y)
    np.random.shuffle(unique_labels)

    groups_indices = [[] for _ in range(num_groups)]

    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        np.random.shuffle(label_indices)

        n = len(label_indices)
        n_div = (n // num_groups) * num_groups
        label_indices = label_indices[:n_div]

        splits = np.array_split(label_indices, num_groups)
        for g in range(num_groups):
            groups_indices[g].extend(splits[g])

    groups_x = [x[inds] for inds in groups_indices]
    groups_y = [y[inds] for inds in groups_indices]

    # Mélange synchronisé (permute les positions, conserve la synchronisation des labels)
    perm = np.random.permutation(len(groups_y[0]))
    groups_x = [gx[perm] for gx in groups_x]
    groups_y = [gy[perm] for gy in groups_y]

    # Vérification stricte
    reference = groups_y[0]
    for i in range(1, num_groups):
        if not np.array_equal(reference, groups_y[i]):
            raise ValueError(f"Les labels ne sont pas synchronisés entre les groupes 0 et {i}")

    return groups_x, groups_y


# ---------------------------------------------------------------------
# Statistiques dataset
# ---------------------------------------------------------------------
def dataset_statistics(groups_x, groups_y, show_comparison: bool = True, verbose: bool = True):
    """
    Affiche des statistiques détaillées sur un ou plusieurs groupes de données.
    """
    num_groups = len(groups_x)

    print(f"{'=' * 60}")
    print(f"STATISTIQUES GLOBALES - {num_groups} groupe(s)")
    print(f"{'=' * 60}")

    for g in range(num_groups):
        x = groups_x[g]
        y = groups_y[g]
        num_samples = len(x)

        print(f"\n{'-' * 60}")
        print(f"GROUPE {g + 1}/{num_groups}")
        print(f"{'-' * 60}")

        print(f"Nombre d'exemples    : {num_samples}")
        print(f"Dimension des images : {x.shape[1:]}")
        print(f"Plage de valeurs     : [{x.min():.3f}, {x.max():.3f}]")
        print(f"Type de données      : {x.dtype}")

        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"\nNombre de classes    : {len(unique_labels)}")

        if verbose:
            print(f"\nDistribution des classes :")
            for label, count in zip(unique_labels, counts):
                bar = '#' * int(count / num_samples * 50) if num_samples > 0 else ''
                print(f"  Classe {label:2d} : {count:5d} ({count / num_samples * 100:5.2f}%) {bar}")

        balance_ratio = counts.min() / counts.max() if len(counts) > 0 else 1.0
        balance_status = "Equilibre" if balance_ratio > 0.8 else "Desequilibre"
        print(f"\nEquilibre            : {balance_status} (ratio: {balance_ratio:.2f})")

    if num_groups > 1 and show_comparison:
        print(f"\n{'=' * 60}")
        print("COMPARAISON DES GROUPES")
        print(f"{'=' * 60}")

        all_synced = True
        reference = groups_y[0]
        for g in range(1, num_groups):
            is_synced = np.array_equal(reference, groups_y[g])
            status = "Synchronises" if is_synced else "NON synchronises"
            print(f"Groupe 1 <-> Groupe {g + 1} : {status}")
            all_synced = all_synced and is_synced

        if all_synced:
            print(f"\nTous les groupes sont parfaitement synchronises !")
        else:
            print(f"\nAttention : desynchronisation detectee !")

        ref_shape = groups_x[0].shape
        same_shape = all(gx.shape == ref_shape for gx in groups_x)
        print(f"\nDimensions identiques  : {'Oui' if same_shape else 'Non'}")

        ref_size = len(groups_x[0])
        same_size = all(len(gx) == ref_size for gx in groups_x)
        print(f"Taille identique       : {'Oui' if same_size else 'Non'}")

    print(f"\n{'=' * 60}\n")


# ---------------------------------------------------------------------
# DataLoader -> tf.data.Dataset
# ---------------------------------------------------------------------
def make_tf_datasets_from_groups(groups_x: List[np.ndarray],
                                groups_y: List[np.ndarray],
                                batch_size: int = 100,
                                shuffle: bool = False,
                                drop_remainder: bool = False) -> List[tf.data.Dataset]:
    """
    Construit une liste de tf.data.Dataset (un par groupe).
    """
    if len(groups_x) != len(groups_y):
        raise ValueError(f"Nombre de groupes incompatible: {len(groups_x)} vs {len(groups_y)}")

    datasets = []
    for idx, (x_group, y_group) in enumerate(zip(groups_x, groups_y)):
        if len(x_group) != len(y_group):
            raise ValueError(f"Groupe {idx}: taille incompatible")

        y_group_int32 = y_group.astype(np.int32)

        ds = tf.data.Dataset.from_tensor_slices((x_group, y_group_int32))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(x_group), reshuffle_each_iteration=True)

        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        datasets.append(ds)

    return datasets


def verify_batch_synchronization(datasets):
    """
    Vérifie que les batches restent synchronisés entre groupes.
    """
    iterators = [iter(ds) for ds in datasets]
    batch_idx = 0
    try:
        while True:
            batches = [next(it) for it in iterators]
            labels = [batch[1].numpy() for batch in batches]

            reference = labels[0]
            for i, label_batch in enumerate(labels[1:], 1):
                if not np.array_equal(reference, label_batch):
                    print(f"Batch {batch_idx}: Groupe 0 et {i} non synchronisés")
                    return False

            batch_idx += 1
    except StopIteration:
        return True


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------
def create_agent(hidden_size: int = 50, num_classes: int = 10, input_shape: tuple = None):
    """
    Crée un modèle Keras séquentiel qui renvoie des probabilités (Softmax).
    """
    layers = []

    if input_shape is not None:
        layers.append(tf.keras.layers.Input(shape=input_shape))

        num_channels = input_shape[-1] if len(input_shape) == 3 else 1
        num_filters = 16 if num_channels == 1 else 32

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
        tf.keras.layers.Dense(
            hidden_size,
            activation='relu',
            use_bias=False,
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


# ---------------------------------------------------------------------
# MultiAgentAI class (conservée, mise en forme)
# ---------------------------------------------------------------------
class MultiAgentAI:
    def __init__(self, k=3, hidden_size=150, lr=1e-3, input_shape=None, num_classes=10,
                 optimizer_type='adamw', optimizer_kwargs=None, batch_size=100):
        """
        Multi-agent wrapper: k sous-réseaux indépendants + optimisateurs.
        """
        self.k = k
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Création des sous-réseaux
        self.sous_reseaux = [create_agent(hidden_size, num_classes, input_shape) for _ in range(k)]

        # Paramètres par défaut des optimiseurs
        default_kwargs = {
            'adamw': {'beta_1': 0.9, 'beta_2': 0.96, 'epsilon': 1e-9, 'weight_decay': 1e-5, 'amsgrad': True},
            'adam': {'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7},
            'sgd': {'momentum': 0.987, 'nesterov': True},
            'rmsprop': {'rho': 0.9, 'momentum': 0.0, 'epsilon': 1e-7},
            'adagrad': {'initial_accumulator_value': 0.1, 'epsilon': 1e-7}
        }

        opt_kwargs = default_kwargs.get(optimizer_type.lower(), {})
        if optimizer_kwargs:
            opt_kwargs.update(optimizer_kwargs)

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
            raise ValueError(f"Optimisateur '{optimizer_type}' non reconnu.")

        # Création des optimiseurs
        self.optims = [base_optim() for _ in range(k)]
        if on_gpu:
            print(f"✓ GPU détecté - Optimisateur: {optimizer_type.upper()}")
        else:
            print(f"✓ CPU détecté - Optimisateur: {optimizer_type.upper()}")

        # Fonctions de perte
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                     reduction="sum_over_batch_size")
        self.kl_fn = tf.keras.losses.KLDivergence(reduction="sum_over_batch_size")

        # Historique
        self.training_history = {
            "losses": [[] for _ in range(k)],
            "accuracies": [[] for _ in range(k)],
            "ensemble_accuracy": []
        }

        # Option de tracking de lg (utilisé précédemment dans ton code)
        self.track_lg = False

    # -----------------------------------------------------------------
    @tf.function
    def cross_information(self, outputs, targets, strategy="majority", alpha=1.0, beta=0.0):
        """
        Calcule les pertes locales + termes de consensus/divergence.
        Retourne un vecteur de pertes (taille k).
        """
        targets = tf.cast(targets, tf.int32)
        batch_size = tf.shape(outputs[0])[0]
        self.batch_size = batch_size

        if self.k == 1:
            return tf.stack([self.loss_fn(targets, outputs[0])])

        # 1) prédictions catégoriques par agent (indices)
        preds = tf.stack([tf.argmax(o, axis=-1, output_type=tf.int32) for o in outputs], axis=0)

        # 2) calcul maj_targets selon strategy
        if strategy == "majority":
            onehot = tf.one_hot(preds, depth=self.num_classes)
            votes = tf.reduce_sum(onehot, axis=0)
            maj_targets = tf.argmax(votes, axis=-1, output_type=tf.int32)
        elif strategy == "vote":
            rand_idx = tf.random.uniform([batch_size], 0, self.k, tf.int32)
            preds_t = tf.transpose(preds, [1, 0])
            maj_targets = tf.gather_nd(preds_t, tf.stack([tf.range(batch_size), rand_idx], axis=1))
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

        # 3) losses locales L_i
        L_i = tf.stack([self.loss_fn(targets, o) for o in outputs])

        # 4) consensus L_loc : cross-entropy avec maj_targets
        L_loc = tf.stack([self.loss_fn(maj_targets, o) for o in outputs])

        # 5) divergence (KL moyen entre distributions)
        loss_div_list = []
        for i in range(self.k):
            div_i = tf.reduce_mean([self.kl_fn(outputs[i], outputs[j]) for j in range(self.k) if j != i])
            div_i = tf.math.multiply_no_nan(div_i, 1.0)
            loss_div_list.append(div_i)
        loss_div = tf.stack(loss_div_list)

        # 6) composition finale
        losses = L_i * (L_i + alpha * tf.minimum(L_loc, L_i) + beta * loss_div)
        return losses

    # -----------------------------------------------------------------
    @tf.function
    def train_step_batch(self, x_batch_list, y_batch_list, strategy="majority", alpha=0.5, beta=0.1):
        """
        Effectue une étape d'entraînement sur un batch synchronisé (un batch par agent).
        """
        targets = tf.cast(y_batch_list[0], tf.int32)
        gradients_list = []

        # Calcul des gradients pour chaque agent
        for i in range(self.k):
            with tf.GradientTape() as tape:
                outputs = [self.sous_reseaux[j](x_batch_list[j], training=True) for j in range(self.k)]
                losses = self.cross_information(outputs, targets, strategy=strategy, alpha=alpha, beta=beta)

                loss_i = losses[i]
                grads = tape.gradient(loss_i, self.sous_reseaux[i].trainable_variables)

            gradients_list.append(grads)

        # Application des gradients
        for i in range(self.k):
            self.optims[i].apply_gradients(zip(gradients_list[i], self.sous_reseaux[i].trainable_variables))

        # Forward final en mode inference pour monitoring
        final_outputs = [self.sous_reseaux[i](x_batch_list[i], training=False) for i in range(self.k)]

        per_model_losses = self.cross_information(final_outputs, targets, strategy=strategy, alpha=alpha, beta=beta)
        return per_model_losses, final_outputs

    # -----------------------------------------------------------------
    def _compute_warmup_alpha(self, epoch, alpha, alpha_warmup):
        """Calcule alpha avec warm-up progressif."""
        if alpha_warmup <= 0:
            return alpha
        return alpha * min(1.0, (epoch + 1) / alpha_warmup)

    # -----------------------------------------------------------------
    def train_step(self, train_datasets, epochs=10, test_datasets=None, strategy="majority",
                   verbose=True, alpha=0.5, beta=0.1, alpha_warmup=0):
        """
        Boucle d'entraînement principale.
        """
        for epoch in range(epochs):
            current_alpha = self._compute_warmup_alpha(epoch, alpha, alpha_warmup)

            epoch_losses = [[] for _ in range(self.k)]
            epoch_accs = [[] for _ in range(self.k)]

            for batches in zip(*train_datasets):
                x_batch_list = [b[0] for b in batches]
                y_batch_list = [b[1] for b in batches]

                losses, outputs = self.train_step_batch(x_batch_list, y_batch_list,
                                                       strategy=strategy, alpha=current_alpha, beta=beta)

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
                print(f"\nEpoch {epoch + 1}/{epochs} (alpha={current_alpha:.3f})")
                for i in range(self.k):
                    print(f"  Agent {i + 1} - Loss: {mean_losses[i]:.4f}, Acc: {mean_accs[i]:.4f}")
                if test_acc is not None:
                    print(f"  Ensemble Test Accuracy: {test_acc:.4f}")

        return self.training_history

    # -----------------------------------------------------------------
    def evaluate_ensemble(self, test_datasets, strategy="majority", verbose=True):
        """
        Évalue le vote d'ensemble sur les jeux de test.
        """
        all_preds_ensemble = []
        all_trues = []
        all_preds_agents = [[] for _ in range(self.k)]

        for batches in zip(*test_datasets):
            x_batches = [b[0] for b in batches]
            y_batch = tf.cast(batches[0][1], tf.int32)

            outputs = [self.sous_reseaux[i](x_batches[i], training=False) for i in range(self.k)]
            preds = [tf.argmax(o, axis=-1, output_type=tf.int32) for o in outputs]

            for i in range(self.k):
                all_preds_agents[i].append(preds[i])

            if strategy == "majority":
                votes = tf.one_hot(tf.stack(preds), depth=self.num_classes)
                votes_sum = tf.reduce_sum(votes, axis=0)
                ensemble_pred = tf.argmax(votes_sum, axis=-1, output_type=tf.int32)
            elif strategy == "vote":
                batch_size = tf.shape(preds[0])[0]
                rand_idx = tf.random.uniform([batch_size], 0, self.k, tf.int32)
                preds_stack = tf.stack(preds, axis=1)
                ensemble_pred = tf.gather_nd(preds_stack, tf.stack([tf.range(batch_size), rand_idx], axis=1))
            else:
                raise ValueError(f"Unknown strategy '{strategy}'")

            all_preds_ensemble.append(ensemble_pred)
            all_trues.append(y_batch)

        all_trues = tf.concat(all_trues, axis=0)
        all_preds_ensemble = tf.concat(all_preds_ensemble, axis=0)
        all_preds_agents = [tf.concat(all_preds_agents[i], axis=0) for i in range(self.k)]

        acc_ensemble = tf.reduce_mean(tf.cast(tf.equal(all_preds_ensemble, all_trues), tf.float32))
        if verbose:
            print("\nEvaluation des agents :")
            for i in range(self.k):
                acc_i = tf.reduce_mean(tf.cast(tf.equal(all_preds_agents[i], all_trues), tf.float32))
                print(f"  Agent {i + 1} - Acc: {float(acc_i):.4f}")
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

# ---------------------------------------------------------------------
# Module guard
# ---------------------------------------------------------------------
if __name__ == '__main__':
    # Minimal smoke test when lancé directement (n'exécute rien de lourd)
    print("Module chargé : utilitaires multi-agent prêts.")
    print(f"GPU disponible: {on_gpu}")
