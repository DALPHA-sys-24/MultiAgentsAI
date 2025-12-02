import os
import sys
import numpy as np
import pandas as pd

# ============================================================
# LIMITATION DES THREADS (évite erreurs pthread_create)
# ============================================================
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================
# IMPORTS
# ============================================================
try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow non installé.")
    sys.exit(1)

try:
    import multi_agents_tools as mat
except ImportError:
    print("ERROR: module 'multi_agents_tools' introuvable.")
    sys.exit(1)

# ============================================================
# GPU CONFIG
# ============================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        device = "/GPU:0"
        print("GPU détecté et configuré.")
    except RuntimeError as e:
        print(f"Erreur configuration GPU: {e}")
        device = "/CPU:0"
else:
    device = "/CPU:0"
    print("Aucun GPU détecté, utilisation du CPU.")

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

print(f"TensorFlow version: {tf.__version__}")
print(f"Device: {device}")

# ============================================================
# SLURM IDS
# ============================================================
ARRAY_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
PROC_ID = int(os.environ.get("SLURM_PROCID", -1))

print(f"[SLURM] ARRAY_ID={ARRAY_ID}  PROC_ID={PROC_ID}")

# ============================================================
# CONFIG
# ============================================================
if not os.path.exists("results"):
    os.makedirs("results")

# Paramètres optimisés AdamW par dataset
ADAMW_PARAMS = {
    "mnist": {
        "beta_1": 0.9263308979269205,
        "beta_2": 0.9861890938090508,
        "epsilon": 2.7851878986840997e-08,
        "weight_decay": 7.910186675859548e-06
    },
    "fashion_mnist": {
        "beta_1": 0.8759167704333098,
        "beta_2": 0.9837977048616715,
        "epsilon": 1.7513459879031582e-07,
        "weight_decay": 1.5718277562655645e-06
    },
    "cifar10": {
        "beta_1": 0.9420998974853213,
        "beta_2": 0.9584291321636338,
        "epsilon": 7.991172694690768e-08,
        "weight_decay": 0.0007305335070905817
    }
}

# Configuration du dataset (MNIST par défaut)
DATASET_CONFIG = {
    'data': 'mnist',
    'strategy': 'majority',
    'rate': 0.0045,
    'input_shape':None
}

# ============================================================
# PARAMETRES GRID SEARCH ALPHA-BETA
# ============================================================
R = 5                    # Nombre de répétitions
batch_size = 100
epochs = 10
verbose = False
hidden_dim = [1200]
K = 7                    # Nombre d'agents
a = 2                    # Amplitude [-a, a]
N = 10                   # Nombre de points par dimension
h = 2 * a / N            # Pas de la grille

# Création de la grille alpha-beta
ALPHA = np.arange(-a, a, h).astype(np.float32)
BETA = np.arange(-a, a, h).astype(np.float32)
ALPHA_BETA = [(alpha, beta) for alpha in ALPHA for beta in BETA]

print(f"[CONFIG] Grille: {len(ALPHA)}x{len(BETA)} = {len(ALPHA_BETA)} combinaisons")
print(f"[CONFIG] ALPHA range: [{ALPHA[0]:.4f}, {ALPHA[-1]:.4f}]")
print(f"[CONFIG] BETA range: [{BETA[0]:.4f}, {BETA[-1]:.4f}]")

# ============================================================
# CALCUL DE LA POSITION DANS LA GRILLE
# ============================================================
if PROC_ID > -1:
    # Mode multi-processus SLURM
    POSITION = PROC_ID + N * ARRAY_ID
else:
    # Mode array simple
    POSITION = ARRAY_ID

# Vérification des bornes
if POSITION >= len(ALPHA_BETA):
    print(f"ERROR: POSITION={POSITION} >= {len(ALPHA_BETA)} combinaisons.")
    print(f"       ARRAY_ID doit être entre 0 et {len(ALPHA)-1}")
    print(f"       PROC_ID doit être entre 0 et {len(BETA)-1}")
    sys.exit(1)

# Récupération des valeurs alpha et beta
alpha_val, beta_val = ALPHA_BETA[POSITION]
print(f"[CONFIG] Testing alpha={alpha_val:.6f}, beta={beta_val:.6f} (POSITION={POSITION})")

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
print(f"\n[DATA] Chargement du dataset '{DATASET_CONFIG['data']}'...")

x_train, y_train, x_test, y_test = mat.get_data(dataset_name=DATASET_CONFIG['data'])

print(f"[DATA] Train: {x_train.shape}, Test: {x_test.shape}")

# Split des données en K groupes synchronisés
train_groups_x, train_groups_y = mat.split_dataset_fully_synchronized(
    x_train, y_train, num_groups=K
)

test_groups_x, test_groups_y = mat.split_dataset_fully_synchronized(
    x_test, y_test, num_groups=K
)

# Création des datasets TensorFlow
train_datasets = mat.make_tf_datasets_from_groups(
    train_groups_x, train_groups_y, batch_size=batch_size, shuffle=False
)

# Dataset de boost (test unifié)
boost_x, boost_y = mat.split_dataset_fully_synchronized(
    x_test, y_test, num_groups=1, seed=24
)

boost_ = mat.make_tf_datasets_from_groups(
    boost_x, boost_y, batch_size=batch_size, shuffle=False
)

test_datasets = [boost_[0] for _ in range(K)]

# Vérification de la synchronisation
if not mat.verify_batch_synchronization(train_datasets):
    print("ERROR: Training batches NOT synchronized.")
    sys.exit(1)

print("[DATA] Datasets créés et synchronisés.")

# ============================================================
# BOUCLE D'ENTRAÎNEMENT
# ============================================================
print(f"\n[TRAINING] Début des {R} répétitions...\n")

all_results = []

for h in hidden_dim:
    accuracies = []

    for r in range(R):
        print(f"[RUN {r+1}/{R}] hidden_dim={h}, alpha={alpha_val:.4f}, beta={beta_val:.4f}")

        # Création du modèle sur le device approprié
        with tf.device(device):
            model = mat.MultiAgentAI(
                k=K,
                hidden_size=int(h / K),
                lr=DATASET_CONFIG['rate'],
                input_shape=DATASET_CONFIG['input_shape'],
                num_classes=10,
                optimizer_type='adamw',
                optimizer_kwargs=ADAMW_PARAMS[DATASET_CONFIG['data']],
                batch_size=batch_size
            )

            # Entraînement
            history = model.train_step(
                train_datasets=train_datasets,
                epochs=epochs,
                test_datasets=None,
                strategy=DATASET_CONFIG['strategy'],
                verbose=verbose,
                alpha=alpha_val,
                beta=beta_val
            )

        # Évaluation
        acc = model.evaluate_ensemble(
            test_datasets,
            strategy=DATASET_CONFIG['strategy'],
            verbose=False
        )

        accuracies.append(acc)
        print(f"  → Accuracy: {acc:.4f}")

        # Libération de la mémoire
        del model
        tf.keras.backend.clear_session()

    # Calcul des statistiques
    median_acc = np.median(accuracies)
    q05_acc = np.quantile(accuracies, 0.05)
    q95_acc = np.quantile(accuracies, 0.95)

    print(f"\n[STATS] hidden={h}")
    print(f"  Median: {median_acc:.4f}")
    print(f"  Q05-Q95: [{q05_acc:.4f}, {q95_acc:.4f}]\n")

    all_results.append({
        'hidden': h,
        **{f'r{i+1}': acc for i, acc in enumerate(accuracies)},
        'median': median_acc,
        'q05': q05_acc,
        'q95': q95_acc
    })

# ============================================================
# SAUVEGARDE DES RÉSULTATS
# ============================================================
df = pd.DataFrame(all_results).set_index("hidden")

# Nom de fichier avec valeurs réelles ET indices pour débug
filename = f"results/accuracy_a{alpha_val:.4f}_b{beta_val:.4f}_pos{POSITION:03d}.csv"

try:
    df.to_csv(filename)
    print(f"[SAVE] Résultats enregistrés dans '{filename}'")
    print(f"       (ARRAY_ID={ARRAY_ID}, PROC_ID={PROC_ID}, POSITION={POSITION})")
    print(f"\n{df}\n")
except Exception as e:
    print(f"ERROR: Impossible d'écrire le fichier '{filename}'")
    print(f"       {e}")
    sys.exit(1)

# ============================================================
# SAUVEGARDE DU MAPPING (une seule fois par job)
# ============================================================
mapping_file = f"results/mapping_pos{POSITION:03d}.txt"
try:
    with open(mapping_file, 'w') as f:
        f.write(f"POSITION={POSITION}\n")
        f.write(f"ARRAY_ID={ARRAY_ID}\n")
        f.write(f"PROC_ID={PROC_ID}\n")
        f.write(f"alpha={alpha_val:.6f}\n")
        f.write(f"beta={beta_val:.6f}\n")
    print(f"[SAVE] Mapping enregistré dans '{mapping_file}'")
except Exception as e:
    print(f"WARNING: Impossible d'écrire le mapping: {e}")

print("[FIN] Expérience terminée avec succès.")