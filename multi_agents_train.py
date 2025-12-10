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

if not os.path.exists("results"):
    os.makedirs("results")


hidden_dim = [50, 150, 300, 450, 600, 750, 900, 1200] 
global_parameters = {
    'batch_size': 100, 
    'epochs': 10, 
    'R': 5, 
    'verbose': False,
    'optimizer': 'sgd',  # 'sgd' ou 'adamw'
    'num_classes': 10
}

ARRAYS = [
    {'data': 'mnist',          'strategy': 'majority', 'rate_sgd': 0.001, 'rate_adamw':0.0045, 'input_shape': (28, 28, 1), 'input_shape_linear': None},
    {'data': 'fashion_mnist',  'strategy': 'majority', 'rate_sgd': 0.001, 'rate_adamw':0.0034, 'input_shape': (28, 28, 1), 'input_shape_linear': None},
    {'data': 'cifar10',        'strategy': 'majority', 'rate_sgd': 0.001, 'rate_adamw':0.0004, 'input_shape': (32, 32, 3), 'input_shape_linear': None},
    {'data': 'mnist',          'strategy': 'vote',     'rate_sgd': 0.001, 'rate_adamw':0.0045, 'input_shape': (28, 28, 1), 'input_shape_linear': None},
    {'data': 'fashion_mnist',  'strategy': 'vote',     'rate_sgd': 0.001, 'rate_adamw':0.0034, 'input_shape': (28, 28, 1), 'input_shape_linear': None},
    {'data': 'cifar10',        'strategy': 'vote',     'rate_sgd': 0.001, 'rate_adamw':0.0004, 'input_shape': (32, 32, 3), 'input_shape_linear': None},
]

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

# ============================================================
# PARAMÈTRES SLURM
# ============================================================
_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
R = global_parameters.get('R')
number_of_classes = global_parameters.get('num_classes')
epochs = global_parameters.get('epochs')
verbose = global_parameters.get('verbose')
batch_size = global_parameters.get('batch_size')
optimizer_type = global_parameters.get('optimizer')



# Configuration dataset
dataset_name = ARRAYS[_ID].get('data')
strategy = ARRAYS[_ID].get('strategy')
linear = True
input_shape = ARRAYS[_ID].get('input_shape_linear') if linear else ARRAYS[_ID].get('input_shape')

# Sélection du learning rate selon l'optimiseur
if optimizer_type == 'sgd':
    rate = ARRAYS[_ID].get('rate_sgd')
    optimizer_params = None
else:  # adamw
    rate = ARRAYS[_ID].get('rate_adamw')
    optimizer_params = ADAMW_PARAMS.get(dataset_name)

print(f"[CONFIG] Dataset={dataset_name}, Strategy={strategy}, Optimizer={optimizer_type}, LR={rate}")

# ============================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================
x_train, y_train, x_test, y_test = mat.get_data(dataset_name=dataset_name)
print(f"[DATA] Train: {x_train.shape}, Test: {x_test.shape}")

with tf.device(device):
    # Boucle sur différents nombres d'agents K
    for K in [1,3,5,7]:
        print(f"[SLURM] ARRAY_ID={_ID}  K={K}")
        # Split des données en K groupes synchronisés
        train_groups_x, train_groups_y = mat.split_dataset_fully_synchronized(x_train, y_train, num_groups=K)
        # Création des datasets TensorFlow
        train_datasets = mat.make_tf_datasets_from_groups(train_groups_x, train_groups_y, batch_size=batch_size, shuffle=False)
        # Dataset de boost (test unifié)
        boost_x, boost_y = mat.split_dataset_fully_synchronized(x_test, y_test, num_groups=1, seed=24)
        boost_ = mat.make_tf_datasets_from_groups(boost_x, boost_y, batch_size=batch_size, shuffle=False)
        test_datasets = [boost_[0] for _ in range(K)]
        # Vérification de la synchronisation
        if not mat.verify_batch_synchronization(train_datasets):
            print("ERROR: Training batches NOT synchronized.")
            sys.exit(1)

        print("[DATA] Datasets créés et synchronisés.")

        # ============================================================
        # ENTRAÎNEMENT ET ÉVALUATION
        # ============================================================
        all_results = []
        print(f"Multiple runs for K = {K}")
        
        for h in hidden_dim:
            print(f"\nTesting hidden_dim = {h}")
            accuracies_for_h = []
            
            for r in range(R):
                print(f"  Run {r+1}/{R}", end='\r')
                
                # Créer le modèle avec les bons paramètres
                model = mat.MultiAgentAI(
                    k=K, 
                    hidden_size=int(h/K), 
                    lr=rate,
                    input_shape=input_shape,
                    num_classes=number_of_classes,
                    optimizer_type=optimizer_type,
                    batch_size=batch_size,
                    optimizer_kwargs=optimizer_params  # Passé si adamw
                )
                
                history = model.train_step(
                    train_datasets=train_datasets,
                    epochs=epochs, 
                    test_datasets=None,
                    strategy=strategy,
                    verbose=verbose,
                    alpha=1.0, 
                    beta=0.0
                )
                
                accuracy = model.evaluate_ensemble(test_datasets, strategy=strategy, verbose=False)
                accuracies_for_h.append(accuracy)
            
            # Calculer les statistiques
            median_acc = np.median(accuracies_for_h)
            q05_acc = np.quantile(accuracies_for_h, 0.05)
            q95_acc = np.quantile(accuracies_for_h, 0.95)
            
            # Créer une ligne avec toutes les infos
            row = {
                'N': h,
                **{f'r{i+1}': acc for i, acc in enumerate(accuracies_for_h)},
                'med': median_acc,
                'q1': q05_acc,
                'q2': q95_acc
            }
            
            all_results.append(row)
            print(f"\n  N={h}: med={median_acc:.4f}, q1={q05_acc:.4f}, q2={q95_acc:.4f}")
        
        df = pd.DataFrame(all_results)
        df.set_index('N', inplace=True)

        # Sauvegarder en CSV
        filename = f'results/accuracy_results{_ID}_{K}.csv'
        df.to_csv(filename)
        print(f"\n=== Données sauvegardées dans {filename} ===")

print("\n=== FIN ===")