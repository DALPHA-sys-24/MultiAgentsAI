import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
import multi_agents_train as mat

# =============================================================================
# CONFIGURATION GPU AVANCÉE
# =============================================================================

def configure_gpu(memory_limit=None, allow_growth=True, gpu_index=0):
    """
    Configure TensorFlow pour utiliser le GPU.
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"GPU(s) détecté(s): {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        try:
            # Sélectionner un GPU spécifique si plusieurs disponibles
            if len(gpus) > 1:
                tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
                print(f"  GPU {gpu_index} sélectionné")
            
            for gpu in gpus:
                if memory_limit:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                    print(f"  Mémoire limitée à {memory_limit} MB")
                elif allow_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"  Memory growth activé")
            
            print("Configuration GPU réussie!")
            return True
            
        except RuntimeError as e:
            print(f"Erreur de configuration GPU: {e}")
            return False
    else:
        print("Aucun GPU détecté. Utilisation du CPU.")
        return False


def get_gpu_memory_info():
    """Retourne les informations de mémoire GPU."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Utiliser nvidia-smi pour obtenir les infos mémoire
            import subprocess
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            lines = result.strip().split('\n')
            memory_info = []
            for line in lines:
                used, total = map(int, line.split(','))
                memory_info.append({'used': used, 'total': total, 'percent': used/total*100})
            return memory_info
    except Exception:
        return None


def print_device_info():
    """Affiche les informations sur les devices disponibles."""
    print("\n" + "="*50)
    print("INFORMATIONS DEVICES")
    print("="*50)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"Device utilisé: {tf.test.gpu_device_name() or 'CPU'}")
    
    mem_info = get_gpu_memory_info()
    if mem_info:
        for i, info in enumerate(mem_info):
            print(f"GPU {i} Mémoire: {info['used']}MB / {info['total']}MB ({info['percent']:.1f}%)")
    
    print("="*50 + "\n")


# =============================================================================
# FONCTION D'ENTRAÎNEMENT PRINCIPALE
# =============================================================================

def run_experiment(hidden_dims, j_values, c_values, n_runs=5, 
                   batch_size=100, lr=0.001, epochs=10, K=1,
                   name_data='mnist', verbose=False, save_interval=10):
    """
    Exécute l'expérience complète avec GPU.
    
    Parameters
    ----------
    save_interval : int
        Sauvegarde intermédiaire tous les N combinaisons (hidden_dim, j)
    """
    
    # Configurer GPU
    configure_gpu(memory_limit=None, allow_growth=True)
    print_device_info()
    
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"Exécution sur: {device}\n")
    
    results = []
    results_all_runs = []
    
    # Calculer le nombre total d'itérations pour la barre de progression
    total_iterations = len(hidden_dims) * len(j_values) * n_runs
    
    # Barre de progression globale
    pbar = tqdm(total=total_iterations, desc="Progression globale", unit="run")
    
    start_time = time.time()
    iteration_count = 0
    
    for hidden_dim in hidden_dims:
        print(f"\n{'='*50}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"{'='*50}")
        
        for j in j_values:
            accuracies_per_class = {c: [] for c in c_values}
            
            for run in range(n_runs):
                with tf.device(device):
                    # Libérer la mémoire
                    tf.keras.backend.clear_session()
                    
                    # Charger les données
                    x_train, y_train, _, _ = mat.get_data(name_data, j=j)
                    train_groups_x, train_groups_y = mat.split_dataset_fully_synchronized(
                        x_train, y_train, num_groups=K, seed=None
                    )
                    train_datasets = mat.make_tf_datasets_from_groups(
                        train_groups_x, train_groups_y, batch_size=batch_size, shuffle=False
                    )
                    
                    if mat.verify_batch_synchronization(train_datasets):
                        model = mat.MultiAgentAI(
                            k=K,
                            hidden_size=int(hidden_dim / K),
                            lr=lr,
                            input_shape=None,
                            num_classes=10,
                            optimizer_type='sgd',
                            batch_size=batch_size
                        )
                        history = model.train_step(
                            train_datasets=train_datasets,
                            epochs=epochs,
                            test_datasets=None,
                            strategy='majority',
                            verbose=False,
                            alpha=1,
                            beta=0.0,
                            alpha_warmup=0
                        )
                        
                        for c in c_values:
                            _, _, x_test, y_test = mat.get_data(name_data, j=j, filter_class=c)
                            test_groups_x, test_groups_y = mat.split_dataset_fully_synchronized(
                                x_test, y_test, num_groups=K, seed=None
                            )
                            test_datasets = mat.make_tf_datasets_from_groups(
                                test_groups_x, test_groups_y, batch_size=batch_size, shuffle=False
                            )
                            
                            if mat.verify_batch_synchronization(test_datasets):
                                accuracy = model.evaluate_ensemble(
                                    test_datasets, strategy='majority', verbose=False
                                )
                                
                                accuracies_per_class[c].append(accuracy)
                                
                                results_all_runs.append({
                                    'hidden_dim': hidden_dim,
                                    'j_augment': j,
                                    'c_test_class': c,
                                    'run': run + 1,
                                    'accuracy': accuracy
                                })
                
                # Mise à jour de la barre de progression
                pbar.update(1)
                iteration_count += 1
                
                # Afficher les infos GPU périodiquement
                if iteration_count % 10 == 0:
                    mem_info = get_gpu_memory_info()
                    if mem_info:
                        pbar.set_postfix({
                            'GPU_mem': f"{mem_info[0]['percent']:.1f}%",
                            'hidden': hidden_dim,
                            'j': j
                        })
            
            # Calculer les statistiques pour chaque classe
            for c in c_values:
                if len(accuracies_per_class[c]) > 0:
                    results.append({
                        'hidden_dim': hidden_dim,
                        'j_augment': j,
                        'c_test_class': c,
                        'accuracy_median': np.median(accuracies_per_class[c]),
                        'accuracy_mean': np.mean(accuracies_per_class[c]),
                        'accuracy_std': np.std(accuracies_per_class[c]),
                        'accuracy_min': np.min(accuracies_per_class[c]),
                        'accuracy_max': np.max(accuracies_per_class[c]),
                        'n_runs': len(accuracies_per_class[c])
                    })
            
            # Sauvegarde intermédiaire
            if iteration_count % (save_interval * n_runs) == 0:
                pd.DataFrame(results).to_csv('results_median_checkpoint.csv', index=False)
                pd.DataFrame(results_all_runs).to_csv('results_all_runs_checkpoint.csv', index=False)
    
    pbar.close()
    
    # Temps total
    elapsed_time = time.time() - start_time
    print(f"\nTemps total: {elapsed_time/60:.2f} minutes")
    
    # Créer les DataFrames finaux
    df_results = pd.DataFrame(results)
    df_all_runs = pd.DataFrame(results_all_runs)
    
    # Sauvegarder
    df_results.to_csv('results_median.csv', index=False)
    df_all_runs.to_csv('results_all_runs.csv', index=False)
    
    print("\nFichiers sauvegardés:")
    print("  - 'results_median.csv'")
    print("  - 'results_all_runs.csv'")
    
    return df_results, df_all_runs


# =============================================================================
# EXÉCUTION
# =============================================================================

if __name__ == "__main__":
    # Paramètres
    hidden_dims = [50, 150, 300, 450, 600, 750, 900, 1200]
    j_values = list(range(10))
    c_values = list(range(10))
    
    # Lancer l'expérience
    df_results, df_all_runs = run_experiment(
        hidden_dims=hidden_dims,
        j_values=j_values,
        c_values=c_values,
        n_runs=5,
        batch_size=100,
        lr=0.001,
        epochs=10,
        K=1,
        name_data='mnist',
        verbose=False,
        save_interval=10
    )
    
    print("\n" + "="*60)
    print("RÉSULTATS FINAUX")
    print("="*60)
    print(df_results.head(20))