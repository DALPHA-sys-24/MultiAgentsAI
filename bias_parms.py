import pandas as pd
import numpy as np

# Paramètres fixes
K = 1
batch_size = 100
input_dim = 784
output_dim = 10
lr = 0.001
epochs = 10
verbose = False
name_data = 'mnist'

# Paramètres à varier
hidden_dims = [50, 150, 300, 450, 600, 750, 900, 1200]
j_values = list(range(10))
c_values = list(range(10))

# Nombre de runs
n_runs = 5

# Liste pour stocker tous les résultats
results = []
results_all_runs = []  # Pour garder tous les runs si besoin

# Boucle principale
for hidden_dim in hidden_dims:
    print(f"\n{'='*50}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"{'='*50}")
    
    for j in j_values:
        print(f"\n  Training avec j={j} (augmentation classe {j})...")
        
        # Dictionnaire pour stocker les accuracies de chaque run par classe c
        accuracies_per_class = {c: [] for c in c_values}
        
        for run in range(n_runs):
            print(f"    Run {run + 1}/{n_runs}...", end=" ")
            
            # Charger et préparer les données d'entraînement augmentées
            x_train, y_train, _, _ = mat.get_data(name_data, j=j)
            train_groups_x, train_groups_y = mat.split_dataset_fully_synchronized(
                x_train, y_train, num_groups=K, seed=None
            )
            train_datasets = mat.make_tf_datasets_from_groups(
                train_groups_x, train_groups_y, batch_size=batch_size, shuffle=False
            )
            
            # Entraîner le modèle
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
                
                # Évaluer sur chaque classe c
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
                        
                        # Stocker l'accuracy pour ce run
                        accuracies_per_class[c].append(accuracy)
                        
                        # Stocker le résultat détaillé de chaque run
                        results_all_runs.append({
                            'hidden_dim': hidden_dim,
                            'j_augment': j,
                            'c_test_class': c,
                            'run': run + 1,
                            'accuracy': accuracy
                        })
            
            print("Done")
        
        # Calculer la médiane pour chaque classe c
        for c in c_values:
            if len(accuracies_per_class[c]) > 0:
                median_accuracy = np.median(accuracies_per_class[c])
                mean_accuracy = np.mean(accuracies_per_class[c])
                std_accuracy = np.std(accuracies_per_class[c])
                min_accuracy = np.min(accuracies_per_class[c])
                max_accuracy = np.max(accuracies_per_class[c])
                
                # Stocker le résultat agrégé
                results.append({
                    'hidden_dim': hidden_dim,
                    'j_augment': j,
                    'c_test_class': c,
                    'accuracy_median': median_accuracy,
                    'accuracy_mean': mean_accuracy,
                    'accuracy_std': std_accuracy,
                    'accuracy_min': min_accuracy,
                    'accuracy_max': max_accuracy,
                    'n_runs': len(accuracies_per_class[c])
                })
                
                if verbose:
                    print(f"    Classe c={c}: median={median_accuracy:.4f}, "
                          f"mean={mean_accuracy:.4f}, std={std_accuracy:.4f}")
        
        print(f"  Évaluations terminées pour j={j}")

# Créer les DataFrames
df_results = pd.DataFrame(results)
df_all_runs = pd.DataFrame(results_all_runs)

# Afficher un aperçu
print("\n" + "="*60)
print("RÉSULTATS AGRÉGÉS (MÉDIANE)")
print("="*60)
print(df_results.head(20))

# Sauvegarder les résultats
df_results.to_csv('results_median.csv', index=False)
df_all_runs.to_csv('results_all_runs.csv', index=False)

print("\nFichiers sauvegardés:")
print("  - 'results_median.csv' (résultats agrégés)")
print("  - 'results_all_runs.csv' (tous les runs détaillés)")