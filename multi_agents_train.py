import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import multi_agents_tools  as mat
print(tf.__version__)


R =5
batch_size=100
epochs = 10
verbose=False
hidden_dim = [450] 
if len(sys.argv) < 2:
    print("Erreur : argument 'nom' manquant.")
    sys.exit(1)
    
_ID = int(sys.argv[1])-1
ARRAYS=[{'data':'mnist','strategy':'majority','rate':0.001,'input_shape':(28,28,1)},
        {'data':'fashion_mnist','strategy':'majority','rate':0.001,'input_shape':(28,28,1)},
        {'data':'cifar10','strategy':'majority','rate':0.001,'input_shape':(32,32,3)},
        {'data':'mnist','strategy':'vote','rate':0.001,'input_shape':(28,28,1)},
        {'data':'fashion_mnist','strategy':'vote','rate':0.001,'input_shape':(28,28,1)},
        {'data':'cifar10','strategy':'vote','rate':0.001,'input_shape':(32,32,3)}]

x_train,y_train, x_test, y_test =mat.get_data(dataset_name=ARRAYS[_ID].get('data'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"\n=== Using GPU: {gpus} ===\n")
    except RuntimeError as e:
        print("Erreur lors de la configuration du GPU :", e)
        # Invalid device or cannot modify virtual devices once initialized
else:
    print("\nNo GPU found. Exiting.")
    exit()
    
if gpus:
    with tf.device('/GPU:0'):
        for K in [1,3,5,7]:
            all_results = []
            print("Multiple runs for K =", K)
            train_groups_x, train_groups_y = mat.split_dataset_fully_synchronized(x_train, y_train,num_groups=K)
            test_groups_x, test_groups_y = mat.split_dataset_fully_synchronized(x_test, y_test,num_groups=K)
            train_datasets = mat.make_tf_datasets_from_groups(train_groups_x, train_groups_y, batch_size=batch_size, shuffle=False)
            test_datasets = mat.make_tf_datasets_from_groups(test_groups_x, test_groups_y, batch_size=batch_size, shuffle=False)
            
            if mat.verify_batch_synchronization(train_datasets):
                for h in hidden_dim:
                    print(f"Testing hidden_dim = {h}")
                    accuracies_for_h = []
                    
                    for r in range(R):
                        print(f"  Run {r+1}/{R}", end='\r')
                        model=mat.MultiAgentAI(k=K, hidden_size= int(h/K), lr=ARRAYS[_ID].get('rate'),input_shape=None,num_classes=10,optimizer_type='sgd',batch_size=batch_size)
                        history=model.train_step(train_datasets=train_datasets,epochs=epochs, test_datasets=None,strategy=ARRAYS[_ID].get('strategy'),verbose=verbose,alpha=1.0, beta=0.0)
                        accuracy = model.evaluate_ensemble(test_datasets,strategy=ARRAYS[_ID].get('strategy'),verbose=False)
                        accuracies_for_h.append(accuracy)
                        
                    # Calculer les statistiques
                    median_acc = np.median(accuracies_for_h)
                    q25_acc = np.quantile(accuracies_for_h, 0.05)
                    q95_acc = np.quantile(accuracies_for_h, 0.95)
                    
                    # Créer une ligne avec toutes les infos
                    row = {
                        'N': h,
                        **{f'r{i+1}': acc for i, acc in enumerate(accuracies_for_h)},
                        'med': median_acc,
                        'q1': q25_acc,
                        'q2': q95_acc
                    }
                    
                    all_results.append(row)
                    print(f"\n  N={h}: med={median_acc:.4f}, q1={q25_acc:.4f}, q2={q95_acc:.4f}")
                
                df = pd.DataFrame(all_results)
                df.set_index('N', inplace=True)

                # Sauvegarder en CSV
                filename = f'results/accuracy_results{_ID}_{K}.csv'
                df.to_csv(filename)
                print(f"\n=== Données sauvegardées dans {filename} ===")
            else:
                sys.exit(0)    
    print("\n=== FIN ===")
else:
    print("No GPU found. Please run this code on a machine with a GPU.")