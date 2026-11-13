#!/bin/bash
#SBATCH --job-name=best_hyperparams
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7
#SBATCH --output=Logs/best_hyperparams_%A_%a.out
#SBATCH --error=Logs/best_hyperparams_%A_%a.err

# =============================================================================
# RECHERCHE DES MEILLEURS HYPERPARAMETRES D'UN MODELE (Optuna)
# Un seul modèle ARCH par essai (pas d'agents) ; les meilleurs paramètres donnent ensuite
# runs/best_hyperparams_<arch>_<dataset>/best_collab_vs_big.json pour construire K agents.
# Une tâche = un dataset x un worker ; les workers d'un dataset partagent la même étude
# (N_WORKERS x N_TRIALS essais au total par dataset).
# 1 dataset x 8 workers = 8 tâches -> --array=0-7
# À lancer depuis la racine du dépôt :  sbatch slurm/best_hyperparams.sh
# Optuna n'est pas un module : « pip install --user optuna » une fois sur le nœud de connexion.
# Relancer ajoute des essais à la recherche existante. Espace de recherche : configs/best_hyperparams.json
# Sortie : runs/best_hyperparams_<arch>_<dataset>[_<tag>]/ (best.json, best_collab_vs_big.json, trials.csv)
# =============================================================================
DATASETS=("fashion_mnist")
N_WORKERS=8
N_TRIALS=10                  # essais par worker

N_DATASETS=${#DATASETS[@]}
TASKS_PER_DATASET=$N_WORKERS
# N_TOTAL = N_DATASETS x N_WORKERS (valeur actuelle : 8 -> --array=0-7)

# =============================================================================
# REGLAGES FIXES (le reste, dont search_space, vient de configs/best_hyperparams.json)
# =============================================================================
CONFIG="configs/best_hyperparams.json"
ARCH="resnet20"
LOSS="ce"                    # ce, ou same_as_agents (L_k², échelle de gradient proche des agents)
EPOCHS=20                    # époques par essai
BATCH_SIZE=128
TAG=""
VERBOSE=0

# =============================================================================
# CALCUL DES INDICES
# =============================================================================
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_DATASET))
TASK_IDX=$((SLURM_ARRAY_TASK_ID % TASKS_PER_DATASET))
if [ "$DATASET_IDX" -ge "$N_DATASETS" ]; then
    echo "Tâche $SLURM_ARRAY_TASK_ID hors de la liste ($((N_DATASETS * TASKS_PER_DATASET)) tâches) : rien à faire."
    exit 0
fi
DATASET=${DATASETS[$DATASET_IDX]}

ARGS=(--config "$CONFIG" --dataset "$DATASET" --arch "$ARCH" --loss "$LOSS"
      --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --n-workers "$N_WORKERS" --n-trials "$N_TRIALS")
[ -n "$TAG" ] && ARGS+=(--tag "$TAG")

# =============================================================================
# CHARGEMENT DES MODULES (modules de Lyra, sans environnement virtuel)
# =============================================================================
module --force purge
ml releases/2023a
ml tis/2025
ml CUDA/12.1.1
ml Python/3.11.3-GCCcore-12.3.0
ml SciPy-bundle/2023.07-gfbf-2023a
ml TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
ml matplotlib/3.7.2
ml Seaborn/0.13.2-gfbf-2023a
ml scikit-learn/1.3.1-gfbf-2023a

[ -d dml ] || { echo "Lancer sbatch depuis la racine du dépôt (dossier qui contient dml/)"; exit 1; }
source slurm/common.sh

# =============================================================================
# DEBUG INFO
# =============================================================================
echo "=========================================="
echo "RECHERCHE D'HYPERPARAMETRES"
echo "=========================================="
echo "Date: $(date)"
echo "Job: $SLURM_ARRAY_TASK_ID / $((N_DATASETS * TASKS_PER_DATASET - 1))"
echo "Dataset: $DATASET | worker $TASK_IDX / $N_WORKERS ($N_TRIALS essais)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "=========================================="

# =============================================================================
# EXECUTION
# =============================================================================
check_array_size $((N_DATASETS * TASKS_PER_DATASET))
submit_merge best_hyperparams "$TASK_IDX" "${ARGS[@]}"     # la 1re tâche de chaque dataset programme la fusion

python -m experiments.best_hyperparams "${ARGS[@]}" --task-index "$TASK_IDX" --verbose "$VERBOSE"

# =============================================================================
# RESULTAT
# =============================================================================
exit_code=$?
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: $DATASET | worker $TASK_IDX"
else
    echo "FAILED: $DATASET | worker $TASK_IDX (code $exit_code)"
fi
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
