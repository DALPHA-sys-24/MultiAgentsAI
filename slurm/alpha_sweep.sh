#!/bin/bash
#SBATCH --job-name=alpha_sweep
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-59
#SBATCH --output=Logs/alpha_sweep_%A_%a.out
#SBATCH --error=Logs/alpha_sweep_%A_%a.err

# =============================================================================
# BALAYAGE DE ALPHA : impact de la collaboration
# Une tâche = un dataset x une méthode (ta loss, DML...) x un run x une valeur de alpha
# 2 datasets x 2 méthodes x 5 runs x 3 alphas = 60 tâches -> --array=0-59
# À lancer depuis la racine du dépôt :  sbatch slurm/alpha_sweep.sh
# Une tâche relancée reprend à la dernière epoch terminée (ou est sautée si elle est finie).
# Sortie : runs/alpha_sweep_<arch>_<dataset>_x<K>[_<tag>]/ ; results.json écrit par le job de fusion.
# =============================================================================
DATASETS=("fashion_mnist" "cifar10")
N_RUNS=5                     # graines 0, 1, ..., N_RUNS-1
ALPHAS=(0 1 10)              # alpha = 0 : agents indépendants, consensus par vote majoritaire

METHODS=(                    # une méthode par ligne : "loss_type:forme:tag" (tag = suffixe du dossier)
    "global_interaction:multiplicative:"     # ta méthode
    "dml:additive:dml"                       # baseline DML (Zhang et al. 2018) ; alpha = 1 : DML classique
)

N_DATASETS=${#DATASETS[@]}
N_ALPHAS=${#ALPHAS[@]}
N_METHODS=${#METHODS[@]}
TASKS_PER_METHOD=$((N_RUNS * N_ALPHAS))
TASKS_PER_DATASET=$((N_METHODS * TASKS_PER_METHOD))
# N_TOTAL = N_DATASETS x N_METHODS x N_RUNS x N_ALPHAS (valeur actuelle : 2 x 2 x 5 x 3 = 60 -> --array=0-59)

# =============================================================================
# HYPERPARAMETRES (identiques pour tous les datasets ; le reste vient de configs/alpha_sweep.json)
# =============================================================================
CONFIG="configs/alpha_sweep.json"
ARCH="resnet20"
NUM_AGENTS=5
USE_BEST=1                   # 1 : si runs/best_hyperparams_<ARCH>_<dataset>/ existe (experiments.best_hyperparams),
                             #     ses paramètres (optimiseur, lr, weight decay, batch...) remplacent les 3 lignes
                             #     ci-dessous, pour chaque dataset ; sinon (ou USE_BEST=0) : les valeurs ci-dessous
OPTIMIZER="sgd"
LEARNING_RATE=0.01
EPOCHS=200
BATCH_SIZE=128
TAG=""                       # suffixe du dossier pour une variante (vide : aucun)
VERBOSE=2

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
METHOD_IDX=$((TASK_IDX / TASKS_PER_METHOD))
EXP_IDX=$((TASK_IDX % TASKS_PER_METHOD))          # indice de la tâche dans l'expérience (--task-index)
IFS=":" read -r LOSS_TYPE LOSS_FORM METHOD_TAG <<< "${METHODS[$METHOD_IDX]}"
RUN_IDX=$((EXP_IDX / N_ALPHAS))
ALPHA=${ALPHAS[$((EXP_IDX % N_ALPHAS))]}

# Paramètres d'entraînement : ceux du best model du dataset s'il existe, sinon ceux du script
BEST="runs/best_hyperparams_${ARCH}_${DATASET}/best_alpha_sweep.json"
if [ "$USE_BEST" = 1 ] && [ -f "$BEST" ]; then
    CONFIG="$BEST"
    HPARAMS=()
    BEST_TAG="best"
    HP_SOURCE="best model ($BEST)"
else
    HPARAMS=(--optimizer "$OPTIMIZER" --lr "$LEARNING_RATE" --batch-size "$BATCH_SIZE")
    BEST_TAG=""
    HP_SOURCE="script (pas de $BEST)"
    [ "$USE_BEST" = 1 ] || HP_SOURCE="script (USE_BEST=0)"
fi

ARGS=(--config "$CONFIG" --dataset "$DATASET" --arch "$ARCH" --num-agents "$NUM_AGENTS"
      --loss-type "$LOSS_TYPE" --loss-form "$LOSS_FORM" "${HPARAMS[@]}"
      --epochs "$EPOCHS" --alphas "${ALPHAS[@]}" --n-runs "$N_RUNS")
# Dossier : <exp>_<arch>_<dataset>_x<K>[_best][_<TAG>][_<tag de la méthode>]
FULL_TAG=$(printf "%s\n" "$BEST_TAG" "$TAG" "$METHOD_TAG" | grep -v '^$' | paste -sd_ -)
[ -n "$FULL_TAG" ] && ARGS+=(--tag "$FULL_TAG")

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
echo "BALAYAGE DE ALPHA"
echo "=========================================="
echo "Date: $(date)"
echo "Job: $SLURM_ARRAY_TASK_ID / $((N_DATASETS * TASKS_PER_DATASET - 1))"
echo "Paramètres: $HP_SOURCE"
echo "Méthode: $LOSS_TYPE ($LOSS_FORM)${FULL_TAG:+ | dossier _$FULL_TAG}"
echo "Dataset: $DATASET"
echo "Graine: $RUN_IDX (run $((RUN_IDX + 1))/$N_RUNS) | alpha = $ALPHA"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "=========================================="

# =============================================================================
# EXECUTION
# =============================================================================
check_array_size $((N_DATASETS * TASKS_PER_DATASET))
submit_merge alpha_sweep "$EXP_IDX" "${ARGS[@]}"     # la 1re tâche de chaque (dataset, méthode) programme la fusion

python -m experiments.alpha_sweep "${ARGS[@]}" --task-index "$EXP_IDX" --verbose "$VERBOSE"

# =============================================================================
# RESULTAT
# =============================================================================
exit_code=$?
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: $DATASET | graine $RUN_IDX | alpha $ALPHA"
else
    echo "FAILED: $DATASET | graine $RUN_IDX | alpha $ALPHA (code $exit_code)"
fi
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
