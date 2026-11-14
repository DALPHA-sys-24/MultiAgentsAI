#!/bin/bash
#SBATCH --job-name=collaboration
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-69
#SBATCH --output=slurm_logs/collaboration_%A_%a.out
#SBATCH --error=slurm_logs/collaboration_%A_%a.err

# =============================================================================
# COLLABORATION : K agents pour chaque alpha, et (option) un grand modèle seul au même nombre de paramètres
# Une tâche = un dataset x une méthode (ta loss, DML...) x un run x (une valeur de alpha | le grand modèle)
# Par méthode : N_RUNS x (N_ALPHAS + 1 si grand modèle) tâches
#   valeurs actuelles : ta méthode 5 x (3 + 1) = 20, DML 5 x 3 = 15 -> 35 par dataset
#   2 datasets x 35 = 70 tâches -> --array=0-69 (le script prévient si --array ne correspond pas)
# Un run simple : ALPHAS=(1). Un balayage : ALPHAS=(0 1 10 ...).
# À lancer depuis la racine du dépôt :  sbatch slurm/collaboration.sh
# Une tâche relancée reprend à la dernière epoch terminée (ou est sautée si elle est finie).
# Les K agents d'un run sont dans la même tâche : ils sont entraînés ensemble (loss commune).
# Sortie : results/collaboration/<dataset>/<arch>_<K>agents[_<tag>]/ ; results.json écrit par le job de fusion.
# =============================================================================
DATASETS=("fashion_mnist" "cifar10")
N_RUNS=5                     # graines 0, 1, ..., N_RUNS-1
ALPHAS=(0 1 10)              # alpha = 0 : agents indépendants (cross-entropy classique), vote majoritaire

METHODS=(                    # une méthode par ligne : "loss_type:forme:tag:grand modèle (1/0)" (tag = suffixe du dossier)
    "global_interaction:multiplicative::1"   # ta méthode, + le grand modèle (entraîné une seule fois, ici)
    "dml:additive:dml:0"                     # baseline DML (Zhang et al. 2018) ; alpha = 1 : DML classique
)

# =============================================================================
# HYPERPARAMETRES (identiques pour tous les datasets ; le reste vient de configs/collaboration.json)
# =============================================================================
CONFIG="configs/collaboration.json"
ARCH="resnet20"              # architecture des agents (le grand modèle a la même, élargie)
NUM_AGENTS=3
USE_TUNED_PARAMS=1           # 1 : si results/hyperparameter_search/<dataset>/<ARCH>/ existe (experiments.hyperparameter_search),
                             #     ses paramètres (optimiseur, lr, weight decay, batch...) remplacent les 3 lignes
                             #     ci-dessous, pour chaque dataset ; sinon (ou USE_TUNED_PARAMS=0) : les valeurs ci-dessous
OPTIMIZER="sgd"
LEARNING_RATE=0.01
BATCH_SIZE=128
EPOCHS=200
TAG=""                       # suffixe du dossier pour une variante (vide : aucun)
VERBOSE=2

# =============================================================================
# CALCUL DES INDICES
# =============================================================================
N_DATASETS=${#DATASETS[@]}
N_ALPHAS=${#ALPHAS[@]}
TASKS_PER_DATASET=0          # somme sur les méthodes de N_RUNS x (N_ALPHAS + grand modèle)
for m in "${METHODS[@]}"; do
    IFS=":" read -r _ _ _ big <<< "$m"
    TASKS_PER_DATASET=$((TASKS_PER_DATASET + N_RUNS * (N_ALPHAS + ${big:-0})))
done
N_TOTAL=$((N_DATASETS * TASKS_PER_DATASET))

DATASET_IDX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_DATASET))
TASK_IDX=$((SLURM_ARRAY_TASK_ID % TASKS_PER_DATASET))
if [ "$DATASET_IDX" -ge "$N_DATASETS" ]; then
    echo "Tâche $SLURM_ARRAY_TASK_ID hors de la liste ($N_TOTAL tâches) : rien à faire."
    exit 0
fi
DATASET=${DATASETS[$DATASET_IDX]}

# Méthode de la tâche : les tâches d'une méthode se suivent
OFFSET=0
for m in "${METHODS[@]}"; do
    IFS=":" read -r LOSS_TYPE LOSS_FORM METHOD_TAG BIG <<< "$m"
    BIG=${BIG:-0}
    N_METHOD=$((N_RUNS * (N_ALPHAS + BIG)))
    [ "$TASK_IDX" -lt $((OFFSET + N_METHOD)) ] && break
    OFFSET=$((OFFSET + N_METHOD))
done
EXP_IDX=$((TASK_IDX - OFFSET))                     # indice de la tâche dans l'expérience (--task-index)
PER_SEED=$((N_ALPHAS + BIG))                       # par graine : un alpha après l'autre, puis le grand modèle
RUN_IDX=$((EXP_IDX / PER_SEED))
J=$((EXP_IDX % PER_SEED))
if [ "$J" -lt "$N_ALPHAS" ]; then
    WHAT="alpha = ${ALPHAS[$J]}"
else
    WHAT="grand modèle"
fi
BIG_FLAG=$([ "$BIG" = 1 ] && echo true || echo false)

# Paramètres d'entraînement : ceux de la recherche d'hyperparamètres du dataset s'il y en a une, sinon ceux du script
TUNED_CONFIG="results/hyperparameter_search/${DATASET}/${ARCH}/tuned_collaboration.json"
if [ "$USE_TUNED_PARAMS" = 1 ] && [ -f "$TUNED_CONFIG" ]; then
    CONFIG="$TUNED_CONFIG"
    HPARAMS=()
    TUNED_TAG="tuned"
    HP_SOURCE="recherche d'hyperparamètres ($TUNED_CONFIG)"
else
    HPARAMS=(--optimizer "$OPTIMIZER" --lr "$LEARNING_RATE" --batch-size "$BATCH_SIZE")
    TUNED_TAG=""
    HP_SOURCE="script (pas de $TUNED_CONFIG)"
    [ "$USE_TUNED_PARAMS" = 1 ] || HP_SOURCE="script (USE_TUNED_PARAMS=0)"
fi

ARGS=(--config "$CONFIG" --dataset "$DATASET" --arch "$ARCH" --num-agents "$NUM_AGENTS"
      --loss-type "$LOSS_TYPE" --loss-form "$LOSS_FORM" "${HPARAMS[@]}"
      --epochs "$EPOCHS" --alphas "${ALPHAS[@]}" --big-model "$BIG_FLAG" --n-runs "$N_RUNS")
# Dossier : results/collaboration/<dataset>/<arch>_<K>agents[_<tag de la méthode>][_<TAG>][_tuned]
FULL_TAG=$(printf "%s\n" "$METHOD_TAG" "$TAG" "$TUNED_TAG" | grep -v '^$' | paste -sd_ -)
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

[ -d collaborative_learning ] || { echo "Lancer sbatch depuis la racine du dépôt (dossier qui contient collaborative_learning/)"; exit 1; }
source slurm/helpers.sh

# =============================================================================
# DEBUG INFO
# =============================================================================
echo "=========================================="
echo "COLLABORATION"
echo "=========================================="
echo "Date: $(date)"
echo "Job: $SLURM_ARRAY_TASK_ID / $((N_TOTAL - 1))"
echo "Paramètres: $HP_SOURCE"
echo "Méthode: $LOSS_TYPE ($LOSS_FORM)${FULL_TAG:+ | dossier _$FULL_TAG}"
echo "Dataset: $DATASET"
echo "Graine: $RUN_IDX (run $((RUN_IDX + 1))/$N_RUNS) | $WHAT"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "=========================================="

# =============================================================================
# EXECUTION
# =============================================================================
check_array_size "$N_TOTAL"
submit_merge collaboration "$EXP_IDX" "${ARGS[@]}"     # la 1re tâche de chaque (dataset, méthode) programme la fusion

python -m experiments.collaboration "${ARGS[@]}" --task-index "$EXP_IDX" --verbose "$VERBOSE"

# =============================================================================
# RESULTAT
# =============================================================================
exit_code=$?
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: $DATASET | graine $RUN_IDX | $WHAT"
else
    echo "FAILED: $DATASET | graine $RUN_IDX | $WHAT (code $exit_code)"
fi
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
