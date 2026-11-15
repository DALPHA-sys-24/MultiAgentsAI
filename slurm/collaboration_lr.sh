#!/bin/bash
#SBATCH --job-name=collaboration_lr
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-89
#SBATCH --output=slurm_logs/collaboration_lr_%A_%a.out
#SBATCH --error=slurm_logs/collaboration_lr_%A_%a.err

# =============================================================================
# COLLABORATION SELON LE LEARNING RATE : quelle méthode collaborative résiste le mieux au choix du lr ?
# K agents (alpha = ALPHA) pour chaque méthode de METHODS et chaque lr de LRS ; tous les autres
# hyperparamètres sont ceux de la recherche (tuned_collaboration.json du dataset).
# Référence : le grand modèle seul (même nombre de paramètres que les K agents), entraîné une fois
# par graine avec le vrai lr* de la recherche.
#
# Une tâche = un dataset x (une méthode x un lr | le grand modèle) x un run
# Par dataset : N_METHODS x N_LRS x N_RUNS + N_RUNS (grand modèle)
#   valeurs actuelles : 2 x 4 x 5 + 5 = 45 par dataset ; 2 datasets -> 90 tâches -> --array=0-89
# À lancer depuis la racine du dépôt :  sbatch slurm/collaboration_lr.sh
# Une tâche relancée reprend à la dernière epoch terminée (ou est sautée si elle est finie).
#
# Dossiers : results/collaboration/<dataset>/<arch>_<K>agents[_<tag méthode>]_lr<lr>[_<TAG>]_tuned/  (agents)
#            results/collaboration/<dataset>/<arch>_<K>agents[_<TAG>]_tuned/                       (grand modèle, lr*)
# Figures  : results/comparisons/<dataset>/<arch>_<K>agents/accuracy_vs_lr.png/.pdf (faites par la fusion)
# =============================================================================
DATASETS=("fashion_mnist" "cifar10")
N_RUNS=5                     # graines 0, 1, ..., N_RUNS-1
LRS=(0.1 0.01 0.001 0.0001)  # lr des agents (le grand modèle garde lr*)
ALPHA=1

METHODS=(                    # une méthode par ligne : "loss_type:forme:tag" (tag = suffixe du dossier)
    "global_interaction:multiplicative:"     # interaction globale (méthode proposée)
    "dml:additive:dml"                       # DML (Zhang et al. 2018) ; alpha = 1 : DML classique
)
BIG_MODEL=1                  # 1 : grand modèle seul à lr*, référence ; 0 : pas de grand modèle

# =============================================================================
# HYPERPARAMETRES (le reste vient de tuned_collaboration.json, ou de configs/collaboration.json)
# =============================================================================
CONFIG="configs/collaboration.json"
ARCH="resnet20"              # architecture des agents (le grand modèle a la même, élargie)
NUM_AGENTS=5
USE_TUNED_PARAMS=1           # 1 : paramètres de results/hyperparameter_search/<dataset>/<ARCH>/tuned_collaboration.json
                             #     (optimiseur, weight decay, batch... et lr* pour le grand modèle) ;
                             #     sinon (pas de recherche, ou 0) : les 3 valeurs ci-dessous, LEARNING_RATE = lr*
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
N_LRS=${#LRS[@]}
N_METHODS=${#METHODS[@]}
N_AGENT_TASKS=$((N_METHODS * N_LRS * N_RUNS))
TASKS_PER_DATASET=$((N_AGENT_TASKS + BIG_MODEL * N_RUNS))
N_TOTAL=$((N_DATASETS * TASKS_PER_DATASET))

DATASET_IDX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_DATASET))
TASK_IDX=$((SLURM_ARRAY_TASK_ID % TASKS_PER_DATASET))
if [ "$DATASET_IDX" -ge "$N_DATASETS" ]; then
    echo "Tâche $SLURM_ARRAY_TASK_ID hors de la liste ($N_TOTAL tâches) : rien à faire."
    exit 0
fi
DATASET=${DATASETS[$DATASET_IDX]}

# Paramètres de la recherche d'hyperparamètres du dataset s'il y en a une, sinon ceux du script
TUNED_CONFIG="results/hyperparameter_search/${DATASET}/${ARCH}/tuned_collaboration.json"
if [ "$USE_TUNED_PARAMS" = 1 ] && [ -f "$TUNED_CONFIG" ]; then
    CONFIG="$TUNED_CONFIG"
    BASE_HPARAMS=()
    TUNED_TAG="tuned"
    HP_SOURCE="recherche d'hyperparamètres ($TUNED_CONFIG)"
    LR_STAR=$(python3 -c "import json; print(json.load(open('$TUNED_CONFIG'))['lr'])" 2>/dev/null || echo "?")
else
    BASE_HPARAMS=(--optimizer "$OPTIMIZER" --batch-size "$BATCH_SIZE")
    TUNED_TAG=""
    HP_SOURCE="script (pas de $TUNED_CONFIG)"
    [ "$USE_TUNED_PARAMS" = 1 ] || HP_SOURCE="script (USE_TUNED_PARAMS=0)"
    LR_STAR="$LEARNING_RATE"
fi

if [ "$TASK_IDX" -lt "$N_AGENT_TASKS" ]; then
    # K agents : une méthode, un lr, une graine (les tâches d'un même dossier se suivent)
    METHOD_IDX=$((TASK_IDX / (N_LRS * N_RUNS)))
    LR_IDX=$(((TASK_IDX / N_RUNS) % N_LRS))
    RUN_IDX=$((TASK_IDX % N_RUNS))
    IFS=":" read -r LOSS_TYPE LOSS_FORM METHOD_TAG <<< "${METHODS[$METHOD_IDX]}"
    LR=${LRS[$LR_IDX]}
    WHAT="$LOSS_TYPE ($LOSS_FORM), alpha = $ALPHA, lr = $LR"
    ARGS=(--config "$CONFIG" --dataset "$DATASET" --arch "$ARCH" --num-agents "$NUM_AGENTS"
          --loss-type "$LOSS_TYPE" --loss-form "$LOSS_FORM" "${BASE_HPARAMS[@]}" --lr "$LR"
          --epochs "$EPOCHS" --alphas "$ALPHA" --big-model false --n-runs "$N_RUNS")
    FULL_TAG=$(printf "%s\n" "$METHOD_TAG" "lr$LR" "$TAG" "$TUNED_TAG" | grep -v '^$' | paste -sd_ -)
    EXP_IDX=$RUN_IDX                              # une tâche par graine dans ce dossier
    MERGE_IDX=$RUN_IDX
else
    # Grand modèle seul à lr* : seule la tâche big_model_seed<s> du dossier est lancée
    # (tâches du dossier par graine : alpha<ALPHA>_seed<s> = 2s, big_model_seed<s> = 2s + 1)
    RUN_IDX=$((TASK_IDX - N_AGENT_TASKS))
    IFS=":" read -r LOSS_TYPE LOSS_FORM _ <<< "${METHODS[0]}"
    WHAT="grand modèle seul, lr* = $LR_STAR"
    ARGS=(--config "$CONFIG" --dataset "$DATASET" --arch "$ARCH" --num-agents "$NUM_AGENTS"
          --loss-type "$LOSS_TYPE" --loss-form "$LOSS_FORM" "${BASE_HPARAMS[@]}")
    [ -z "$TUNED_TAG" ] && ARGS+=(--lr "$LEARNING_RATE")
    ARGS+=(--epochs "$EPOCHS" --alphas "$ALPHA" --big-model true --n-runs "$N_RUNS")
    FULL_TAG=$(printf "%s\n" "$TAG" "$TUNED_TAG" | grep -v '^$' | paste -sd_ -)
    EXP_IDX=$((2 * RUN_IDX + 1))
    MERGE_IDX=$RUN_IDX
fi
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
echo "COLLABORATION SELON LE LR"
echo "=========================================="
echo "Date: $(date)"
echo "Job: $SLURM_ARRAY_TASK_ID / $((N_TOTAL - 1))"
echo "Paramètres: $HP_SOURCE (lr* = $LR_STAR)"
echo "Dataset: $DATASET | $NUM_AGENTS x $ARCH"
echo "Tâche: $WHAT | graine $RUN_IDX (run $((RUN_IDX + 1))/$N_RUNS)${FULL_TAG:+ | dossier _$FULL_TAG}"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "=========================================="

# =============================================================================
# EXECUTION
# =============================================================================
check_array_size "$N_TOTAL"
MERGE_NAME=collaboration_lr_merge submit_merge collaboration "$MERGE_IDX" "${ARGS[@]}"   # 1re graine de chaque dossier

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
