#!/bin/bash
#SBATCH --job-name=alpha_sweep
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-23:59:00
#SBATCH --array=0-17
#SBATCH --output=Logs/alpha_sweep_%A_%a.out
#SBATCH --error=Logs/alpha_sweep_%A_%a.err

# =============================================================================
# BALAYAGE DE ALPHA : un job par couple (alpha, graine)
#
# Réglages communs (dataset, architecture, K, loss, stratégie, epochs, lr...) :
# configs/alpha_sweep.json. Ce script ne fixe que alpha et la graine de chaque job.
#
# Grille : 6 alphas x 3 graines = 18 jobs  ->  --array=0-17
# (si tu changes ALPHAS ou SEEDS : --array=0-(N_ALPHAS*N_SEEDS - 1))
#
# Lancement, depuis le dossier du code (dml_v2) :
#   mkdir -p Logs
#   sbatch --export=ALL,CAMPAIGN=alpha_sweep_cifar10 cluster/soumission_alpha_sweep.sh
#
# Sorties : results/$CAMPAIGN/jobs/a<alpha>_s<graine>/{config.json,results.json,probs/}
# Puis : python merge_sweep.py results/$CAMPAIGN   (rassemble tout pour le notebook)
#
# Relancer un job qui a planté (même CAMPAIGN) : il reprend là où il s'était arrêté.
#   sbatch --array=7 --export=ALL,CAMPAIGN=alpha_sweep_cifar10 cluster/soumission_alpha_sweep.sh
# =============================================================================

CONFIG="configs/alpha_sweep.json"
ALPHAS=(0 0.5 1 10 50 100)
SEEDS=(0 1 2)
CAMPAIGN=${CAMPAIGN:-alpha_sweep_${SLURM_ARRAY_JOB_ID}}

N_ALPHAS=${#ALPHAS[@]}
N_SEEDS=${#SEEDS[@]}
N_TOTAL=$((N_ALPHAS * N_SEEDS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_TOTAL" ]; then
    echo "Index $SLURM_ARRAY_TASK_ID hors de la grille ($N_TOTAL jobs) : rien à faire."
    exit 0
fi

# Décodage de l'index de tâche
ALPHA=${ALPHAS[$((SLURM_ARRAY_TASK_ID % N_ALPHAS))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / N_ALPHAS))]}
RUN_DIR="results/${CAMPAIGN}/jobs/a${ALPHA}_s${SEED}"

# -----------------------------------------------------------------------------
# Environnement : Python du cluster + venv avec TensorFlow >= 2.16 (Keras 3).
# Ne pas charger le module TensorFlow/2.15 : il embarque Keras 2, incompatible.
# Installation du venv une seule fois : voir INSTALL.md, section 3.
# -----------------------------------------------------------------------------
module --force purge
ml releases/2023a
ml Python/3.11.3-GCCcore-12.3.0
source ~/venvs/dml-gpu/bin/activate

cd "$SLURM_SUBMIT_DIR"
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=2

echo "=========================================="
echo "SWEEP alpha | campagne $CAMPAIGN"
echo "Date: $(date)"
echo "Job: $SLURM_ARRAY_TASK_ID / $((N_TOTAL - 1))"
echo "Alpha: $ALPHA   Graine: $SEED"
echo "Config: $CONFIG"
echo "Sortie: $RUN_DIR"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "=========================================="

python run_alpha_sweep.py --config "$CONFIG" --alphas "$ALPHA" --seeds "$SEED" \
    --run-dir "$RUN_DIR" --verbose 2
exit_code=$?

echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: alpha=$ALPHA | graine=$SEED"
else
    echo "FAILED: alpha=$ALPHA | graine=$SEED (code $exit_code)"
fi
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
