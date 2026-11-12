#!/bin/bash
#SBATCH --job-name=collab_vs_big
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-23:59:00
#SBATCH --output=Logs/collab_vs_big_%j.out
#SBATCH --error=Logs/collab_vs_big_%j.err

# =============================================================================
# K petits agents collaboratifs vs 1 grand agent (même nombre de paramètres)
# Réglages : configs/collab_vs_big.json (toutes les graines dans ce job).
#
# Lancement, depuis le dossier du code (dml_v2) :
#   sbatch soumission_collab_vs_big.sh
# Sortie : results/<date>_collab_vs_big/ (results.json réécrit après chaque graine)
# =============================================================================

# Lancé avec « bash fichier.sh » (hors SLURM) : crée Logs/ puis se soumet lui-même.
# SLURM ouvre les fichiers --output/--error avant le script : Logs/ doit exister avant sbatch.
if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p Logs
    exec sbatch "$0" "$@"
fi
mkdir -p Logs results

CONFIG="configs/collab_vs_big.json"

# -----------------------------------------------------------------------------
# Environnement : exactement les modules de soumission_experiment_collab.sh (GPU testé)
# -----------------------------------------------------------------------------
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
ml tqdm

cd "$SLURM_SUBMIT_DIR"
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "=========================================="
echo "COLLAB vs BIG | $(date) | config $CONFIG"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "=========================================="


python run_experiment.py --config "$CONFIG" --verbose 2
exit_code=$?

echo "=========================================="
[ $exit_code -eq 0 ] && echo "SUCCESS" || echo "FAILED (code $exit_code)"
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
