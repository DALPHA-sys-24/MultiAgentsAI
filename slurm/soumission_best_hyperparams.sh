#!/bin/bash
#SBATCH --job-name=best_hparams
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --array=0-3
#SBATCH --output=Logs/best_hparams_%A_%a.out
#SBATCH --error=Logs/best_hparams_%A_%a.err

# =============================================================================
# RECHERCHE D'HYPERPARAMÈTRES (Optuna)
#
# Réglages (jeu de données, architecture, K, espace de recherche, n_trials...) :
# configs/best_hyperparams.json.
#
# --array=0-3 lance 4 jobs qui travaillent EN MÊME TEMPS sur la même recherche
# (même dossier results/$CAMPAIGN) : chacun fait n_trials essais, ils s'additionnent.
# Plus de jobs = recherche plus rapide (--array=0-7 pour 8 GPU).
#
# Une seule fois, sur le nœud de connexion, après les « ml » ci-dessous :
#   pip install --user optuna
#
# Lancement, depuis la racine du dépôt (là où sont dml/, configs/, slurm/) :
#   sbatch slurm/soumission_best_hyperparams.sh
# Résultats : results/$CAMPAIGN/{best.json, best_config.json, trials.csv}
# Relancer la même commande complète la recherche avec de nouveaux essais.
# Nouvelle recherche (sans mélanger avec l'ancienne) : changer CAMPAIGN ci-dessous.
# =============================================================================

# Lancé avec « bash fichier.sh » (hors SLURM) : crée Logs/ puis se soumet lui-même.
# SLURM ouvre les fichiers --output/--error avant le script : Logs/ doit exister avant sbatch.
if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p Logs
    exec sbatch "$0" "$@"
fi
mkdir -p Logs results

CONFIG="configs/best_hyperparams.json"
CAMPAIGN="best_hyperparams"       # dossier results/$CAMPAIGN
RUN_DIR="results/${CAMPAIGN}"

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

# Racine du dépôt : dossier de lancement de sbatch (ou son parent si lancé depuis slurm/)
ROOT="$SLURM_SUBMIT_DIR"
[ -d "$ROOT/dml" ] || ROOT="$(dirname "$ROOT")"
[ -d "$ROOT/dml" ] || { echo "Lancer sbatch depuis la racine du dépôt (dossier qui contient dml/)"; exit 1; }
cd "$ROOT"
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "=========================================="
echo "BEST HYPERPARAMS | $(date) | job ${SLURM_ARRAY_TASK_ID:-0} | config $CONFIG | sortie $RUN_DIR"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "=========================================="

# Décalage de départ : les jobs n'écrivent pas tous dans l'étude au même instant
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 20 ))

python -m experiments.best_hyperparams --config "$CONFIG" --run-dir "$RUN_DIR" \
    --study-seed "${SLURM_ARRAY_TASK_ID:-0}"
exit_code=$?

echo "=========================================="
[ $exit_code -eq 0 ] && echo "SUCCESS" || echo "FAILED (code $exit_code)"
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
