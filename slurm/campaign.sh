#!/bin/bash
#SBATCH --job-name=campaign
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/campaign_%A_%a.out
#SBATCH --error=slurm_logs/campaign_%A_%a.err

# =============================================================================
# CAMPAGNE : une tâche du job array = une tâche de la campagne (configs/campaigns/<nom>.yaml).
# Tout ce qui varie (datasets, méthodes, alpha, lr, K...) est dans le fichier de campagne.
#
# Depuis la racine du dépôt, sur le nœud de connexion :
#   python -m experiments.campaign lr_sweep --list      # nombre de tâches N, dossiers, données téléchargées
#   sbatch --array=0-<N-1> slurm/campaign.sh lr_sweep   # la commande exacte est affichée par --list
#
# La tâche 0 soumet la fusion (slurm/merge_campaign.sh), qui démarre quand tout le job array est
# fini : results.json de chaque dossier, puis figures dans results/<campagne>/<dataset>/<arch>_<K>agents/figures/.
# Relancer la même commande ne refait que les tâches non terminées ; une tâche coupée reprend
# à sa dernière époque. Une seule tâche : sbatch --array=7 slurm/campaign.sh lr_sweep
# =============================================================================
CAMPAIGN="$1"
VERBOSE=2

if [ -z "$CAMPAIGN" ]; then
    echo "Usage : sbatch --array=0-<N-1> slurm/campaign.sh <campagne>   (N : python -m experiments.campaign <campagne> --list)"
    exit 1
fi
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Pas de job array : lancer avec sbatch --array=0-<N-1> (N : python -m experiments.campaign $CAMPAIGN --list)"
    exit 1
fi
[ -d collaborative_learning ] || { echo "Lancer sbatch depuis la racine du dépôt (dossier qui contient collaborative_learning/)"; exit 1; }

source slurm/modules.sh || exit 1
source slurm/helpers.sh

echo "=========================================="
echo "CAMPAGNE $CAMPAIGN | tâche $SLURM_ARRAY_TASK_ID (job $SLURM_ARRAY_JOB_ID) | $(date)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "=========================================="

if [ "$SLURM_ARRAY_TASK_ID" -eq "${SLURM_ARRAY_TASK_MIN:-0}" ]; then
    python -m experiments.campaign "$CAMPAIGN" --check-array "${SLURM_ARRAY_TASK_MAX:-0}"
    submit_merge "$CAMPAIGN"
fi

python -m experiments.campaign "$CAMPAIGN" --task "$SLURM_ARRAY_TASK_ID" --verbose "$VERBOSE"
exit_code=$?

echo "=========================================="
[ $exit_code -eq 0 ] && echo "SUCCESS" || echo "FAILED (code $exit_code)"
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
