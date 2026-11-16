#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --partition=batch
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=slurm_logs/merge_%j.out
#SBATCH --error=slurm_logs/merge_%j.err

# =============================================================================
# FUSION D'UNE CAMPAGNE (sans GPU) : results.json de chaque dossier, puis figures et summary.csv.
# Soumis automatiquement par la tâche 0 de slurm/campaign.sh. À la main, depuis la racine du dépôt :
#   sbatch slurm/merge_campaign.sh lr_sweep        (ou directement : python -m experiments.campaign lr_sweep --merge)
# =============================================================================
CAMPAIGN="$1"
[ -n "$CAMPAIGN" ] || { echo "Usage : sbatch slurm/merge_campaign.sh <campagne>"; exit 1; }
source slurm/modules.sh || exit 1
source slurm/helpers.sh
echo "FUSION $CAMPAIGN | $(date)"
python -m experiments.campaign "$CAMPAIGN" --merge --cpu
