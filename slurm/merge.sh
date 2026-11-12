#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --partition=batch
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=Logs/merge_%j.out
#SBATCH --error=Logs/merge_%j.err

# =============================================================================
# FUSION : rassemble les tâches terminées dans results.json (sans GPU, quelques secondes).
# Soumis automatiquement par la 1re tâche de chaque dataset. À la main, depuis la racine du dépôt :
#   sbatch slurm/merge.sh collab_vs_big --dataset fashion_mnist --num-agents 3 ...   (mêmes options)
# ou directement sur le nœud de connexion :
#   python -m experiments.collab_vs_big --dataset fashion_mnist --num-agents 3 ... --merge-only
# =============================================================================
EXP="$1"; shift

# Modules de Lyra (sans environnement virtuel)
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

source slurm/common.sh
echo "FUSION $EXP | $(date) | options : $*"
python -m experiments."$EXP" "$@" --merge-only --cpu
