#!/bin/bash
# Fonctions communes aux scripts sbatch de slurm/ (chargé avec « source slurm/helpers.sh »).

export TF_GPU_ALLOCATOR=cuda_malloc_async     # moins de fragmentation de la mémoire GPU
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1                     # sorties .out écrites au fil de l'eau

# submit_merge CAMPAGNE : la tâche 0 du job array soumet la fusion de la campagne, qui démarre
# quand tout le job array est fini (results.json de chaque dossier, puis figures).
# AUTO_MERGE=0 pour ne pas le faire.
submit_merge() {
    local campaign=$1
    [ "${SLURM_ARRAY_TASK_ID:-0}" -eq "${SLURM_ARRAY_TASK_MIN:-0}" ] && [ "${AUTO_MERGE:-1}" != 0 ] || return 0
    local name="merge_$(basename "$campaign" .yaml)"
    sbatch --dependency="afterany:$SLURM_ARRAY_JOB_ID" --job-name="$name" \
           --output="slurm_logs/${name}_%j.out" --error="slurm_logs/${name}_%j.err" \
           slurm/merge_campaign.sh "$campaign" \
        || echo "Fusion non soumise : lancer « sbatch slurm/merge_campaign.sh $campaign » après la fin des tâches."
}
