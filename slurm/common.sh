#!/bin/bash
# Fonctions communes aux scripts sbatch de slurm/ (chargé avec « source slurm/common.sh »).

export TF_GPU_ALLOCATOR=cuda_malloc_async     # moins de fragmentation de la mémoire GPU
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1                     # sorties .out écrites au fil de l'eau

# check_array_size N : prévient si --array ne couvre pas exactement les N tâches
check_array_size() {
    local expected=$1
    if [ -n "$SLURM_ARRAY_TASK_MAX" ] && [ "$SLURM_ARRAY_TASK_ID" = "${SLURM_ARRAY_TASK_MIN:-0}" ] \
       && [ $((SLURM_ARRAY_TASK_MAX + 1)) -lt "$expected" ]; then
        echo "ATTENTION : --array s'arrête à $SLURM_ARRAY_TASK_MAX mais il y a $expected tâches"
        echo "            (mettre --array=0-$((expected - 1)) en tête du script)."
    fi
}

# submit_merge EXP TASK_IDX ARGS... : la tâche 0 de chaque dataset soumet le job de fusion,
# qui démarre quand tout le job array est fini et écrit results.json. AUTO_MERGE=0 pour ne pas le faire.
submit_merge() {
    local exp=$1 idx=$2
    shift 2
    [ "$idx" -eq 0 ] && [ "${AUTO_MERGE:-1}" != 0 ] || return 0
    sbatch --dependency="afterany:$SLURM_ARRAY_JOB_ID" --job-name="${exp}_merge" \
           --output="Logs/${exp}_merge_%j.out" --error="Logs/${exp}_merge_%j.err" \
           slurm/merge.sh "$exp" "$@" \
        || echo "Job de fusion non soumis : lancer « sbatch slurm/merge.sh $exp $* » après la fin des tâches."
}
