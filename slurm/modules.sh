#!/bin/bash
# Modules de Lyra (sans environnement virtuel), chargés par campaign.sh et merge_campaign.sh.
# --ignore_cache : ne pas se fier au cache de Lmod (un cache périmé rend des modules « unknown »).
# Si un module manque : module spider <nom> pour trouver la bonne version, puis corriger la ligne ici.
module --force purge
_load() {
    module --ignore_cache load "$1" || {
        echo "ERREUR : module $1 introuvable (module spider ${1%%/*} pour voir les versions disponibles)"
        return 1
    }
}
_load releases/2023a                            || return 1 2>/dev/null || exit 1
_load tis/2025                                  || return 1 2>/dev/null || exit 1
_load OpenSSL/1.1                               || return 1 2>/dev/null || exit 1   # requis par TensorFlow
_load CUDA/12.1.1                               || return 1 2>/dev/null || exit 1
_load Python/3.11.3-GCCcore-12.3.0              || return 1 2>/dev/null || exit 1
_load SciPy-bundle/2023.07-gfbf-2023a           || return 1 2>/dev/null || exit 1
_load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1  || return 1 2>/dev/null || exit 1
_load matplotlib/3.7.2                          || return 1 2>/dev/null || exit 1
_load Seaborn/0.13.2-gfbf-2023a                 || return 1 2>/dev/null || exit 1
_load scikit-learn/1.3.1-gfbf-2023a             || return 1 2>/dev/null || exit 1
_load Optuna/3.5.0-foss-2023a                   || return 1 2>/dev/null || exit 1
python -c "import yaml" 2>/dev/null || module --ignore_cache load PyYAML/6.0-GCCcore-12.3.0 2>/dev/null
python -c "import yaml" 2>/dev/null || {
    echo "PyYAML introuvable : chercher le module (module spider PyYAML) et l'ajouter dans slurm/modules.sh,"
    echo "ou l'installer une fois : pip install --user pyyaml"
    return 1 2>/dev/null || exit 1
}
python -c "import tensorflow" 2>/dev/null || {
    echo "ERREUR : TensorFlow ne s'importe pas après le chargement des modules"
    return 1 2>/dev/null || exit 1
}
