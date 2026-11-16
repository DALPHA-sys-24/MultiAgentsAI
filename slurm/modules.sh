#!/bin/bash
# Modules de Lyra (sans environnement virtuel), chargés par campaign.sh et merge_campaign.sh.
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
ml Optuna/3.5.0-foss-2023a
python -c "import yaml" 2>/dev/null || ml PyYAML/6.0-GCCcore-12.3.0 2>/dev/null
python -c "import yaml" 2>/dev/null || {
    echo "PyYAML introuvable : chercher le module (module spider PyYAML) et l'ajouter dans slurm/modules.sh,"
    echo "ou l'installer une fois : pip install --user pyyaml"
    return 1 2>/dev/null || exit 1
}
