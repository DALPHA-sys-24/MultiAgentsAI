# Installation des environnements

Python 3.10 à 3.13 (3.12 conseillé). Toutes les commandes se lancent depuis la racine du dépôt (le dossier qui contient `collaborative_learning/`).

## 1. CPU sous Windows (PowerShell)

```powershell
cd C:\Users\jketchak\Documents\DALPHAcommunity\UNAMUR\EUREKA\btwg_ml\MultiAgentsAI
py -3.12 -m venv .venv-cpu
.\.venv-cpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_cpu.txt
python -m experiments.collaboration --cpu --train-subset 2000 --epochs 2 --warmup-epochs 1 --alphas 1 --n-runs 1
```

Si PowerShell refuse d'activer l'environnement : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, une seule fois.
En invite de commandes (cmd) : `.venv-cpu\Scripts\activate.bat`.

## 2. GPU sur ta machine (RTX 4060) via WSL2

TensorFlow n'utilise pas le GPU sous Windows natif : il faut WSL2 (Ubuntu). Le pilote NVIDIA Windows déjà installé suffit, n'installe pas de pilote ni de CUDA dans Ubuntu.

Dans PowerShell **administrateur**, une seule fois (redémarrage demandé) :

```powershell
wsl --install -d Ubuntu-24.04
```

Puis dans le terminal Ubuntu :

```bash
nvidia-smi                                   # doit afficher la RTX 4060
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/venvs/collab-gpu              # environnement dans Linux (plus rapide que sur /mnt/c)
source ~/venvs/collab-gpu/bin/activate
pip install --upgrade pip

cd /mnt/c/Users/jketchak/Documents/DALPHAcommunity/UNAMUR/EUREKA/btwg_ml/MultiAgentsAI
pip install -r requirements_gpu.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -m experiments.collaboration --train-subset 2000 --epochs 2 --warmup-epochs 1 --alphas 1 --n-runs 1
```

La ligne `[device]` au démarrage doit afficher le GPU. À chaque nouvelle session : `source ~/venvs/collab-gpu/bin/activate`.

## 3. GPU sur le cluster (Linux)

Sur Lyra, pas d'environnement virtuel : `slurm/campaign.sh` charge les modules listés dans `slurm/modules.sh` (TensorFlow 2.15, matplotlib, Optuna, PyYAML). Vérifier une fois depuis le nœud de connexion :

```bash
source slurm/modules.sh && python -m experiments.campaign quick_test --list
```

Si PyYAML manque, le script le dit : ajouter le bon module (`module spider PyYAML`) dans `slurm/modules.sh`, ou `pip install --user pyyaml`.

Ailleurs, avec un environnement virtuel :

```bash
module load Python/3.12          # nom exact selon le cluster : module avail python
python3 -m venv ~/venvs/collab-gpu
source ~/venvs/collab-gpu/bin/activate
pip install --upgrade pip
pip install -r requirements_gpu.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # sur un nœud GPU
```

Les nœuds de calcul n'ont souvent pas internet : télécharger les données une fois depuis le nœud de connexion :

```bash
python -c "from collaborative_learning.data import load_dataset; load_dataset('cifar10'); load_dataset('fashion_mnist')"
```

## Avec conda (au lieu de venv)

```bash
conda create -n collab-gpu python=3.12 -y
conda activate collab-gpu
pip install -r requirements_gpu.txt      # ou requirements_cpu.txt
```
