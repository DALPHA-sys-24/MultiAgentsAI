"""
Données.

    datasets.py   chargement et normalisation, registre @register_dataset
    pipelines.py  pipelines tf.data (mélange, augmentation, batchs)
"""
from dml.data.datasets import available_datasets, load_dataset, register_dataset
from dml.data.pipelines import make_eval_dataset, make_train_dataset

__all__ = ["available_datasets", "load_dataset", "register_dataset", "make_eval_dataset", "make_train_dataset"]
