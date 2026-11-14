"""Recherche d'hyperparamètres : espace de recherche, config, petite étude Optuna."""
import json

import pytest

from conftest import TINY
from collaborative_learning.tuning.space import decode_params, validate_space
from collaborative_learning.utils.config import HyperparamSearchConfig, load_config

optuna = pytest.importorskip("optuna")


def test_shipped_config_loads():
    cfg = load_config(HyperparamSearchConfig, "configs/hyperparameter_search.json")
    assert cfg.direction == "maximize"


def test_space_errors_are_reported():
    errors = validate_space({"learning_rate": {"type": "float", "low": 0, "high": 1},
                             "lr": {"type": "float", "low": 0.0, "high": 1.0, "log": True},
                             "optimizer": {"type": "categorical"}}, ["lr", "optimizer"])
    assert len(errors) == 3


def test_unknown_search_key_refused():
    with pytest.raises(SystemExit, match="learning_rate"):
        load_config(HyperparamSearchConfig, overrides={"search_space": {
            "learning_rate": {"type": "float", "low": 1e-3, "high": 1e-1}}})


def test_suggest_and_decode_dict_choices():
    from collaborative_learning.tuning.space import suggest
    space = {"lr": {"type": "float", "low": 1e-3, "high": 1e-1, "log": True},
             "optimizer_kwargs": {"type": "categorical", "choices": [None, {"global_clipnorm": 1.0}]}}
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    seen = set()
    for _ in range(10):
        trial = study.ask()
        params = suggest(trial, space)
        assert 1e-3 <= params["lr"] <= 1e-1
        seen.add(json.dumps(params["optimizer_kwargs"]))
        study.tell(trial, 0.0)
    assert seen == {"null", '{"global_clipnorm": 1.0}'}
    assert decode_params({"optimizer_kwargs": '{"global_clipnorm": 1.0}'}) == {"optimizer_kwargs": {"global_clipnorm": 1.0}}


def test_tiny_study_runs_and_exports(tmp_path, tiny_data):
    from collaborative_learning.tuning.study import create_study, export_results
    from experiments.hyperparameter_search import make_objective

    cfg = load_config(HyperparamSearchConfig, overrides={
        "arch": TINY["archs"], "epochs": 2, "batch_size": 16, "warmup_epochs": 0,
        "augment": False, "jit": False, "pruner": "none",
        "search_space": {"lr": {"type": "float", "low": 1e-3, "high": 1e-1, "log": True}}})
    study = create_study(str(tmp_path), "t", cfg.direction, "random", cfg.pruner, 0, 1, cfg.epochs)
    study.optimize(make_objective(tiny_data, cfg, str(tmp_path), 0), n_trials=2)
    summary = export_results(study, str(tmp_path), cfg.to_dict(), decode_params)
    assert summary["n_trials"] == 2 and "lr" in summary["best_params"]
    assert (tmp_path / "trials.csv").exists() and (tmp_path / "best.json").exists()
    # une 2e étude sur le même dossier reprend les essais
    again = create_study(str(tmp_path), "t", cfg.direction, "random", cfg.pruner, 1, 1, cfg.epochs)
    assert len(again.trials) == 2


def test_best_params_build_multi_agent_config(tmp_path):
    from collaborative_learning.utils.config import CollaborationConfig, config_from_dict
    from experiments.hyperparameter_search import multi_agent_config

    cfg = load_config(HyperparamSearchConfig, overrides={"dataset": "fashion_mnist", "arch": "resnet32"})
    best = {"lr": 0.004, "optimizer": "adamw", "weight_decay": 1e-4}
    tuned = config_from_dict(CollaborationConfig, multi_agent_config(cfg, best))
    assert (tuned.arch, tuned.dataset, tuned.lr, tuned.optimizer, tuned.weight_decay) == \
        ("resnet32", "fashion_mnist", 0.004, "adamw", 1e-4)
    tuned3 = load_config(CollaborationConfig, overrides={**multi_agent_config(cfg, best), "num_agents": 3})
    assert tuned3.run_name() == "collaboration/fashion_mnist/resnet32_3agents_tuned"


def test_search_is_on_one_model():
    cfg = load_config(HyperparamSearchConfig, "configs/hyperparameter_search.json")
    assert not hasattr(cfg, "alpha") and cfg.num_agents == 1
    assert cfg.run_name() == "hyperparameter_search/fashion_mnist/resnet20"
    with pytest.raises(SystemExit, match="alpha"):
        load_config(HyperparamSearchConfig, overrides={"search_space": {"alpha": {"type": "float", "low": 0, "high": 1}}})
