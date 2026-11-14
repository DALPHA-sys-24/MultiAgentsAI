"""Config : clés inconnues refusées, listes par agent vérifiées, ligne de commande."""
import argparse
import json

import pytest

from collaborative_learning.utils.config import CollaborationConfig, add_config_arguments, load_config, overrides_from_args


def test_shipped_configs_load():
    load_config(CollaborationConfig, "configs/collaboration.json")


def test_unknown_key_is_refused_with_suggestion(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"epoch": 3}))
    with pytest.raises(SystemExit, match="epochs"):
        load_config(CollaborationConfig, str(path))


def test_comment_keys_are_ignored(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"_note": "essai", "epochs": 3}))
    assert load_config(CollaborationConfig, str(path)).epochs == 3


def test_per_agent_list_length_checked():
    with pytest.raises(SystemExit, match="lr"):
        load_config(CollaborationConfig, overrides={"num_agents": 3, "lr": [0.1, 0.2]})


def test_bad_choice_refused():
    with pytest.raises(SystemExit, match="loss_type"):
        load_config(CollaborationConfig, overrides={"loss_type": "inconnue"})


def test_big_model_needs_its_own_lr_when_lr_is_a_list():
    load_config(CollaborationConfig, overrides={"num_agents": 2, "lr": [0.1, 0.2]})   # sans grand modèle : ok
    with pytest.raises(SystemExit, match="big_lr"):
        load_config(CollaborationConfig, overrides={"num_agents": 2, "lr": [0.1, 0.2], "big_model": True})


def test_command_line_overrides():
    p = argparse.ArgumentParser()
    add_config_arguments(p, CollaborationConfig)
    args = p.parse_args(["--alphas", "0", "10", "--n-runs", "2", "--lr", "0.05", "--augment", "false",
                         "--optimizer-kwargs", '{"global_clipnorm": 1.0}', "--jit", "false"])
    cfg = load_config(CollaborationConfig, "configs/collaboration.json", overrides_from_args(args, CollaborationConfig))
    assert cfg.alphas == [0.0, 10.0] and cfg.seeds == [0, 1] and cfg.lr == 0.05
    assert cfg.augment is False and cfg.optimizer_kwargs == {"global_clipnorm": 1.0} and cfg.jit is False


def test_run_dir_name():
    cfg = load_config(CollaborationConfig, overrides={"dataset": "fashion_mnist", "num_agents": 3, "arch": "resnet20"})
    assert cfg.run_dir().replace("\\", "/") == "results/collaboration/fashion_mnist/resnet20_3agents"
    tagged = load_config(CollaborationConfig, overrides={"dataset": "cifar10", "num_agents": 5, "tag": "adamw"})
    assert tagged.run_name() == "collaboration/cifar10/resnet20_5agents_adamw"


def test_signature_ignores_extensible_settings():
    a = load_config(CollaborationConfig, overrides={"n_runs": 3})
    b = load_config(CollaborationConfig, overrides={"n_runs": 10, "steps_per_execution": 1, "alphas": [0, 1],
                                                    "big_model": True})
    c = load_config(CollaborationConfig, overrides={"lr": 0.5})
    assert a.signature() == b.signature() != c.signature()
