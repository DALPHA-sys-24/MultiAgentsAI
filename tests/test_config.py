"""Config : clés inconnues refusées, listes par agent vérifiées, ligne de commande."""
import argparse
import json

import pytest

from collaborative_learning.utils.config import (CollaborationConfig, ReferenceConfig, add_config_arguments, load_config,
                                                 overrides_from_args)
from collaborative_learning.utils.config_files import read_config_file


def test_shipped_configs_load():
    cfg = load_config(CollaborationConfig, "configs/collaboration.yaml")
    assert cfg.method == "global_interaction" and cfg.dataset == "cifar10" and cfg.epochs == 200
    ref = load_config(ReferenceConfig, "configs/reference.yaml")
    assert ref.loss == "ce"


def test_yaml_defaults_are_composed(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("epochs: 7\nlr: 0.1\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"defaults: [{base}, methods/dml]\nlr: 0.3\n")
    values = read_config_file(str(child))
    assert (values["epochs"], values["lr"], values["loss_type"], values["loss_form"]) == (7, 0.3, "dml", "additive")
    assert "defaults" not in values
    loop = tmp_path / "loop.yaml"
    loop.write_text(f"defaults: [{loop}]\n")
    with pytest.raises(SystemExit, match="Boucle"):
        read_config_file(str(loop))


def test_old_json_keys_still_accepted(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"epochs": 3, "big_model": True, "big_lr": 0.1}))   # anciens config.json
    assert load_config(CollaborationConfig, str(path)).epochs == 3


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


def test_command_line_overrides():
    p = argparse.ArgumentParser()
    add_config_arguments(p, CollaborationConfig)
    args = p.parse_args(["--alphas", "0", "10", "--n-runs", "2", "--lr", "0.05", "--augment", "false",
                         "--optimizer-kwargs", '{"global_clipnorm": 1.0}', "--jit", "false"])
    cfg = load_config(CollaborationConfig, "configs/collaboration.yaml", overrides_from_args(args, CollaborationConfig))
    assert cfg.alphas == [0.0, 10.0] and cfg.seeds == [0, 1] and cfg.lr == 0.05
    assert cfg.augment is False and cfg.optimizer_kwargs == {"global_clipnorm": 1.0} and cfg.jit is False


def test_run_dir_name():
    cfg = load_config(CollaborationConfig, overrides={"dataset": "fashion_mnist", "num_agents": 3, "arch": "resnet20"})
    assert cfg.run_dir().replace("\\", "/") == \
        "results/collaboration/fashion_mnist/resnet20_3agents/global_interaction_multiplicative"
    named = load_config(CollaborationConfig, overrides={"dataset": "cifar10", "num_agents": 5, "method": "dml",
                                                        "loss_type": "dml", "loss_form": "additive", "tag": "adamw"})
    assert named.run_name() == "collaboration/cifar10/resnet20_5agents/dml_adamw"
    ref = load_config(ReferenceConfig, overrides={"dataset": "cifar10", "num_agents": 3, "lr": 0.01, "tag": "tuned"})
    assert ref.run_name() == "reference/cifar10/resnet20_x3params_sgd_lr0.01_tuned"


def test_signature_ignores_extensible_settings():
    a = load_config(CollaborationConfig, overrides={"n_runs": 3})
    b = load_config(CollaborationConfig, overrides={"n_runs": 10, "steps_per_execution": 1, "alphas": [0, 1]})
    c = load_config(CollaborationConfig, overrides={"lr": 0.5})
    assert a.signature() == b.signature() != c.signature()
