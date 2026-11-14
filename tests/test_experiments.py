"""Découpage en tâches, exécution d'une seule tâche (--task-index), fusion, protection du dossier."""
import json
import os

import pytest

import experiments.common as common
from experiments.collaboration import Collaboration

ARGS = ["--epochs", "1", "--batch-size", "16", "--num-agents", "2", "--warmup-epochs", "0",
        "--augment", "false", "--big-model", "false", "--cpu", "--verbose", "0"]   # options ajoutées après : prioritaires


@pytest.fixture
def fake_data(monkeypatch, tiny_data):
    monkeypatch.setattr(common, "load_dataset", lambda *a, **k: dict(tiny_data))
    monkeypatch.setattr(common, "download_dataset", lambda *a, **k: None)


def test_list_tasks(capsys, tmp_path, fake_data):
    Collaboration().main(ARGS + ["--n-runs", "3", "--alphas", "1", "--run-dir", str(tmp_path), "--list-tasks"])
    assert capsys.readouterr().out.strip() == f"TASKS 3 {tmp_path}"
    Collaboration().main(ARGS + ["--n-runs", "3", "--alphas", "0", "1", "--big-model", "true",
                                 "--run-dir", str(tmp_path), "--list-tasks"])
    assert capsys.readouterr().out.strip() == f"TASKS 9 {tmp_path}"
    ids = [t.id for t in Collaboration().tasks(Collaboration.CONFIG_CLS(n_runs=2, alphas=[0, 1], big_model=True))]
    assert ids == ["alpha0_seed0", "alpha1_seed0", "big_model_seed0", "alpha0_seed1", "alpha1_seed1", "big_model_seed1"]


def test_tasks_then_merge_with_big_model(tmp_path, fake_data):
    run_dir = str(tmp_path / "run")
    args = ARGS + ["--n-runs", "1", "--alphas", "1", "--big-model", "true", "--run-dir", run_dir]
    Collaboration().main(args + ["--task-index", "0"])          # agents de la graine 0, alpha = 1
    assert os.path.exists(os.path.join(run_dir, "tasks", "alpha1_seed0", "result.json"))
    with open(os.path.join(run_dir, "results.json")) as f:
        assert "big_model" not in json.load(f)                   # grand modèle pas encore fini
    Collaboration().main(args + ["--task-index", "1"])          # grand modèle de la graine 0
    Collaboration().main(args + ["--merge-only"])
    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    assert [r["seed"] for r in res["runs"]] == [0] and res["big_model"]["runs"][0]["seed"] == 0
    assert "Grand modèle seul" in res["vs_big_model"]["1"] and res["model_sizes"]["big_width"] > 1
    assert sorted(os.listdir(os.path.join(run_dir, "test_probabilities"))) == ["alpha1_seed0.npy", "big_model_seed0.npy"]


def test_alpha_sweep_without_big_model(tmp_path, fake_data):
    run_dir = str(tmp_path / "sweep")
    Collaboration().main(ARGS + ["--n-runs", "1", "--alphas", "0", "1", "--run-dir", run_dir])
    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    assert sorted(os.listdir(os.path.join(run_dir, "tasks"))) == ["alpha0_seed0", "alpha1_seed0"]
    assert len(res["runs"]) == 2 and set(res["summary"]) == {"0", "1"} and "big_model" not in res
    assert res["summary"]["0"]["strategy"] == "majority"
    assert sorted(os.listdir(os.path.join(run_dir, "test_probabilities"))) == ["alpha0_seed0.npy", "alpha1_seed0.npy"]


def test_big_model_and_alphas_can_be_added_later(tmp_path, fake_data):
    run_dir = str(tmp_path / "run")
    Collaboration().main(ARGS + ["--n-runs", "1", "--alphas", "1", "--run-dir", run_dir])
    Collaboration().main(ARGS + ["--n-runs", "1", "--alphas", "1", "--big-model", "true", "--run-dir", run_dir])
    Collaboration().main(ARGS + ["--n-runs", "1", "--alphas", "1", "2", "--run-dir", run_dir])   # sans grand modèle
    with open(os.path.join(run_dir, "config.json")) as f:
        saved = json.load(f)
    assert saved["big_model"] is True and "model_sizes" in saved   # gardés par un lancement sans grand modèle
    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    assert set(res["vs_big_model"]) == {"1", "2"} and saved["alphas"] == [1.0, 2.0]


def test_other_settings_refuse_same_folder(tmp_path, fake_data):
    run_dir = str(tmp_path / "run")
    Collaboration().main(ARGS + ["--n-runs", "1", "--alphas", "1", "--run-dir", run_dir, "--task-index", "0"])
    Collaboration().main(ARGS + ["--n-runs", "2", "--alphas", "0", "1", "--run-dir", run_dir, "--list-tasks"])
    with pytest.raises(SystemExit, match="--tag"):
        Collaboration().main(ARGS + ["--lr", "0.5", "--run-dir", run_dir, "--list-tasks"])


def test_underscore_flags_accepted():
    args = Collaboration().parse_args(["--n_runs", "3", "--batch_size=64", "--task_index", "2", "--big_model", "true"])
    assert (args.n_runs, args.batch_size, args.task_index, args.big_model) == (3, 64, 2, True)
