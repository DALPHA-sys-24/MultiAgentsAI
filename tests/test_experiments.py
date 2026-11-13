"""Découpage en tâches, exécution d'une seule tâche (--task-index), fusion, protection du dossier."""
import json
import os

import pytest

import experiments.common as common
from experiments.alpha_sweep import AlphaSweep
from experiments.collab_vs_big import CollabVsBig

ARGS = ["--epochs", "1", "--batch-size", "16", "--num-agents", "2", "--warmup-epochs", "0",
        "--augment", "false", "--cpu", "--verbose", "0"]


@pytest.fixture
def fake_data(monkeypatch, tiny_data):
    monkeypatch.setattr(common, "load_dataset", lambda *a, **k: dict(tiny_data))
    monkeypatch.setattr(common, "download_dataset", lambda *a, **k: None)


def test_list_tasks(capsys, tmp_path, fake_data):
    CollabVsBig().main(ARGS + ["--n-runs", "3", "--run-dir", str(tmp_path), "--list-tasks"])
    assert capsys.readouterr().out.strip() == f"TASKS 6 {tmp_path}"


def test_tasks_then_merge(tmp_path, fake_data):
    run_dir = str(tmp_path / "run")
    args = ARGS + ["--n-runs", "1", "--run-dir", run_dir]
    CollabVsBig().main(args + ["--task-index", "0"])          # petits agents de la graine 0
    assert os.path.exists(os.path.join(run_dir, "parts", "small_r0", "result.json"))
    assert not os.path.exists(os.path.join(run_dir, "results.json"))   # run pas encore complet
    CollabVsBig().main(args + ["--task-index", "1"])          # grand agent de la graine 0
    CollabVsBig().main(args + ["--merge-only"])
    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    assert res["n_runs_done"] == 1 and res["runs"][0]["seed"] == 0
    assert os.path.exists(os.path.join(run_dir, "probs", "small_agents_s0.npy"))
    assert os.path.exists(os.path.join(run_dir, "probs", "big_agent_s0.npy"))


def test_alpha_sweep_tasks(tmp_path, fake_data):
    run_dir = str(tmp_path / "sweep")
    AlphaSweep().main(ARGS + ["--n-runs", "1", "--alphas", "0", "1", "--run-dir", run_dir])
    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    assert sorted(os.listdir(os.path.join(run_dir, "parts"))) == ["a0_r0", "a1_r0"]
    assert len(res["runs"]) == 2


def test_other_settings_refuse_same_folder(tmp_path, fake_data):
    run_dir = str(tmp_path / "run")
    CollabVsBig().main(ARGS + ["--n-runs", "1", "--run-dir", run_dir, "--task-index", "1"])
    CollabVsBig().main(ARGS + ["--n-runs", "2", "--run-dir", run_dir, "--list-tasks"])   # n_runs extensible
    with pytest.raises(SystemExit, match="--tag"):
        CollabVsBig().main(ARGS + ["--lr", "0.5", "--run-dir", run_dir, "--list-tasks"])


def test_underscore_flags_accepted():
    args = CollabVsBig().parse_args(["--n_runs", "3", "--batch_size=64", "--task_index", "2"])
    assert (args.n_runs, args.batch_size, args.task_index) == (3, 64, 2)
