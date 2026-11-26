"""Découpage en tâches, exécution d'une seule tâche (--task-index), fusion, protection du dossier, campagnes."""
import json
import os

import pytest

import experiments.common as common
from experiments.collaboration import Collaboration
from experiments.reference import Reference

ARGS = ["--epochs", "1", "--batch-size", "16", "--num-agents", "2", "--warmup-epochs", "0",
        "--augment", "false", "--cpu", "--verbose", "0"]   # options ajoutées après : prioritaires
FAST = {"epochs": 1, "batch_size": 16, "num_agents": 2, "warmup_epochs": 0, "augment": False, "n_runs": 1}


@pytest.fixture
def fake_data(monkeypatch, tiny_data):
    monkeypatch.setattr(common, "load_dataset", lambda *a, **k: dict(tiny_data))
    monkeypatch.setattr(common, "download_dataset", lambda *a, **k: None)
    import collaborative_learning.data as data
    monkeypatch.setattr(data, "download_dataset", lambda *a, **k: None)


def read(path):
    with open(path) as f:
        return json.load(f)


def test_list_tasks(capsys, tmp_path, fake_data):
    Collaboration().main(ARGS + ["--n-runs", "3", "--alphas", "0", "1", "--run-dir", str(tmp_path), "--list-tasks"])
    assert capsys.readouterr().out.strip() == f"TASKS 6 {tmp_path}"
    ids = [t.id for t in Collaboration().tasks(Collaboration.CONFIG_CLS(n_runs=2, alphas=[0, 1]))]
    assert ids == ["alpha0_seed0", "alpha1_seed0", "alpha0_seed1", "alpha1_seed1"]


def test_tasks_then_merge(tmp_path, fake_data):
    run_dir = str(tmp_path / "sweep")
    args = ARGS + ["--n-runs", "1", "--alphas", "0", "1", "--run-dir", run_dir]
    Collaboration().main(args + ["--task-index", "1"])          # agents de la graine 0, alpha = 1
    assert os.listdir(os.path.join(run_dir, "tasks")) == ["alpha1_seed0"]
    assert set(read(os.path.join(run_dir, "results.json"))["summary"]) == {"1"}   # fusion partielle
    Collaboration().main(args)                                   # alpha = 0 manquant, puis fusion
    res = read(os.path.join(run_dir, "results.json"))
    assert len(res["runs"]) == 2 and set(res["summary"]) == {"0", "1"} and res["agent_params"] > 0
    assert res["summary"]["0"]["strategy"] == "majority"
    alpha0 = next(r for r in res["runs"] if r["alpha"] == 0)    # agents indépendants en CE classique
    alpha1 = next(r for r in res["runs"] if r["alpha"] == 1)
    assert (alpha0["loss_type"], alpha0["loss_form"], alpha0["strategy"]) == ("independent", "additive", "majority")
    assert (alpha1["loss_type"], alpha1["loss_form"]) == ("global_interaction", "multiplicative")
    assert sorted(os.listdir(os.path.join(run_dir, "test_probabilities"))) == ["alpha0_seed0.npy", "alpha1_seed0.npy"]


def test_reference_experiment(tmp_path, fake_data):
    run_dir = str(tmp_path / "ref")
    Reference().main(ARGS + ["--n-runs", "2", "--run-dir", run_dir, "--task-index", "0"])
    Reference().main(ARGS + ["--n-runs", "2", "--run-dir", run_dir])
    res = read(os.path.join(run_dir, "results.json"))
    assert [r["seed"] for r in res["runs"]] == [0, 1] and set(res["summary"]) == {"accuracy", "nll", "ece"}
    assert res["model_sizes"]["big_width"] > 1                  # 2x plus de paramètres qu'un agent
    assert sorted(os.listdir(os.path.join(run_dir, "test_probabilities"))) == ["seed0.npy", "seed1.npy"]


def test_alphas_and_runs_can_be_added_later(tmp_path, fake_data):
    run_dir = str(tmp_path / "run")
    Collaboration().main(ARGS + ["--n-runs", "1", "--alphas", "1", "--run-dir", run_dir])
    Collaboration().main(ARGS + ["--n-runs", "1", "--alphas", "1", "2", "--run-dir", run_dir])
    saved = read(os.path.join(run_dir, "config.json"))
    assert saved["alphas"] == [1.0, 2.0] and set(read(os.path.join(run_dir, "results.json"))["summary"]) == {"1", "2"}


def test_other_settings_refuse_same_folder(tmp_path, fake_data):
    run_dir = str(tmp_path / "run")
    Collaboration().main(ARGS + ["--n-runs", "1", "--alphas", "1", "--run-dir", run_dir, "--task-index", "0"])
    Collaboration().main(ARGS + ["--n-runs", "2", "--alphas", "0", "1", "--run-dir", run_dir, "--list-tasks"])
    with pytest.raises(SystemExit, match="--tag"):
        Collaboration().main(ARGS + ["--lr", "0.5", "--run-dir", run_dir, "--list-tasks"])


def test_merge_without_anything_does_nothing(tmp_path):
    Collaboration().main(["--run-dir", str(tmp_path / "vide"), "--merge-only", "--verbose", "0"])
    assert not os.path.exists(tmp_path / "vide" / "results.json")


def test_underscore_flags_accepted():
    args = Collaboration().parse_args(["--n_runs", "3", "--batch_size=64", "--task_index", "2", "--per_agent_views", "true"])
    assert (args.n_runs, args.batch_size, args.task_index, args.per_agent_views) == (3, 64, 2, True)


# --- campagnes ---------------------------------------------------------------------------------

def write_campaign(tmp_path, grid, **settings):
    import yaml
    spec = {"experiment": "collaboration", "datasets": ["fashion_mnist"], "methods": ["global_interaction", "dml"],
            "settings": {"arch": "resnet20", **FAST, "out_dir": str(tmp_path / "results"), **settings},
            "grid": grid, "reference": True}
    path = tmp_path / "essai.yaml"
    path.write_text(yaml.safe_dump(spec))
    return str(path)


def test_campaign_tasks_and_folders(tmp_path):
    from experiments.campaign import Campaign
    camp = Campaign(write_campaign(tmp_path, {"alpha": [0, 1], "lr": [0.1, 0.01]}, n_runs=2))
    runs = camp.runs()
    names = [os.path.relpath(r.run_dir, tmp_path / "results").replace("\\", "/") for r in runs]
    group = "essai/fashion_mnist/resnet20_2agents"
    assert names == [f"{group}/independent/alpha0_lr0.1", f"{group}/independent/alpha0_lr0.01",   # alpha = 0 : une fois
                     f"{group}/global_interaction/alpha1_lr0.1", f"{group}/global_interaction/alpha1_lr0.01",
                     f"{group}/dml/alpha1_lr0.1", f"{group}/dml/alpha1_lr0.01",
                     "reference/fashion_mnist/resnet20_x2params_sgd_lr0.01"]                  # lr de base
    assert len(camp.tasks()) == 7 * 2
    dml = runs[4].cfg
    assert (dml.loss_type, dml.loss_form, dml.lr, dml.alphas) == ("dml", "additive", 0.1, [1.0])
    assert camp.reference_for(runs[0]) is runs[-1]
    assert Campaign(camp.path, {"n_runs": 1}).tasks()[-1][0].experiment == "reference"
    (tmp_path / "bad.yaml").write_text("datasets: [cifar10]\ngrille: {alpha: [1]}\n")
    with pytest.raises(SystemExit, match="grid"):                # clé mal écrite : suggestion
        Campaign(str(tmp_path / "bad.yaml"))


def test_campaign_uses_tuned_params(tmp_path):
    from experiments.campaign import Campaign
    folder = tmp_path / "results" / "hyperparameter_search" / "fashion_mnist" / "resnet20"
    folder.mkdir(parents=True)
    (folder / "tuned_params.yaml").write_text("lr: 0.004\noptimizer: adamw\nweight_decay: 0.0001\n")
    path = write_campaign(tmp_path, {"alpha": [1]})
    text = open(path).read() + "use_tuned_params: true\n"
    open(path, "w").write(text)
    runs = Campaign(path).runs()
    assert all((r.cfg.lr, r.cfg.optimizer) == (0.004, "adamw") for r in runs)
    assert runs[-1].run_dir.endswith(os.path.join("reference", "fashion_mnist", "resnet20_x2params_adamw_lr0.004_tuned"))


def test_campaign_runs_task_by_task_then_merges(tmp_path, fake_data):
    from experiments.campaign import Campaign, main as campaign_main
    path = write_campaign(tmp_path, {"alpha": [0, 1]})
    camp = Campaign(path)
    assert len(camp.tasks()) == 4                                # indépendants, GI, DML, grand modèle
    for i in range(4):
        campaign_main([path, "--task", str(i), "--cpu", "--verbose", "0"])
    campaign_main([path, "--task", "99", "--cpu"])               # hors campagne : rien
    campaign_main([path, "--merge"])
    for run in camp.runs():
        assert os.path.exists(os.path.join(run.run_dir, "results.json")), run.run_dir
    assert read(os.path.join(camp.runs()[1].run_dir, "results.json"))["method"] == "global_interaction"
    pytest.importorskip("matplotlib")
    figures = tmp_path / "results" / "essai" / "fashion_mnist" / "resnet20_2agents" / "figures"
    files = set(os.listdir(figures))
    assert {"summary.csv", "accuracy_vs_alpha.png", "accuracy_vs_alpha.pdf", "learning_curves.png"} <= files



def test_heterogeneous_agents_config():
    from collaborative_learning.utils.config import CollaborationConfig, config_from_dict
    cfg = config_from_dict(CollaborationConfig, {"arch": "resnet32+mobilenetv2", "num_agents": 5}).validate()
    assert (cfg.num_agents, cfg.agent_archs) == (2, ["resnet32", "mobilenetv2"])     # K = nombre d'architectures
    assert cfg.variant_name().startswith("resnet32+mobilenetv2_2agents/")
    cfg = config_from_dict(CollaborationConfig, {"arch": ["resnet20", "resnet32"]}).validate()   # liste YAML
    assert cfg.arch == "resnet20+resnet32"
    assert config_from_dict(CollaborationConfig, {"arch": "resnet20", "num_agents": 3}).validate().agent_archs == ["resnet20"] * 3
    with pytest.raises(SystemExit, match="pas_un_modele"):
        config_from_dict(CollaborationConfig, {"arch": "resnet20+pas_un_modele"}).validate()


def test_campaign_heterogeneous_and_k_sweep_folders(tmp_path):
    from experiments.campaign import Campaign
    camp = Campaign(write_campaign(tmp_path, {"arch": ["resnet20+resnet32", "resnet20"], "alpha": [1]}))
    names = [os.path.relpath(r.run_dir, tmp_path / "results").replace("\\", "/") for r in camp.runs()]
    assert "essai/fashion_mnist/resnet20+resnet32_2agents/global_interaction/alpha1" in names
    assert "essai/fashion_mnist/resnet20_2agents/dml/alpha1" in names
    assert [n for n in names if n.startswith("reference")] == ["reference/fashion_mnist/resnet20_x2params_sgd_lr0.01"]
    camp = Campaign(write_campaign(tmp_path, {"num_agents": [2, 3], "alpha": [1]}))
    refs = [r for r in camp.runs() if r.experiment == "reference"]
    assert [r.cfg.num_agents for r in refs] == [2, 3]                     # un grand modèle par K
    assert all(camp.reference_for(r) is not None for r in camp.runs() if r.experiment == "collaboration")


def test_heterogeneous_campaign_runs_and_reports_each_agent(tmp_path, fake_data):
    from experiments.campaign import Campaign, main as campaign_main
    path = write_campaign(tmp_path, {"arch": ["resnet20+mobilenetv2"], "alpha": [0, 1]})
    camp = Campaign(path)
    for i in range(len(camp.tasks())):
        campaign_main([path, "--task", str(i), "--cpu", "--verbose", "0"])
    campaign_main([path, "--merge"])
    res = read(os.path.join(camp.runs()[1].run_dir, "results.json"))
    assert len(res["runs"][0]["accuracy_agents"]) == 2 and len(res["agent_params"]) == 2
    pytest.importorskip("matplotlib")
    figures = tmp_path / "results" / "essai" / "fashion_mnist" / "resnet20+mobilenetv2_2agents" / "figures"
    header = open(figures / "summary.csv").readline()
    assert "agent1_resnet20_median" in header and "agent2_mobilenetv2_median" in header


def test_search_campaign_over_architectures(tmp_path):
    import yaml
    from experiments.campaign import Campaign
    spec = {"experiment": "hyperparameter_search", "datasets": ["fashion_mnist", "cifar10"],
            "settings": {"n_workers": 2, "out_dir": str(tmp_path / "results")},
            "grid": {"arch": ["resnet20", "resnet32"]}}
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(spec))
    runs = Campaign(str(path)).runs()
    names = [os.path.relpath(r.run_dir, tmp_path / "results").replace("\\", "/") for r in runs]
    assert names == ["hyperparameter_search/fashion_mnist/resnet20", "hyperparameter_search/fashion_mnist/resnet32",
                     "hyperparameter_search/cifar10/resnet20", "hyperparameter_search/cifar10/resnet32"]
    assert len(Campaign(str(path)).tasks()) == 4 * 2
    spec["grid"] = {"lr": [0.1]}
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(SystemExit, match="arch"):
        Campaign(str(path)).runs()


def test_heterogeneous_agents_take_tuned_params_of_their_architecture(tmp_path):
    from experiments.campaign import Campaign
    for arch, text in (("resnet20", "lr: 0.004\noptimizer: adamw\nweight_decay: 0.0001\nval_size: 3000\n"),
                       ("resnet32", "lr: 0.02\noptimizer: sgd\nweight_decay: 0.0005\nval_size: 4000\n")):
        folder = tmp_path / "results" / "hyperparameter_search" / "fashion_mnist" / arch
        folder.mkdir(parents=True)
        (folder / "tuned_params.yaml").write_text(text)
    path = write_campaign(tmp_path, {"arch": ["resnet20+resnet32", "resnet20+mobilenetv2"], "alpha": [1]})
    open(path, "a").write("use_tuned_params: true\n")
    runs = {r.cfg.arch: r.cfg for r in Campaign(path).runs()}
    cfg = runs["resnet20+resnet32"]
    assert (cfg.lr, cfg.optimizer, cfg.weight_decay) == ([0.004, 0.02], ["adamw", "sgd"], [0.0001, 0.0005])
    assert cfg.val_size == 3000                                  # réglage commun : celui du premier agent
    assert runs["resnet20+mobilenetv2"].lr == 0.01               # pas de recherche pour mobilenetv2 : configs
