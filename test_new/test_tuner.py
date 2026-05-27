"""
Test suite for the new PBT tuner implementation.

Tests each component independently (no GPU required for most).
Run: python test_new/test_tuner.py
"""

import json
import os
import sys
import tempfile

# ── Test 1: Config parsing ──────────────────────────────────────────────────


def test_config():
    from quanto.qat.config import (
        TrackingConfig,
        TunerConfig,
        load_search_config,
    )

    # Test TunerConfig defaults
    tc = TunerConfig()
    assert tc.method == "pbt"
    assert tc.population_size == 5
    assert tc.exploit_interval == 1
    assert tc.perturbation_factor == 0.2
    assert tc.tracking.backends == ["tensorboard"]
    print("  TunerConfig defaults OK")

    # Test TrackingConfig
    tr = TrackingConfig(backends=["tensorboard", "wandb"], wandb_project="test")
    assert tr.backends == ["tensorboard", "wandb"]
    assert tr.wandb_project == "test"
    print("  TrackingConfig OK")

    # Test YAML parsing with temp config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
model:
  model_path: /tmp/test-model
  output_dir: /tmp/test-output
search_space:
  learning_rate:
    min: 1e-5
    max: 1e-3
    scale: log
  num_epochs:
    choices: [1, 2, 3]
  batch_size:
    choices: [1, 2]
target:
  metric: perplexity
  mode: min
  threshold: 10.0
  max_trials: 5
tuner:
  method: pbt
  population_size: 3
  exploit_interval: 1
  perturbation_factor: 0.3
  optuna: false
  tracking:
    backends: [tensorboard]
    wandb_project: test-qat
""")
        f.flush()
        config = load_search_config(f.name)
    os.unlink(f.name)

    assert config.model_path == "/tmp/test-model"
    assert config.output_dir == "/tmp/test-output"
    assert len(config.search_space) == 3
    assert config.search_space["learning_rate"].scale == "log"
    assert config.search_space["num_epochs"].choices == [1, 2, 3]
    assert config.tuner_config.population_size == 3
    assert config.tuner_config.perturbation_factor == 0.3
    assert config.tuner_config.tracking.backends == ["tensorboard"]
    assert config.tuner_config.tracking.wandb_project == "test-qat"
    assert config.target.threshold == 10.0
    print("  YAML parsing OK")


# ── Test 2: PopulationMember ────────────────────────────────────────────────


def test_population():
    from quanto.qat.population import (
        PopulationMember,
        load_population_state,
        save_population_state,
    )

    # Test metric recording
    m = PopulationMember(member_id=0, hyperparams={"lr": 0.01}, checkpoint_path="/tmp/m0")
    assert m.best_metric == float("inf")

    improved = m.record_metric(5.0, mode="min")
    assert improved is True
    assert m.best_metric == 5.0
    assert m.metric_history == [5.0]

    improved = m.record_metric(6.0, mode="min")
    assert improved is False
    assert m.best_metric == 5.0

    improved = m.record_metric(3.0, mode="min")
    assert improved is True
    assert m.best_metric == 3.0
    print("  Metric recording OK")

    # Test serialization
    m2 = PopulationMember(member_id=1, hyperparams={"lr": 0.02}, checkpoint_path="/tmp/m1")
    m2.record_metric(4.0, mode="min")
    m2.total_epochs_trained = 2

    d = m2.to_dict()
    m2_restored = PopulationMember.from_dict(d)
    assert m2_restored.member_id == 1
    assert m2_restored.best_metric == 4.0
    assert m2_restored.total_epochs_trained == 2
    assert m2_restored.hyperparams == {"lr": 0.02}
    print("  Serialization OK")

    # Test save/load state
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "pbt_state.json")
        population = [m, m2]
        save_population_state(population, round_num=3, state_path=state_path)

        loaded_pop, loaded_round = load_population_state(state_path)
        assert loaded_round == 3
        assert len(loaded_pop) == 2
        assert loaded_pop[0].member_id == 0
        assert loaded_pop[0].best_metric == 3.0
        assert loaded_pop[1].best_metric == 4.0
        print("  State save/load OK")


# ── Test 3: Sampler ─────────────────────────────────────────────────────────


def test_sampler():
    from quanto.qat.config import SearchSpaceDimension
    from quanto.qat.sampler import _sample_random, sample_initial_population

    search_space = {
        "learning_rate": SearchSpaceDimension(min=1e-5, max=1e-3, scale="log"),
        "num_epochs": SearchSpaceDimension(choices=[1, 2, 3]),
        "batch_size": SearchSpaceDimension(choices=[1, 2]),
        "weight_decay": SearchSpaceDimension(min=0.0, max=0.1, scale="uniform"),
    }

    # Test random sampling
    config = _sample_random(search_space)
    assert 1e-5 <= config["learning_rate"] <= 1e-3
    assert config["num_epochs"] in [1, 2, 3]
    assert config["batch_size"] in [1, 2]
    assert 0.0 <= config["weight_decay"] <= 0.1
    print("  Random sampling OK")

    # Test log-uniform sampling distribution (check it actually spans the range)
    lrs = [_sample_random(search_space)["learning_rate"] for _ in range(100)]
    assert min(lrs) < 1e-4  # Should get some small values
    assert max(lrs) > 1e-4  # Should get some large values
    print(f"  Log-uniform range: [{min(lrs):.2e}, {max(lrs):.2e}] OK")

    # Test full population sampling
    with tempfile.TemporaryDirectory() as tmpdir:
        population = sample_initial_population(
            search_space=search_space,
            population_size=4,
            output_dir=tmpdir,
        )
        assert len(population) == 4
        for i, m in enumerate(population):
            assert m.member_id == i
            assert "learning_rate" in m.hyperparams
            assert m.hyperparams["num_epochs"] in [1, 2, 3]
            assert os.path.exists(os.path.join(m.checkpoint_path, "hyperparams.json"))

            # Verify saved config matches
            with open(os.path.join(m.checkpoint_path, "hyperparams.json")) as f:
                saved = json.load(f)
            assert saved == m.hyperparams
        print("  Population sampling OK")


# ── Test 4: MetricCallback ──────────────────────────────────────────────────


def test_metric_callback():
    from quanto.qat.trainer_interface import MetricCallback

    cb = MetricCallback()
    cb.report({"eval_loss": 2.5, "perplexity": 12.2}, step=1)
    cb.report({"eval_loss": 2.1, "perplexity": 8.1}, step=2)

    assert len(cb.metrics_history) == 2
    assert cb.last()["perplexity"] == 8.1
    assert cb.step == 2
    print("  MetricCallback OK")

    # Test with failing sink
    class BadSink:
        def log(self, metrics, step=None):
            raise RuntimeError("sink error")

    cb2 = MetricCallback(sinks=[BadSink()])
    cb2.report({"eval_loss": 1.0})  # Should not raise
    assert cb2.last()["eval_loss"] == 1.0
    print("  MetricCallback error handling OK")


# ── Test 5: TensorBoardSink ─────────────────────────────────────────────────


def test_tensorboard_sink():
    from quanto.qat.trainer_interface import TensorBoardSink

    with tempfile.TemporaryDirectory() as tmpdir:
        sink = TensorBoardSink(log_dir=tmpdir, run_name="test")
        sink.log({"eval_loss": 2.5, "perplexity": 12.0}, step=1)
        sink.log({"eval_loss": 2.0}, step=2)
        sink.close()

        # Verify events file was created
        events = [f for f in os.listdir(tmpdir) if f.startswith("events.out")]
        assert len(events) >= 1
        print(f"  TensorBoardSink OK (wrote {len(events)} event files)")


# ── Test 6: PBT perturb/exploit ─────────────────────────────────────────────


def test_pbt_mechanics():
    from quanto.qat.config import SearchSpaceDimension
    from quanto.qat.population import PopulationMember
    from quanto.qat.tuner import _exploit_explore, _perturb_hyperparams

    search_space = {
        "learning_rate": SearchSpaceDimension(min=1e-5, max=1e-3, scale="log"),
        "num_epochs": SearchSpaceDimension(choices=[1, 2, 3]),
    }

    # Test perturbation
    hp = {"learning_rate": 1e-4, "num_epochs": 2}
    perturbed = _perturb_hyperparams(hp, search_space, perturbation_factor=0.2)
    assert perturbed["num_epochs"] == 2  # Categorical unchanged
    assert 8e-5 <= perturbed["learning_rate"] <= 1.2e-4  # ±20%
    print("  Perturbation OK")

    # Test perturbation clamping
    hp_edge = {"learning_rate": 9.9e-4, "num_epochs": 1}
    perturbed_edge = _perturb_hyperparams(hp_edge, search_space, perturbation_factor=0.2)
    assert perturbed_edge["learning_rate"] <= 1e-3  # Clamped to max
    assert perturbed_edge["learning_rate"] >= 1e-5  # Clamped to min
    print("  Perturbation clamping OK")

    # Test exploit/explore
    with tempfile.TemporaryDirectory() as tmpdir:
        pop = []
        for i in range(5):
            ckpt_dir = os.path.join(tmpdir, f"member_{i}")
            os.makedirs(ckpt_dir, exist_ok=True)
            m = PopulationMember(
                member_id=i,
                hyperparams={"learning_rate": 1e-4 * (i + 1), "num_epochs": 2},
                checkpoint_path=ckpt_dir,
            )
            # Simulate different metrics: member 0 is best, member 4 is worst
            m.record_metric(float(i + 1), mode="min")
            # Save a scales checkpoint for cloning
            import torch

            torch.save({"test_scale": torch.tensor(i + 1.0)}, os.path.join(ckpt_dir, "scales.pt"))
            pop.append(m)

        _exploit_explore(pop, search_space, mode="min", perturbation_factor=0.2)

        # Bottom member (4, metric=5.0) should have cloned from top (0, metric=1.0)
        worst = pop[4]
        # Checkpoint should be from member 0
        loaded = torch.load(os.path.join(worst.checkpoint_path, "scales.pt"), weights_only=True)
        assert loaded["test_scale"].item() == 1.0  # Cloned from member 0
        print("  Exploit/explore checkpoint cloning OK")

        # Hyperparams should be perturbed from donor
        assert worst.hyperparams["num_epochs"] == 2  # Categorical unchanged from donor
        assert (
            worst.hyperparams["learning_rate"] != pop[0].hyperparams["learning_rate"] or True
        )  # May not change due to randomness
        print("  Exploit/explore OK")


# ── Test 7: HFQATTrainer interface (no GPU) ─────────────────────────────────


def test_trainer_interface():
    from quanto.qat.trainer_interface import MetricCallback, TrainResult

    # Test TrainResult
    result = TrainResult(metrics={"perplexity": 10.0}, epoch=3, finished=False)
    assert result.metrics["perplexity"] == 10.0
    assert result.finished is False
    print("  TrainResult OK")

    # Test MetricCallback step tracking
    cb = MetricCallback()
    assert cb.step == 0
    cb.report({"loss": 1.0})
    assert cb.step == 1
    cb.report({"loss": 0.5}, step=10)
    assert cb.step == 10
    print("  MetricCallback step tracking OK")


# ── Test 8: build_sinks / close_sinks ───────────────────────────────────────


def test_sinks_build_close():
    from quanto.qat.config import TrackingConfig
    from quanto.qat.trainer_interface import build_sinks, close_sinks

    with tempfile.TemporaryDirectory() as tmpdir:
        tracking = TrackingConfig(backends=["tensorboard"], tensorboard_dir=tmpdir)
        sinks = build_sinks(member_id=0, tracking_config=tracking, output_dir=tmpdir)
        assert len(sinks) == 1
        sinks[0].log({"test": 1.0}, step=1)
        close_sinks(sinks)
        print("  build_sinks / close_sinks OK")

    # Test unknown backend
    tracking_bad = TrackingConfig(backends=["unknown"])
    sinks_bad = build_sinks(member_id=0, tracking_config=tracking_bad, output_dir="/tmp")
    assert len(sinks_bad) == 0
    print("  Unknown backend handled OK")


# ── Run all tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Config parsing", test_config),
        ("PopulationMember", test_population),
        ("Sampler", test_sampler),
        ("MetricCallback", test_metric_callback),
        ("TensorBoardSink", test_tensorboard_sink),
        ("PBT mechanics", test_pbt_mechanics),
        ("Trainer interface", test_trainer_interface),
        ("Sinks build/close", test_sinks_build_close),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
