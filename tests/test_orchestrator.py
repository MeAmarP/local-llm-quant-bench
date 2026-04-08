import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.quantbench.config import load_config
from src.quantbench.models import PromptCase, RunResult
from src.quantbench.orchestrator import BenchmarkOrchestrator


@pytest.fixture
def config_manager():
    """Create a test config manager with valid config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create minimal valid configs
        (tmpdir / "benchmark.yaml").write_text("""
project: test
variants:
  - variant_id: test_variant
    quant_family: test
    precision: test
    backend: transformers
""")

        (tmpdir / "models.yaml").write_text("""
test_variant:
  model_id: test/model
  backend: transformers
  quantization: test
""")

        (tmpdir / "generation.yaml").write_text("max_new_tokens: 128")
        (tmpdir / "experiment.yaml").write_text(
            "prompt_file: p.jsonl\noutput_dir: r/\nrepetitions: 1\nwarmup_runs: 1"
        )

        yield load_config(tmpdir)


@pytest.fixture
def prompts():
    """Create test prompts."""
    return [
        PromptCase(id="p1", task="test", prompt="Test prompt 1"),
        PromptCase(id="p2", task="test", prompt="Test prompt 2"),
    ]


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestBenchmarkOrchestrator:
    def test_initialize_run_creates_directory(self, config_manager, prompts, temp_output_dir):
        orchestrator = BenchmarkOrchestrator(config_manager, prompts, temp_output_dir)
        run_dir = orchestrator.initialize_run()

        assert run_dir.exists()
        assert (run_dir / "observations.jsonl").exists()
        assert (run_dir / "notes.md").exists()
        assert (run_dir / "run_meta.json").exists()
        assert (run_dir / "prompts_snapshot.jsonl").exists()

    def test_initialize_run_creates_valid_metadata(self, config_manager, prompts, temp_output_dir):
        orchestrator = BenchmarkOrchestrator(config_manager, prompts, temp_output_dir)
        run_dir = orchestrator.initialize_run()

        meta = json.loads((run_dir / "run_meta.json").read_text())
        assert "run_id" in meta
        assert meta["num_prompts"] == len(prompts)
        assert meta["num_variants"] == 1

    def test_initialize_run_creates_prompts_snapshot(self, config_manager, prompts, temp_output_dir):
        orchestrator = BenchmarkOrchestrator(config_manager, prompts, temp_output_dir)
        run_dir = orchestrator.initialize_run()

        snapshot_path = run_dir / "prompts_snapshot.jsonl"
        lines = snapshot_path.read_text().strip().split("\n")
        assert len(lines) == len(prompts)

        for idx, line in enumerate(lines):
            data = json.loads(line)
            assert data["id"] == prompts[idx].id
            assert data["task"] == prompts[idx].task

    def test_get_runner_transformers(self, config_manager, prompts, temp_output_dir):
        orchestrator = BenchmarkOrchestrator(config_manager, prompts, temp_output_dir)

        # Mock TransformersRunner
        with patch("src.quantbench.orchestrator.TransformersRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            runner = orchestrator.get_runner("test_variant")
            assert runner is not None
            mock_runner_class.assert_called_once()

            # Verify caching
            runner2 = orchestrator.get_runner("test_variant")
            assert runner2 is runner

    def test_get_runner_llamacpp(self, config_manager, prompts, temp_output_dir):
        # Modify config to use llamacpp backend
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "benchmark.yaml").write_text("""
project: test
variants:
  - variant_id: gguf
    quant_family: gguf
    precision: q4
    backend: llamacpp
""")

            (tmpdir / "models.yaml").write_text("""
gguf:
  model_path: models/test.gguf
  backend: llamacpp
  quantization: gguf_q4
""")

            (tmpdir / "generation.yaml").write_text("max_new_tokens: 128")
            (tmpdir / "experiment.yaml").write_text(
                "prompt_file: p.jsonl\noutput_dir: r/\nrepetitions: 1\nwarmup_runs: 1"
            )

            config = load_config(tmpdir)
            orchestrator = BenchmarkOrchestrator(config, prompts, temp_output_dir)

            with patch("src.quantbench.orchestrator.LlamaCppRunner") as mock_runner_class:
                mock_runner = MagicMock()
                mock_runner_class.return_value = mock_runner

                runner = orchestrator.get_runner("gguf")
                assert runner is not None
                mock_runner_class.assert_called_once()

    def test_get_runner_invalid_backend(self, config_manager, prompts, temp_output_dir):
        # Modify config to have invalid backend
        config_manager.benchmark_config.variants[0].backend = "invalid_backend"

        orchestrator = BenchmarkOrchestrator(config_manager, prompts, temp_output_dir)
        runner = orchestrator.get_runner("test_variant")
        assert runner is None

    def test_log_observation_appends_to_jsonl(self, config_manager, prompts, temp_output_dir):
        orchestrator = BenchmarkOrchestrator(config_manager, prompts, temp_output_dir)
        orchestrator.initialize_run()

        result = RunResult(
            run_name="test",
            backend="test",
            quantization="test",
            prompt_id="p1",
            task="test",
            model_ref="model",
            prompt_chars=10,
            prompt_tokens=5,
            output_tokens=20,
            latency_sec=0.5,
            tokens_per_sec=40.0,
            load_time_sec=0.1,
            peak_gpu_mem_mb=1024,
            generated_text="output",
        )

        orchestrator.log_observation("test_variant", "p1", result)

        obs_path = orchestrator.run_dir / "observations.jsonl"
        lines = obs_path.read_text().strip().split("\n")
        assert len(lines) == 1

        obs = json.loads(lines[0])
        assert obs["variant_id"] == "test_variant"
        assert obs["prompt_id"] == "p1"
        assert obs["generated_tokens"] == 20
        assert obs["peak_gpu_memory_mb"] == 1024

    def test_log_observation_multiple_calls_accumulate(
        self, config_manager, prompts, temp_output_dir
    ):
        orchestrator = BenchmarkOrchestrator(config_manager, prompts, temp_output_dir)
        orchestrator.initialize_run()

        for i in range(3):
            result = RunResult(
                run_name="test",
                backend="test",
                quantization="test",
                prompt_id=f"p{i}",
                task="test",
                model_ref="model",
                prompt_chars=10,
                prompt_tokens=5,
                output_tokens=20,
                latency_sec=0.5,
                tokens_per_sec=40.0,
                load_time_sec=0.1,
                peak_gpu_mem_mb=1024,
                generated_text="output",
            )
            orchestrator.log_observation("test_variant", f"p{i}", result)

        obs_path = orchestrator.run_dir / "observations.jsonl"
        lines = obs_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_build_summary_aggregates_stats(self, config_manager, prompts, temp_output_dir):
        orchestrator = BenchmarkOrchestrator(config_manager, prompts, temp_output_dir)

        # Create fake observations
        orchestrator.observations = [
            {
                "variant_id": "v1",
                "wall_clock_latency_ms": 100,
                "tokens_per_sec": 50,
                "peak_gpu_memory_mb": 1000,
            },
            {
                "variant_id": "v1",
                "wall_clock_latency_ms": 110,
                "tokens_per_sec": 45,
                "peak_gpu_memory_mb": 1100,
            },
            {
                "variant_id": "v1",
                "wall_clock_latency_ms": 90,
                "tokens_per_sec": 55,
                "peak_gpu_memory_mb": 900,
            },
        ]

        summary = orchestrator._build_summary()
        assert "v1" in summary
        stats = summary["v1"]
        assert stats["num_runs"] == 3
        assert stats["latency_ms"]["median"] == 100
        assert stats["tokens_per_sec"]["median"] == 50
        assert stats["peak_gpu_memory_mb"]["max"] == 1100

    def test_build_summary_markdown_formats_table(self, config_manager, prompts, temp_output_dir):
        orchestrator = BenchmarkOrchestrator(config_manager, prompts, temp_output_dir)

        summary = {
            "v1": {
                "latency_ms": {"median": 100},
                "tokens_per_sec": {"median": 50},
                "peak_gpu_memory_mb": {"max": 1000},
            },
            "v2": {
                "latency_ms": {"median": 150},
                "tokens_per_sec": {"median": 30},
                "peak_gpu_memory_mb": {"max": 2000},
            },
        }

        markdown = orchestrator._build_summary_markdown(summary)
        assert "Variant" in markdown
        assert "v1" in markdown
        assert "v2" in markdown
        assert "100" in markdown
        assert "150" in markdown

    def test_finalize_run_creates_artifacts(self, config_manager, prompts, temp_output_dir):
        orchestrator = BenchmarkOrchestrator(config_manager, prompts, temp_output_dir)
        orchestrator.initialize_run()

        # Add a fake observation
        result = RunResult(
            run_name="test",
            backend="test",
            quantization="test",
            prompt_id="p1",
            task="test",
            model_ref="model",
            prompt_chars=10,
            prompt_tokens=5,
            output_tokens=20,
            latency_sec=0.1,
            tokens_per_sec=200.0,
            load_time_sec=None,
            peak_gpu_mem_mb=None,
            generated_text="output",
        )
        orchestrator.log_observation("test_variant", "p1", result)

        with patch("src.quantbench.orchestrator.capture_system_info") as mock_sys_info:
            mock_sys_info.return_value = {"platform": "Linux"}
            orchestrator.finalize_run()

        assert (orchestrator.run_dir / "summary.json").exists()
        assert (orchestrator.run_dir / "summary.md").exists()
        assert (orchestrator.run_dir / "system_snapshot.json").exists()

        # Verify summary format
        summary = json.loads((orchestrator.run_dir / "summary.json").read_text())
        assert "test_variant" in summary
        assert "latency_ms" in summary["test_variant"]
        assert "tokens_per_sec" in summary["test_variant"]
