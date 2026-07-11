"""Import smoke tests for core CLI entry points (no API calls)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_script_module(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_arcs_main():
    from arcs.main import PipelineError, classify_pipeline_error, run_pipeline

    assert callable(run_pipeline)
    assert callable(classify_pipeline_error)
    assert issubclass(PipelineError, Exception)


def test_import_repair():
    repair = _load_script_module("repair")
    from arcs.repair.orchestrator import COMPONENTS, repair_all

    assert callable(repair.main)
    assert callable(repair_all)
    assert "ROUTER" in COMPONENTS


def test_import_eval_pipeline():
    eval_pipeline = _load_script_module("eval_pipeline")

    assert callable(eval_pipeline.main)
    assert hasattr(eval_pipeline, "run_eval")


def test_import_rq1_run():
    rq1_run = _load_script_module("rq1_run")

    assert callable(rq1_run.main)
