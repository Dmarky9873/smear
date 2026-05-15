from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODEL_FILES = {
    f"backend/bots/models/neural_3p_v{version}.json"
    for version in range(1, 6)
}


def test_only_runtime_neural_models_are_tracked() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("repository metadata is not available")

    result = subprocess.run(
        ["git", "ls-files", "backend/bots/models"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_model_files = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    }

    assert tracked_model_files == RUNTIME_MODEL_FILES


@pytest.mark.parametrize(
    "ignore_file",
    [".dockerignore", ".railwayignore"],
)
def test_production_ignores_training_model_artifacts(ignore_file: str) -> None:
    patterns = (REPO_ROOT / ignore_file).read_text(encoding="utf-8").splitlines()

    assert "backend/bots/models/**" in patterns
    assert "!backend/bots/models/" in patterns
    assert "!backend/bots/models/neural_3p_v*.json" in patterns
