"""Packaging metadata regression tests for ragctl console script."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import tomllib

import clockify_rag.cli_modern as cli_modern


@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner()


def test_pyproject_exposes_modern_cli_script():
    """Ensure ragctl console script points to the Typer CLI app."""

    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")
    metadata = tomllib.loads(pyproject_text)

    # Support both PEP 621 format and Poetry format
    if "project" in metadata and "scripts" in metadata["project"]:
        scripts = metadata["project"]["scripts"]
    elif "tool" in metadata and "poetry" in metadata["tool"] and "scripts" in metadata["tool"]["poetry"]:
        scripts = metadata["tool"]["poetry"]["scripts"]
    else:
        pytest.fail("Could not find scripts definition in pyproject.toml")

    assert scripts["ragctl"] == "clockify_rag.cli_modern:app"


def test_ragctl_help(cli_runner: CliRunner):
    """The CLI app should show help text just like `ragctl --help`."""

    result = cli_runner.invoke(cli_modern.app, ["--help"])

    assert result.exit_code == 0
    assert "Clockify RAG" in result.stdout
