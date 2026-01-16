"""Environment guardrails for CLI-related dependencies."""

import click


def test_click_version_is_supported():
    """Ensure we stay on the Click version known to work with Typer 0.12.x."""

    parts = tuple(int(p) for p in click.__version__.split(".")[:2])
    assert parts < (8, 2), (
        "Click 8.2+ is incompatible with Typer 0.12.x. " "Pin click<8.2 or upgrade Typer before bumping."
    )
