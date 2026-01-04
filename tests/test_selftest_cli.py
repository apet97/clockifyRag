import subprocess
import sys

from clockify_rag.selftest import run_selftest


def test_run_selftest_returns_true():
    assert run_selftest() is True


def test_cli_selftest_exit_zero():
    result = subprocess.run(
        [sys.executable, "-m", "clockify_rag.selftest"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Selftest exited with {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
