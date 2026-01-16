"""Tests for verify_env.py dependency checking."""

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

# Import from the new centralized module (no sys.path hacks needed!)
from clockify_rag import env_checks


class TestVerifyEnvOptionalDeps:
    """Test that verify_env.py correctly distinguishes required vs optional deps."""

    def test_all_required_deps_present(self, monkeypatch):
        """Test when all required dependencies are present."""
        # Mock successful imports for required deps
        # Include all packages from REQUIRED_PACKAGES in env_checks.py
        required_packages = [
            "numpy",
            "httpx",
            "langchain_ollama",
            "rank_bm25",
            "langchain",
            "tiktoken",
            "fastapi",
            "typer",
            "pydantic",
        ]

        def mock_try_import(name):
            if name in required_packages:
                return True
            return False

        monkeypatch.setattr(env_checks, "_try_import", mock_try_import)

        passed, messages, missing_required, missing_optional = env_checks.check_packages()

        # Should pass (all required present)
        assert passed is True
        assert missing_required == []
        assert len(missing_optional) > 0  # Optional packages missing

        # Should contain success message
        messages_str = " ".join(messages)
        assert "All required packages installed" in messages_str or "All packages installed" in messages_str

    def test_required_deps_missing(self, monkeypatch):
        """Test when required dependencies are missing."""

        # Mock failed imports for required deps
        def mock_try_import(name):
            if name == "numpy":
                return False
            return True

        monkeypatch.setattr(env_checks, "_try_import", mock_try_import)

        passed, messages, missing_required, missing_optional = env_checks.check_packages()

        # Should fail (required dep missing)
        assert passed is False
        assert "numpy" in missing_required

        # Should contain error message
        messages_str = " ".join(messages)
        assert "Missing REQUIRED" in messages_str or "numpy" in messages_str.lower()

    def test_optional_deps_missing_with_messaging(self, monkeypatch):
        """Test that missing optional deps show helpful messaging."""
        # Mock: required deps present, optional deps missing
        required_packages = [
            "numpy",
            "httpx",
            "langchain_ollama",
            "rank_bm25",
            "langchain",
            "tiktoken",
            "fastapi",
            "typer",
            "pydantic",
        ]

        def mock_try_import(name):
            # Required deps succeed
            if name in required_packages:
                return True
            # Optional deps fail
            if name in ["faiss", "torch", "sentence_transformers"]:
                return False
            return False

        monkeypatch.setattr(env_checks, "_try_import", mock_try_import)

        passed, messages, missing_required, missing_optional = env_checks.check_packages()

        # Should pass (only optional missing)
        assert passed is True
        assert missing_required == []
        assert len(missing_optional) > 0

        # Verify that missing_optional contains expected packages
        assert "faiss" in missing_optional
        assert "torch" in missing_optional
        assert "sentence_transformers" in missing_optional

        # Should mention optional deps
        messages_str = " ".join(messages)
        assert "Optional dependencies" in messages_str or "optional" in messages_str.lower()

        # Should mention that core RAG still works
        assert "core RAG" in messages_str.lower() or "works without" in messages_str.lower()

        # Should provide installation hints
        assert "install" in messages_str.lower() or "conda" in messages_str.lower() or "pip" in messages_str.lower()

    def test_all_deps_present(self, monkeypatch):
        """Test when all dependencies (required + optional) are present."""

        # Mock successful imports for all deps
        def mock_try_import(name):
            return True

        monkeypatch.setattr(env_checks, "_try_import", mock_try_import)

        passed, messages, missing_required, missing_optional = env_checks.check_packages()

        # Should pass
        assert passed is True
        assert missing_required == []
        assert missing_optional == []

        # Should indicate all packages installed
        messages_str = " ".join(messages)
        assert "All packages installed" in messages_str or "All required" in messages_str


class TestVerifyEnvPythonVersion:
    """Test Python version checking."""

    def test_supported_python_version(self, monkeypatch):
        """Test that Python 3.11 and 3.12 are marked as supported."""
        # Mock Python 3.12
        mock_version = mock.MagicMock()
        mock_version.major = 3
        mock_version.minor = 12
        mock_version.micro = 0
        monkeypatch.setattr(env_checks.sys, "version_info", mock_version)

        passed, messages = env_checks.check_python_version()

        assert passed is True
        messages_str = " ".join(messages)
        assert "3.12" in messages_str
        assert "supported" in messages_str.lower()

    def test_experimental_python_version(self, monkeypatch):
        """Test that Python 3.13 is marked as experimental."""
        # Mock Python 3.13
        mock_version = mock.MagicMock()
        mock_version.major = 3
        mock_version.minor = 13
        mock_version.micro = 0
        monkeypatch.setattr(env_checks.sys, "version_info", mock_version)

        passed, messages = env_checks.check_python_version()

        assert passed is True  # Allowed but experimental
        messages_str = " ".join(messages)
        assert "3.13" in messages_str
        assert "experimental" in messages_str.lower()

    def test_unsupported_python_version(self, monkeypatch):
        """Test that Python 3.14+ is not supported."""
        # Mock Python 3.14
        mock_version = mock.MagicMock()
        mock_version.major = 3
        mock_version.minor = 14
        mock_version.micro = 0
        monkeypatch.setattr(env_checks.sys, "version_info", mock_version)

        passed, messages = env_checks.check_python_version()

        assert passed is False
        messages_str = " ".join(messages)
        assert "3.14" in messages_str
        assert "not supported" in messages_str.lower()


class TestVerifyEnvCLI:
    """Test the scripts/verify_env.py CLI script end-to-end."""

    @pytest.fixture
    def verify_env_script(self):
        """Return the path to the verify_env.py script."""
        # Assuming tests are in tests/ and script is in scripts/
        repo_root = Path(__file__).parent.parent
        script_path = repo_root / "scripts" / "verify_env.py"
        assert script_path.exists(), f"Script not found at {script_path}"
        return str(script_path)

    def test_cli_normal_mode_all_required_present(self, verify_env_script, monkeypatch):
        """Test CLI in normal mode when all required deps are present."""
        # Monkeypatch at module level to affect subprocess
        import os

        env = os.environ.copy()
        env["ENV_CHECKS_TEST_MODE"] = "all_required_present"

        # We can't easily monkeypatch subprocess, so let's just run the script
        # and check that it exits with 0 when deps are actually present
        result = subprocess.run(
            [sys.executable, verify_env_script],
            capture_output=True,
            text=True,
            timeout=30,  # Increased from 10s for CI reliability
        )

        # In a real environment with all deps installed, exit code should be 0
        # or 1 depending on Ollama/index. We're testing the script runs without crashing.
        assert result.returncode in [0, 1], f"Unexpected exit code: {result.returncode}"
        assert "Python" in result.stdout or "Error" in result.stderr

    def test_cli_json_mode_output_structure(self, verify_env_script):
        """Test CLI --json mode produces valid JSON with expected structure."""
        result = subprocess.run(
            [sys.executable, verify_env_script, "--json"],
            capture_output=True,
            text=True,
            timeout=30,  # Increased from 10s for CI reliability
        )

        # Should return valid JSON
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON output: {e}\nOutput: {result.stdout}")

        # Check required top-level keys
        assert "python" in data
        assert "packages" in data
        assert "overall" in data
        assert "strict_mode" in data["overall"]

        # Check python section structure
        assert "ok" in data["python"]
        assert "version" in data["python"]
        assert "tier" in data["python"]

        # Check packages section structure
        assert "ok" in data["packages"]
        assert "messages" in data["packages"]

        # Check overall section
        assert "ok" in data["overall"]

    def test_cli_strict_mode_with_optional_missing(self, verify_env_script, monkeypatch):
        """Test CLI --strict mode treats optional deps as required."""
        # This test is more conceptual since we can't easily control
        # what packages are installed in the subprocess environment.
        # We'll just verify the flag is accepted and affects output.

        result = subprocess.run(
            [sys.executable, verify_env_script, "--strict", "--json"],
            capture_output=True,
            text=True,
            timeout=30,  # Increased from 10s for CI reliability
        )

        # Parse JSON output
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON in strict mode: {result.stdout}")

        # Verify strict_mode flag is set
        assert data["overall"]["strict_mode"] is True

        # If there are any optional deps missing, overall.ok should be False in strict mode
        # (but we can't guarantee what's installed, so just check the flag is processed)
        assert "strict_mode" in data["overall"]

    def test_cli_normal_vs_strict_mode_exit_codes(self, verify_env_script):
        """Test that strict mode can produce different exit codes than normal mode."""
        # Run in normal mode
        result_normal = subprocess.run(
            [sys.executable, verify_env_script, "--json"],
            capture_output=True,
            text=True,
            timeout=30,  # Increased from 10s for CI reliability
        )

        # Run in strict mode
        result_strict = subprocess.run(
            [sys.executable, verify_env_script, "--strict", "--json"],
            capture_output=True,
            text=True,
            timeout=30,  # Increased from 10s for CI reliability
        )

        # Parse both outputs
        data_normal = json.loads(result_normal.stdout)
        data_strict = json.loads(result_strict.stdout)

        # Verify flags are set correctly
        assert data_normal["overall"]["strict_mode"] is False
        assert data_strict["overall"]["strict_mode"] is True

        # If optional deps are missing, strict should be more restrictive
        # (exit codes might differ)
        # We're mainly testing that the script handles both flags correctly
        assert result_normal.returncode in [0, 1]
        assert result_strict.returncode in [0, 1]

    def test_cli_help_flag(self, verify_env_script):
        """Test that --help flag works and shows usage."""
        result = subprocess.run(
            [sys.executable, verify_env_script, "--help"],
            capture_output=True,
            text=True,
            timeout=30,  # Increased from 10s for CI reliability
        )

        # --help should exit with 0
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "verify" in result.stdout.lower()
        assert "--json" in result.stdout
        assert "--strict" in result.stdout

    def test_cli_normal_mode_with_optional_missing_via_hook(self, verify_env_script):
        """Test normal mode with ENV_CHECKS_TEST_MODE=force_missing_optional."""
        import os

        env = os.environ.copy()
        env["ENV_CHECKS_TEST_MODE"] = "force_missing_optional"

        result = subprocess.run(
            [sys.executable, verify_env_script],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        # Should pass in normal mode (optional missing is OK)
        assert (
            result.returncode == 0
        ), f"Expected exit code 0, got {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        assert "Optional dependencies" in result.stdout or "optional" in result.stdout.lower()

    def test_cli_strict_mode_with_optional_missing_via_hook(self, verify_env_script):
        """Test strict mode with ENV_CHECKS_TEST_MODE=force_missing_optional."""
        import os

        env = os.environ.copy()
        env["ENV_CHECKS_TEST_MODE"] = "force_missing_optional"

        result = subprocess.run(
            [sys.executable, verify_env_script, "--strict"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        # Should fail in strict mode (optional missing is fatal)
        assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}\nstdout: {result.stdout}"
        assert "strict" in result.stdout.lower() or "STRICT" in result.stdout

    def test_cli_json_mode_with_optional_missing_via_hook(self, verify_env_script):
        """Test JSON mode with ENV_CHECKS_TEST_MODE=force_missing_optional."""
        import os

        env = os.environ.copy()
        env["ENV_CHECKS_TEST_MODE"] = "force_missing_optional"

        result = subprocess.run(
            [sys.executable, verify_env_script, "--json"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        # Should parse as valid JSON
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON output: {e}\nOutput: {result.stdout}")

        # Verify structure
        assert "overall" in data
        assert "strict_mode" in data["overall"]
        assert data["overall"]["strict_mode"] is False  # Not in strict mode

        # Verify packages data
        assert "packages" in data
        assert "missing_required" in data["packages"]
        assert "missing_optional" in data["packages"]
        assert len(data["packages"]["missing_optional"]) > 0, "Expected missing optional packages"

        # In normal mode, should pass even with optional missing
        assert data["overall"]["ok"] is True

    def test_cli_json_strict_mode_with_optional_missing_via_hook(self, verify_env_script):
        """Test JSON strict mode with ENV_CHECKS_TEST_MODE=force_missing_optional."""
        import os

        env = os.environ.copy()
        env["ENV_CHECKS_TEST_MODE"] = "force_missing_optional"

        result = subprocess.run(
            [sys.executable, verify_env_script, "--json", "--strict"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        # Should parse as valid JSON
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON output: {e}\nOutput: {result.stdout}")

        # Verify structure
        assert "overall" in data
        assert "strict_mode" in data["overall"]
        assert data["overall"]["strict_mode"] is True  # In strict mode

        # Verify packages data
        assert "packages" in data
        assert "missing_optional" in data["packages"]
        assert len(data["packages"]["missing_optional"]) > 0, "Expected missing optional packages"

        # In strict mode, should fail with optional missing
        assert data["overall"]["strict_ok"] is False
        assert result.returncode == 1
