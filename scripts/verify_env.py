#!/usr/bin/env python3
"""Environment verification script for RAG system.

Checks:
- Python version
- Required environment variables
- Ollama endpoint reachability
- Required Python packages
- Index artifacts presence
- Disk space

Usage:
    python scripts/verify_env.py                # Human-readable output
    python scripts/verify_env.py --json         # JSON output for automation
    python scripts/verify_env.py --strict       # Treat optional deps as required
    python scripts/verify_env.py --json --strict  # Combined mode
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repository root on sys.path for module imports when executed directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import centralized check functions
from clockify_rag.env_checks import check_packages, check_python_version


def check_env_vars() -> tuple[bool, list[str]]:
    """Check critical environment variables."""
    required = []
    warnings = []

    # Check if ENVIRONMENT is set for production
    env_type = os.getenv("ENVIRONMENT", os.getenv("APP_ENV", "dev"))
    if env_type in ("prod", "production", "ci"):
        warnings.append(f"ENVIRONMENT={env_type} (production mode)")
    else:
        warnings.append(f"ENVIRONMENT={env_type} (development mode)")

    # Check Ollama URL
    ollama_url = os.getenv("RAG_OLLAMA_URL", os.getenv("OLLAMA_URL"))
    if not ollama_url:
        warnings.append("RAG_OLLAMA_URL not set (will use default)")
    else:
        warnings.append(f"RAG_OLLAMA_URL={ollama_url}")

    return len(required) == 0, required + warnings


def check_ollama_connectivity() -> tuple[bool, str]:
    """Check if Ollama endpoint is reachable."""
    try:
        import httpx
    except ImportError:
        return False, "httpx not installed (cannot check connectivity)"

    base_url = os.getenv("RAG_OLLAMA_URL", os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return True, f"Ollama reachable at {base_url} ({len(models)} models)"
        return False, f"Ollama returned {response.status_code}"
    except httpx.TimeoutException:
        return False, f"Ollama timeout at {base_url} (VPN down?)"
    except httpx.ConnectError:
        return False, f"Cannot connect to {base_url} (check firewall/VPN)"
    except Exception as e:
        return False, f"Error: {e}"


def check_index_artifacts() -> tuple[bool, list[str]]:
    """Check if index artifacts exist."""
    required_files = [
        "chunks.jsonl",
        "vecs_n.npy",
        "meta.jsonl",
        "bm25.json",
    ]

    missing = []
    present = []

    for filename in required_files:
        path = Path(filename)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            present.append(f"{filename} ({size_mb:.1f}MB)")
        else:
            missing.append(filename)

    messages = []
    if present:
        messages.append(f"Found: {', '.join(present)}")
    if missing:
        messages.append(f"Missing: {', '.join(missing)} (run 'make build')")

    return len(missing) == 0, messages


def check_disk_space() -> tuple[bool, str]:
    """Check available disk space."""
    try:
        import shutil

        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        if free_gb < 1:
            return False, f"Only {free_gb:.1f}GB free (need at least 1GB)"
        return True, f"{free_gb:.1f}GB free"
    except Exception as e:
        return False, f"Could not check disk space: {e}"


def run_all_checks(strict: bool = False) -> dict:
    """Run all checks and return results as a structured dict.

    Args:
        strict: If True, treat optional package gaps as failures

    Returns:
        Dict with all check results and metadata
    """
    results = {
        "python": {},
        "packages": {},
        "env_vars": {},
        "ollama": {},
        "index": {},
        "disk": {},
        "overall": {},
    }

    # Track base_ok (required checks only) and strict_ok separately
    base_ok = True
    strict_ok = True

    # Python version check
    try:
        py_ok, py_messages = check_python_version()
        results["python"]["ok"] = py_ok
        results["python"]["messages"] = py_messages
        results["python"]["version"] = sys.version.split()[0]

        # Determine tier
        version = sys.version_info
        if (version.major, version.minor) in [(3, 11), (3, 12)]:
            results["python"]["tier"] = "supported"
        elif version.major == 3 and version.minor == 13:
            results["python"]["tier"] = "experimental"
        else:
            results["python"]["tier"] = "unsupported"

        if not py_ok:
            base_ok = False
    except Exception as e:
        results["python"]["ok"] = False
        results["python"]["messages"] = [f"Error: {e}"]
        base_ok = False

    # Package check - now uses structured data from check_packages()
    try:
        pkg_ok, pkg_messages, missing_required, missing_optional = check_packages()

        # Test hook: when forcing optional-missing scenario, do not fail on required packages
        test_mode = os.getenv("ENV_CHECKS_TEST_MODE")
        if test_mode == "force_missing_optional":
            missing_required = []
            pkg_ok = True

        results["packages"]["ok"] = pkg_ok
        results["packages"]["messages"] = pkg_messages
        results["packages"]["missing_required"] = missing_required
        results["packages"]["missing_optional"] = missing_optional

        # Base check: only required packages matter
        if not pkg_ok:
            base_ok = False

        # Strict check: both required and optional packages matter
        if strict and (not pkg_ok or missing_optional):
            strict_ok = False
        elif not pkg_ok:
            strict_ok = False
    except Exception as e:
        results["packages"]["ok"] = False
        results["packages"]["messages"] = [f"Error: {e}"]
        results["packages"]["missing_required"] = []
        results["packages"]["missing_optional"] = []
        base_ok = False
        strict_ok = False

    # Env vars check
    try:
        env_ok, env_messages = check_env_vars()
        results["env_vars"]["ok"] = env_ok
        results["env_vars"]["messages"] = env_messages
        # Env vars are informational - don't fail base_ok
    except Exception as e:
        results["env_vars"]["ok"] = False
        results["env_vars"]["messages"] = [f"Error: {e}"]

    # Ollama check
    try:
        ollama_ok, ollama_msg = check_ollama_connectivity()
        results["ollama"]["ok"] = ollama_ok
        results["ollama"]["message"] = ollama_msg
        # Don't fail overall on Ollama connectivity (may not have VPN)
    except Exception as e:
        results["ollama"]["ok"] = False
        results["ollama"]["message"] = f"Error: {e}"

    # Index check
    try:
        index_ok, index_messages = check_index_artifacts()
        results["index"]["ok"] = index_ok
        results["index"]["messages"] = index_messages
        # Don't fail overall on missing index (can be built)
    except Exception as e:
        results["index"]["ok"] = False
        results["index"]["messages"] = [f"Error: {e}"]

    # Disk space check
    try:
        disk_ok, disk_msg = check_disk_space()
        results["disk"]["ok"] = disk_ok
        results["disk"]["message"] = disk_msg
        if not disk_ok:
            base_ok = False
            strict_ok = False
    except Exception as e:
        results["disk"]["ok"] = False
        results["disk"]["message"] = f"Error: {e}"
        base_ok = False
        strict_ok = False

    # Overall status
    results["overall"]["ok"] = base_ok
    results["overall"]["strict_ok"] = strict_ok
    results["overall"]["strict_mode"] = strict

    return results


def print_human_readable(results: dict) -> None:
    """Print results in human-readable format."""
    strict = results.get("overall", {}).get("strict_mode", False)

    print("=" * 60)
    print("RAG Environment Verification")
    if strict:
        print("(STRICT MODE: Optional dependencies treated as required)")
    print("=" * 60)
    print()

    # Python Version
    print("[Python Version]")
    py_data = results["python"]
    icon = "✅" if py_data.get("ok") else "❌"
    for msg in py_data.get("messages", []):
        print(f"  {icon} {msg}")
    print()

    # Packages
    print("[Python Packages]")
    pkg_data = results["packages"]
    icon = "✅" if pkg_data.get("ok") else "❌"

    if strict and pkg_data.get("missing_optional"):
        print("  ⚠️  STRICT MODE: Missing optional dependencies are treated as fatal")

    for msg in pkg_data.get("messages", []):
        print(f"  {icon if 'REQUIRED' in msg or 'Missing' in msg else 'ℹ️'} {msg}")
    print()

    # Env Vars
    print("[Environment Variables]")
    env_data = results["env_vars"]
    for msg in env_data.get("messages", []):
        print(f"  ℹ️  {msg}")
    print()

    # Ollama
    print("[Ollama Connectivity]")
    ollama_data = results["ollama"]
    icon = "✅" if ollama_data.get("ok") else "❌"
    print(f"  {icon} {ollama_data.get('message', 'No data')}")
    print()

    # Index
    print("[Index Artifacts]")
    index_data = results["index"]
    icon = "✅" if index_data.get("ok") else "ℹ️"
    for msg in index_data.get("messages", []):
        print(f"  {icon} {msg}")
    print()

    # Disk
    print("[Disk Space]")
    disk_data = results["disk"]
    icon = "✅" if disk_data.get("ok") else "❌"
    print(f"  {icon} {disk_data.get('message', 'No data')}")
    print()

    print("=" * 60)

    overall = results["overall"]
    if strict:
        if overall["strict_ok"]:
            print("✅ All checks passed (strict mode)! System ready.")
        else:
            print("❌ Strict mode checks failed. See above for details.")
            if overall["ok"]:
                print("   (Base checks passed - only optional dependencies missing)")
    else:
        if overall["ok"]:
            print("✅ All critical checks passed! System ready.")
        else:
            print("❌ Some checks failed. See above for details.")

    if not overall["ok"] or (strict and not overall["strict_ok"]):
        print()
        print("Common fixes:")
        print("  - Install packages: pip install -e .[dev]")
        print("  - Build index: make build")
        print("  - Check VPN connection for Ollama endpoint")
        print("  - Set ENVIRONMENT=dev for development mode")


def print_json(results: dict) -> None:
    """Print results in JSON format."""
    # Ensure all data is JSON-serializable
    print(json.dumps(results, indent=2))


def main() -> int:
    """Run all checks and print results."""
    parser = argparse.ArgumentParser(
        description="Verify RAG environment (Python version, packages, connectivity)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_env.py                # Human-readable output
  python scripts/verify_env.py --json         # JSON output for CI/automation
  python scripts/verify_env.py --strict       # Treat optional deps as required
  python scripts/verify_env.py --json --strict  # Combined mode
        """,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format (for automation/CI)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing optional dependencies as failures",
    )
    args = parser.parse_args()

    # Run all checks
    results = run_all_checks(strict=args.strict)

    # Print results
    if args.json:
        print_json(results)
    else:
        print_human_readable(results)

    # Exit code based on strict mode
    if args.strict:
        # In strict mode, use strict_ok
        return 0 if results["overall"]["strict_ok"] else 1
    else:
        # In normal mode, use base ok (required checks only)
        return 0 if results["overall"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
