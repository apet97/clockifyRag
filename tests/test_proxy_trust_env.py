import importlib

import pytest


@pytest.mark.parametrize("env_var", ["ALLOW_PROXIES", "USE_PROXY"])
def test_proxy_env_toggle_enables_trust_env(monkeypatch, env_var):
    """Setting proxy env vars should toggle requests.Session trust_env."""
    for key in ("ALLOW_PROXIES", "USE_PROXY"):
        monkeypatch.delenv(key, raising=False)

    import clockify_rag.http_utils as http_utils

    # Reload module to reset cached sessions and pick up env changes
    http_utils = importlib.reload(http_utils)
    session = http_utils.get_session(use_thread_local=True)
    assert session.trust_env is False

    monkeypatch.setenv(env_var, "1")

    http_utils = importlib.reload(http_utils)
    session = http_utils.get_session(use_thread_local=True)
    assert session.trust_env is True
