import os
import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Tests never write into data/: DATA_DIR points to a temporary folder."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from core import config
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def pytest_collection_modifyitems(config, items):
    from core import llm
    llm_ok = None
    for item in items:
        if "llm" in item.keywords:
            if llm_ok is None:
                llm_ok = llm.is_available()
            if not llm_ok:
                item.add_marker(pytest.mark.skip(reason="LLM server (Ollama) unreachable"))
        if "gemini" in item.keywords and not os.environ.get("GEMINI_API_KEY"):
            item.add_marker(pytest.mark.skip(reason="GEMINI_API_KEY not set"))
