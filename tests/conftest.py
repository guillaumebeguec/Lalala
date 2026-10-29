import os
import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Les tests n'écrivent jamais dans data/ : DATA_DIR pointe sur un dossier temporaire."""
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
                item.add_marker(pytest.mark.skip(reason="serveur LLM (Ollama) injoignable"))
        if "gemini" in item.keywords and not os.environ.get("GEMINI_API_KEY"):
            item.add_marker(pytest.mark.skip(reason="GEMINI_API_KEY absente"))
