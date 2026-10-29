from core.config import get_settings


def test_defaults_and_data_dir(tmp_path):
    s = get_settings()
    assert s.llm_model
    assert s.data_dir.is_absolute()
    assert s.cache_dir == s.data_dir / "cache"


def test_gemini_disabled_removes_embedder(monkeypatch):
    from core import config
    monkeypatch.setenv("USE_GEMINI", "false")
    monkeypatch.setenv("EMBEDDERS", "clap,peav,gemini")
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.active_embedders == ["clap", "peav"]


def test_gemini_without_key_is_inactive(monkeypatch):
    from core import config
    monkeypatch.setenv("USE_GEMINI", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDERS", "clap,gemini")
    config.get_settings.cache_clear()
    assert config.get_settings().active_embedders == ["clap"]
