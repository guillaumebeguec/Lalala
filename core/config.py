"""Central configuration read from `.env` (see `.env.example`).

No hard-coded key: everything comes from the environment. Paths are resolved to
absolute paths and data directories are created on demand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    # LLM (Llama via Ollama or a hosted API, OpenAI protocol)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.1:8b"
    llm_api_key: str = "ollama"

    # Gemini Embedding 2: one embedder among others, can be disabled
    use_gemini: bool = True
    gemini_api_key: str = ""
    gemini_dim: int = 768
    gemini_model: str = "gemini-embedding-2"

    # Web search
    search_provider: str = "tavily"
    search_api_key: str = ""

    # Active embedders
    embedders: list[str] = field(default_factory=lambda: ["gemma", "peav", "gemini", "coverhunter", "hum2song"])

    # Data
    data_dir: Path = PROJECT_ROOT / "data"

    # ---- derived paths ----
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def manual_dir(self) -> Path:
        return self.data_dir / "manual"

    @property
    def testsets_dir(self) -> Path:
        return self.data_dir / "testsets"

    @property
    def previews_dir(self) -> Path:
        return self.data_dir / "previews"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def active_embedders(self) -> list[str]:
        """Embedders actually usable: `gemini` is removed if USE_GEMINI=false
        or if the key is missing, without breaking the rest of the system."""
        active = list(self.embedders)
        if "gemini" in active and (not self.use_gemini or not self.gemini_api_key):
            active.remove("gemini")
        return active

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.cache_dir, self.manual_dir, self.testsets_dir,
                  self.previews_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load `.env` (without overriding the existing environment) and build the settings."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    env = os.environ

    data_dir = Path(env.get("DATA_DIR", "./data"))
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()

    return Settings(
        llm_base_url=env.get("LLM_BASE_URL", Settings.llm_base_url),
        llm_model=env.get("LLM_MODEL", Settings.llm_model),
        llm_api_key=env.get("LLM_API_KEY", Settings.llm_api_key),
        use_gemini=_as_bool(env.get("USE_GEMINI"), True),
        gemini_api_key=env.get("GEMINI_API_KEY", ""),
        gemini_dim=int(env.get("GEMINI_DIM", Settings.gemini_dim)),
        gemini_model=env.get("GEMINI_MODEL", Settings.gemini_model),
        search_provider=env.get("SEARCH_PROVIDER", Settings.search_provider),
        search_api_key=env.get("SEARCH_API_KEY", ""),
        embedders=_as_list(env.get("EMBEDDERS"), ["gemma", "peav", "gemini", "coverhunter", "hum2song"]),
        data_dir=data_dir,
    )
