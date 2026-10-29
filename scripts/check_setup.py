"""Vérifie l'environnement de la phase 0 : device, LLM, Gemini, Chromaprint, clés.

    uv run python scripts/check_setup.py

Ne modifie rien ; chaque ligne dit OK / ABSENT / ERREUR avec la cause.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings  # noqa: E402


def line(label: str, ok: bool | None, detail: str = "") -> None:
    status = {True: "OK     ", False: "ERREUR ", None: "ABSENT "}[ok]
    print(f"[{status}] {label:<28} {detail}")


def main() -> int:
    s = get_settings()
    s.ensure_dirs()
    print(f"DATA_DIR = {s.data_dir}\n")

    # Device
    try:
        from core.device import device_name
        line("device PyTorch", True, device_name())
    except Exception as e:  # noqa: BLE001
        line("device PyTorch", False, str(e))

    # LLM
    from core import llm
    if llm.is_available():
        try:
            r = llm.chat([{"role": "user", "content": "Réponds par un seul mot : OK"}], max_tokens=5)
            line("LLM " + s.llm_model, True, f"{s.llm_base_url} → {r.content.strip()[:30]!r}")
        except Exception as e:  # noqa: BLE001
            line("LLM " + s.llm_model, False, f"{s.llm_base_url} : {e}")
    else:
        line("LLM " + s.llm_model, None, f"{s.llm_base_url} injoignable (ollama serve ; ollama pull {s.llm_model})")

    # Gemini
    if not s.use_gemini:
        line("Gemini", None, "USE_GEMINI=false (désactivé, le système tourne sans)")
    elif not s.gemini_api_key:
        line("Gemini", None, "GEMINI_API_KEY absente dans .env")
    else:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=s.gemini_api_key)
            resp = client.models.embed_content(
                model=s.gemini_model, contents="task: search result | query: test",
                config=types.EmbedContentConfig(output_dimensionality=s.gemini_dim))
            dim = len(resp.embeddings[0].values)
            line("Gemini " + s.gemini_model, dim == s.gemini_dim, f"dimension {dim} (attendu {s.gemini_dim})")
        except Exception as e:  # noqa: BLE001
            line("Gemini " + s.gemini_model, False, str(e)[:200])

    # Chromaprint
    fpcalc = shutil.which("fpcalc")
    line("Chromaprint fpcalc", bool(fpcalc) or None, fpcalc or "brew install chromaprint")

    # Recherche web
    line(f"recherche web ({s.search_provider})", bool(s.search_api_key) or None,
         "clé présente" if s.search_api_key else "SEARCH_API_KEY absente")

    print(f"\nEmbedders configurés : {s.embedders} ; actifs : {s.active_embedders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
