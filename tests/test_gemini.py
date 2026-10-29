"""Appel Gemini de test (texte court) : un vecteur de la dimension demandée.

Nécessite GEMINI_API_KEY ; ignoré sinon (voir conftest).
"""
import pytest


@pytest.mark.gemini
def test_gemini_text_embedding_dimension():
    import numpy as np
    from google import genai
    from google.genai import types
    from core.config import get_settings

    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    resp = client.models.embed_content(
        model=s.gemini_model,
        # gemini-embedding-2 : pas de task_type, l'instruction est un préfixe texte (doc officielle)
        contents="task: search result | query: a slow piano ballad with a female voice",
        config=types.EmbedContentConfig(output_dimensionality=s.gemini_dim),
    )
    vec = np.asarray(resp.embeddings[0].values, dtype="float32")
    assert vec.shape == (s.gemini_dim,)
    assert np.isfinite(vec).all()
