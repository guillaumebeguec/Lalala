# SoundTrace

Un « Shazam amélioré » : un agent qui retrouve une chanson à partir d'un souvenir flou, d'une reprise ou d'un enregistrement bruité. Voir `CLAUDE.md` (projet), `PLAN.md` (phases) et `DEMO.md` (scénario).

## Installation

```bash
# Python 3.11 + dépendances
uv sync

# Chromaprint (empreinte audio type Shazam, binaire fpcalc) et Ollama (Llama local)
brew install chromaprint ollama
ollama serve &            # ou lancer l'application Ollama
ollama pull llama3.1:8b

# Configuration
cp .env.example .env      # puis renseigner GEMINI_API_KEY, SEARCH_API_KEY

# Vérification
uv run python scripts/check_setup.py
uv run pytest
```

Les tests marqués `llm` et `gemini` sont ignorés automatiquement si Ollama n'est pas joignable ou si `GEMINI_API_KEY` est absente.

## Données externes et envoi vers des API tierces

- **Gemini Embedding 2** (Google) est un embedder parmi d'autres, désactivable avec `USE_GEMINI=false` sans casser le système (PE-AV et CLAP prennent le relais). Quand il est actif, sont envoyés à Google : les extraits de 30 s du catalogue (pour l'index) et, en démo, les enregistrements et descriptions de l'utilisateur. Les réponses sont mises en cache localement par hash du contenu.
- Les extraits audio de 30 s (Deezer, iTunes en repli) sont mis en cache localement dans `data/` (ignoré par git), jamais redistribués. Aucune parole complète n'est stockée.

*(Sections à compléter au fil des phases : données, évaluation, démo.)*
