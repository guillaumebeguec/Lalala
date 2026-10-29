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

- **EmbeddingGemma 2** (`google/embeddinggemma-2`, Apache 2.0) est un embedder multimodal ouvert qui tourne **en local** (texte, audio, vidéo, 100+ langues) : rien ne quitte la machine.
- **Gemini Embedding 2** (Google, API) est un embedder parmi d'autres, désactivable avec `USE_GEMINI=false` sans casser le système (PE-AV et CLAP prennent le relais). Quand il est actif, sont envoyés à Google : les extraits de 30 s du catalogue (pour l'index) et, en démo, les enregistrements et descriptions de l'utilisateur. Les réponses sont mises en cache localement par hash du contenu.
- **Extraits audio de 30 s** : URL d'extrait obtenues via l'API Deezer (iTunes Search API en repli). Les conditions Deezer (`developers.deezer.com/termsofuse`, consultées le 28/10/2026) imposent un usage non commercial et privé ; les *guidelines* développeurs interdisent le stockage local ou hors-ligne de l'audio. Par défaut (`PREVIEW_STORAGE=memory`), **aucun fichier audio n'est écrit sur disque** : les extraits sont lus en mémoire le temps de calculer embeddings et empreintes, et seuls les vecteurs sont conservés ; l'écoute dans l'interface passe par l'URL d'extrait officielle. `PREVIEW_STORAGE=disk` active un cache local dans `data/previews/` (ignoré par git) et n'est à utiliser que si les conditions le permettent.
- **Métadonnées** : WASABI Song Corpus 2.0 (Zenodo 10.5281/zenodo.5603369, CC BY-NC 4.0 ; citer Buffa et al., ESWC 2021), sans paroles ; API Deezer et Wikipédia (résumés courts) ; ToT_Music (github.com/spotify-research/tot, CC0-1.0) ; MUSAN (openslr.org/17, CC BY 4.0) pour le bruit.
- Aucune parole complète n'est stockée (le champ `lyrics` de WASABI est ignoré à l'extraction).

*(Sections à compléter au fil des phases : données, évaluation, démo.)*
