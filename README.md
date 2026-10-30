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
- **Compléments (PLAN_COMPLEMENTS.md)** : CHAD (Amatov et al., ISMIR 2023) — sous-ensemble fredonnements `amanteur/CHAD_hummings` (CC BY-NC 4.0) ; les originaux et reprises YouTube ne sont **pas** téléchargés, seuls les titres publics des vidéos (oEmbed) servent à l'appariement. CoverHunter — code `alanngnet/CoverHunterMPS` et checkpoint original (SHS100K, Google Drive) **sans licence déclarée** : usage expérimental local. hum2song (`luigir5/hum2song`, poids CC BY 4.0), prévu en phase 9. SAM Audio (`facebook/sam-audio-*`, SAM License) : checkpoints **gatés** sur Hugging Face, accès à demander ; documenté pour CUDA.
- Les URL d'extrait Deezer sont signées et expirent en moins d'une heure : elles sont rafraîchies à la demande via `/track/{id}`.

## Front « Lalala » (FRONT_PLAN.md)

```bash
cd web && pnpm install
pnpm dev                      # mode mock (fixtures, sans backend) sur http://localhost:5173
VITE_API_MODE=live pnpm dev   # branché sur l'API FastAPI (proxy /api → :8000)
pnpm test                     # Vitest ; pnpm e2e : Playwright (parcours mock)
pnpm build                    # web/dist, servi par `uv run uvicorn api.main:app` sur /
pnpm dev:lan                  # exposé sur le réseau local en HTTPS auto-signé (micro OK sur téléphone), mode live
pnpm dev:lan:mock             # idem en mode mock
```

Pour un test depuis un téléphone : lancer le backend (`SOUNDTRACE_SUBSET=dev uv run uvicorn api.main:app --port 8000`), puis `pnpm dev:lan` et ouvrir `https://<ip-du-mac>:5173/` (accepter le certificat).

Options d'URL : `?replay=memory|cover|noisy` (scène de secours), `?scale=1.25` (projection), `?accent=mint|violet`.
Appui long (1 s) sur le logo : menu de rejeu.

*(Sections à compléter au fil des phases : données, évaluation, démo.)*
