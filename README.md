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
- **Paroles chantées** : quand l'entrée est une voix seule, Whisper (faster-whisper, local) transcrit les mots chantés ; le fragment est relié au catalogue par des **empreintes de paroles non réversibles** (hachés 64 bits de 4-grammes de mots calculés depuis le corpus WASABI, `scripts/build_lyrics_hashes.py`), par Llama et par les fiches. Aucun texte de paroles n'est conservé : les hachés ne permettent pas de reconstituer les paroles.

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

## Réglages du pipeline (variables d'environnement)

| Variable | Défaut | Effet |
|---|---|---|
| `EMBEDDERS` | `clap,peav,gemma,gemini,coverhunter,hum2song` | embedders actifs (un index absent est simplement ignoré) |
| `SOUNDTRACE_TEXT_TO_AUDIO` | `auto` | `auto` : texte→audio seulement si la réécriture détecte une description sonore ; `always` / `never` |
| `SOUNDTRACE_LLM_RETRIEVER` | `1` | Llama comme moteur de recherche (titres proposés puis résolus dans le catalogue) |
| `LLM_RETRIEVER_MODEL` | `LLM_MODEL` | modèle dédié à cette branche (un Llama plus gros par API y change tout) |
| `SOUNDTRACE_LLM_MIN_CONF` | `0.6` | confiance minimale d'une proposition Llama pour voter |
| `SOUNDTRACE_RERANK` | `1` | reranking Llama (cross-encodeur) ; `0` pour la latence en direct |
| `RERANK_MODE` | `pointwise` | `listwise` : un seul appel classe tous les candidats (modèles plus gros) |
| `HUM2SONG_DEMUCS` / `HUM2SONG_WINDOWS` | `0` / `0` | côté morceau de hum2song : voix séparée (Demucs), fenêtres glissantes |
| `SOUNDTRACE_ASR_MODEL` | `small` | modèle Whisper (faster-whisper, CPU int8) pour les paroles chantées ; `base` plus rapide, `medium` plus juste |
| `SOUNDTRACE_FUSION` | `rrf` | `learned` : fusion apprise (`scripts/train_fusion.py`) si le modèle existe |
| `PREVIEW_STORAGE` | `memory` | aucun audio écrit sur disque (voir ci-dessus) |
| `SAM_AUDIO_ENGINE` | `auto` | isolation de la musique (SAM Audio) sur les enregistrements bruités : `mlx`, `torch`, `off` (voir § SAM Audio) |
| `OMP_NUM_THREADS` / `KMP_DUPLICATE_LIB_OK` | `1` / `TRUE` | posés par `core/__init__.py` : double runtime OpenMP faiss + torch sur macOS |

Architecture visée (après les recommandations du 28/10) : texte → réécriture → [Llama-retriever + fiches (BM25 + dense) + web] → fusion (RRF, ou apprise : `scripts/train_fusion.py`) → reranker. Audio → empreinte robuste (audfprint, repli Chromaprint) → sinon [embedder audio + CoverHunter (+ hum2song si voix seule)] → fusion à seuils → reranker si une note texte accompagne l'audio.

## SAM Audio (isolation de la musique, phase 9)

Le paquet officiel `sam-audio` dépend de `perception_models`, qui installe un paquet Python nommé `core`
(conflit avec `core/` du projet) et de `decord` / `xformers` sans roue macOS. Il est donc installé dans
un venv **séparé** et appelé en sous-processus (`core/isolate.py` → `scripts/sam_runner*.py`), jamais
importé dans le processus de l'API. Deux moteurs :

- **MLX** (recommandé sur Apple Silicon) : `uv venv data/raw/sam_audio/venv_mlx --python 3.11 && uv pip install --python data/raw/sam_audio/venv_mlx/bin/python -U mlx-audio` ; poids `mlx-community/sam-audio-base` (conversion des poids Meta).
- **Officiel (PyTorch)** : `git clone https://github.com/facebookresearch/sam-audio data/raw/sam_audio/repo`, puis le script `scripts/sam_install.sh` (eva-decord à la place de decord, `--no-deps` pour perception-models, stub `xformers`, `transformers<5`, `huggingface_hub<1`, `setuptools`). Poids `facebook/sam-audio-base` à accès restreint : `HF_TOKEN` dans `.env`.

Dans l'app, l'étape « Isolating the music » ne s'exécute que si l'enregistrement est détecté bruité, sans
empreinte exacte et sans être une voix seule ; l'audio isolé remplace l'original pour l'empreinte (seconde
chance) et la similarité. Résultats en cache (`data/cache/sam_audio/`).

## Fredonnement : appariement mélodique direct

hum2song (embeddings de contours) rate même des fredonnements synthétiques rendus depuis la mélodie de
l'extrait ; la branche « Melody matching » repose donc sur `core/melody.py` : contour RMVPE de la requête
aligné sur les contours des extraits (`scripts/build_contours.py` → `data/index/contours_<subset>.npz`),
avec tolérance à la transposition, au tempo (0,8–1,25) et aux silences. hum2song reste un vote secondaire.
Chaque morceau peut avoir **deux fenêtres de 30 s** : l'extrait Deezer et l'extrait iTunes (souvent un autre passage) ;
`scripts/build_contours.py --provider itunes` écrit `contours_<subset>_itunes.npz`, chargé et fusionné automatiquement
(meilleur alignement par morceau). Sur les 393 fredonnements CHAD : 19,6 % top-1 / 29,8 % top-10 (voir `eval/retrieval.md`).
La confiance affichée vient de la marge top-1/top-2 (calibrée sur CHAD, `data/melody_calib_chad.csv` : ≥ 0,05 → 77 % de précision, ≥ 0,03 → 50 %).

Jeux de fredonnements : T6a = CHAD (réels) ; T6b = `data/manual/hums/` (réels, `scripts/record_queries.py --mode hum`)
et `data/manual/hums_synth/` (synthétiques, `scripts/synth_hums.py`). Enregistrements réels bruités pour T3 :
`scripts/record_queries.py --mode rerecord` (lecture dans les haut-parleurs + capture micro).

*(Sections à compléter au fil des phases : données, évaluation, démo.)*
