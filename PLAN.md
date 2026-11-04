# PLAN.md — SoundTrace

Plan d'exécution. Le **pourquoi** (scénario de démo, choix des données, jeux de test) est dans `DEMO.md` : le lire avant les phases 1, 2 et 10.

Chaque phase a un objectif, des tâches et des **critères de validation**. Ne pas passer à la suivante tant qu'ils ne sont pas remplis. En cas de blocage (API différente de la doc, accès refusé, conditions d'utilisation incompatibles), s'arrêter et proposer une alternative plutôt que contourner en silence.

## Calendrier indicatif (1 semaine, 1 personne)

| Jour | Phases |
|---|---|
| J1 | 0, 1 (catalogue) |
| J2 | 2 (jeux de test), 3 (embeddings + index) |
| J3 | 4 (évaluation + fusion), 5 (réécriture) |
| J4 | 6 (agent baseline + reranking Llama) |
| J5 | 7 (gain d'information) |
| J6 | 8 (API + front), 9.1 ou 9.2 (scène 3) |
| J7 | 10 (démo, répétitions, plan B) |

## Briques de modélisation (🧪)

| Brique | Phase | Statut |
|---|---|---|
| Fusion multi-embedders (RRF) | 4 | Obligatoire |
| Réécriture de requête, comparée à la requête brute | 5 | Obligatoire |
| Reranking par Llama (cross-encodeur LLM) | 6.3 | Obligatoire |
| Questions par gain d'information | 7 | Obligatoire (cœur de la scène 1) |
| Adaptateur contrastif sur encodeur gelé | 9.3 | Optionnel |

Chaque brique garde une baseline simple, implémentée d'abord, qui sert de comparaison et de repli.

---

## Arborescence cible

```
.
├── PLAN.md
├── DEMO.md
├── pyproject.toml
├── .env.example
├── core/
│   ├── device.py
│   ├── config.py
│   ├── llm.py              # client OpenAI-compatible vers Llama
│   ├── audio_io.py         # chargement, rééchantillonnage, mixage avec bruit
│   ├── cache.py            # cache disque par hash (appels API, embeddings)
│   ├── embedders/
│   │   ├── base.py
│   │   ├── clap.py
│   │   ├── peav.py
│   │   └── gemini.py
│   ├── index.py            # FAISS (audio, texte) + BM25
│   ├── fingerprint.py      # Chromaprint (baseline type Shazam)
│   ├── fusion.py           # 🧪 RRF
│   ├── query_rewrite.py    # 🧪
│   ├── rerank.py           # 🧪 reranking Llama
│   ├── isolate.py          # SAM Audio
│   └── adapter.py          # 🧪 optionnel
├── catalog/
│   ├── wasabi.py           # chargement et filtrage WASABI
│   ├── previews.py         # extraits Deezer / iTunes
│   ├── cards.py            # fiches morceaux texte
│   └── build.py            # assemblage du catalogue
├── testsets/
│   ├── tot_music.py        # T1
│   ├── fr_memories.py      # T2
│   ├── noisy.py            # T3
│   ├── covers.py           # T4
│   └── video.py            # T5
├── finder/
│   ├── tools.py
│   ├── agent.py
│   ├── belief.py           # 🧪 gain d'information
│   ├── simulated_user.py
│   └── prompts.py
├── api/main.py
├── web/index.html
├── scripts/                # un script par étape rejouable
├── eval/                   # rapports chiffrés (markdown + csv)
├── data/                   # ignoré par git
└── tests/
```

---

## Phase 0 — Mise en place (≈ 1 h)

Tâches :
1. Projet `uv` (Python 3.11). Dépendances : torch, torchaudio, transformers, faiss-cpu, rank-bm25, librosa, soundfile, numpy, pandas, pyarrow, google-genai, openai, httpx, tenacity, fastapi, uvicorn, python-multipart, python-dotenv, pyacoustid, pytest. Chromaprint (`fpcalc`) installé via Homebrew (documenter).
2. `core/device.py`, `core/config.py`, `.env.example` : `LLM_BASE_URL`, `LLM_MODEL=llama3.1:8b`, `LLM_API_KEY`, `USE_GEMINI=true`, `GEMINI_API_KEY`, `GEMINI_DIM=768`, `SEARCH_PROVIDER`, `SEARCH_API_KEY`, `EMBEDDERS=clap,peav,gemini`, `DATA_DIR=./data`.
3. `core/llm.py` : `chat(messages, tools=None, json_schema=None)`, parseur JSON tolérant + une relance si JSON invalide.
4. `core/cache.py` : cache disque clé = hash(contenu + paramètres), utilisé par tous les appels coûteux.
5. `.gitignore` : `data/`, `.env`, caches.

**Validation** : `chat` répond ; test `@pytest.mark.llm` sur un `tool_call` factice et une sortie JSON ; un appel Gemini de test (texte court) renvoie un vecteur de la dimension demandée.

---

## Phase 1 — Catalogue grand public (≈ 1 jour)

Voir `DEMO.md` § 3.2 pour le pourquoi.

Tâches :
1. `catalog/wasabi.py` : télécharger le **WASABI Song Corpus** (lire la page officielle pour l'URL, le format et la licence ; ne pas inventer). Extraire : identifiant, titre, artiste, album, année, genre, identifiants externes (Deezer si présent, MusicBrainz). **Signaler** si les identifiants Deezer sont absents : on passera par la recherche Deezer.
2. `testsets/tot_music.py` (récupération seule à ce stade) : télécharger **ToT_Music** (Bhargav et al., SIGIR 2023 ; lien depuis l'article arXiv 2305.14072 ou le dépôt de code associé). Extraire la liste des morceaux réponses.
3. `catalog/build.py` : composer le catalogue, en ordre de priorité :
   a. tous les morceaux réponses de ToT_Music ;
   b. tubes par décennie 1970–2025, France et international (playlists éditoriales et charts via l'API Deezer) ;
   c. `data/manual/ads_series_fr.csv` : musiques de pubs et génériques (je fournirai la liste ; préparer le fichier vide avec les colonnes `titre, artiste, usage, annee`) ;
   d. distracteurs : autres titres des mêmes artistes et genres.
   Dédoublonner (même titre + artiste normalisés, versions remasterisées). Cible : 20 000 à 30 000 morceaux. Sous-ensemble **dev** de 2 000 morceaux contenant 300 réponses ToT_Music.
4. `catalog/previews.py` : pour chaque morceau, récupérer l'extrait de 30 s via l'**API Deezer** (identifiant direct ou recherche `artist:"…" track:"…"`), repli **iTunes Search API**. Vérifier la correspondance titre/artiste (similarité de chaînes) pour éviter les mauvais appariements. Respecter les limites de débit (`tenacity`, backoff). Cache local, jamais commité. Journaliser le taux de couverture.
5. `catalog/cards.py` : fiche morceau texte = titre, artiste, année, genre, résumé Wikipédia court (API Wikipédia, FR puis EN), usages connus si présents dans la liste manuelle, + **description audio** générée une fois par Llama à partir des métadonnées (marquée comme générée). Pas de paroles.

**Validation** :
- Rapport `eval/catalog.md` : nombre de morceaux, couverture des extraits (objectif > 85 %), répartition par décennie et par langue, part des réponses ToT_Music couvertes (objectif > 70 %).
- Contrôle manuel de 30 appariements extrait ↔ métadonnées : 0 erreur grossière tolérée sur l'échantillon, sinon durcir le filtre.

---

## Phase 2 — Jeux de test (≈ ½ jour)

Voir `DEMO.md` § 3.3. Tous les jeux sont figés dans `data/testsets/*.jsonl` avec une graine fixe.

Tâches :
1. **T1 ToT_Music** : ne garder que les demandes dont la réponse est dans le catalogue avec extrait. Échantillon d'éval de 200 (stratifié par décennie), le reste en réserve. Champs : `id, texte, reponse_id`.
2. **T2 Souvenirs FR** : fichier `data/manual/fr_memories.csv` (`description, titre, artiste`) que je remplirai ; script de validation qui vérifie que chaque réponse existe dans le catalogue.
3. **T3 Bruit** : télécharger **MUSAN** (lire la page officielle) ; mixer 100 extraits du catalogue avec bruit et brouhaha à SNR 0 / 5 / 10 dB, fenêtres de 10 s à position aléatoire. Dossier `data/manual/real_recordings/` pour mes 20 vrais enregistrements (nommés `<track_id>.m4a`).
4. **T4 Reprises** : pour 300 tubes du catalogue, chercher sur Deezer « titre + piano », « titre + acoustic cover », « titre + live », « titre + karaoke ». Garder les résultats dont le titre correspond et dont l'artiste **diffère** (ou mention live). Récupérer leur extrait. Cible : 200 paires reprise → original. Contrôle manuel de 20 paires.
5. **T5 Vidéo** : dossier `data/manual/video/` (je fournirai ~20 extraits `<track_id>.mp4`).
6. `finder/simulated_user.py` : utilisateur simulé conforme à `DEMO.md` § 3.3 (ne voit que le texte d'origine ; variante « informé » qui connaît aussi année et langue). **Test** qui vérifie que la réponse attendue n'apparaît jamais dans son prompt.

**Validation** : statistiques de chaque jeu dans `eval/testsets.md` ; test anti-fuite au vert.

---

## Phase 3 — Embeddings et index (≈ ½ jour + calcul)

Tâches :
1. `core/embedders/base.py` : `embed_audio(items)`, `embed_text(texts, task="query"|"document")`, `embed_video(path)` (optionnel selon l'embedder, sinon `NotImplementedError`), sorties normalisées L2.
2. `clap.py` : `laion/larger_clap_music` via `transformers`, MPS, batching. Note : anglais uniquement, les requêtes françaises passent par une traduction Llama.
3. `peav.py` : PE-AV depuis `facebookresearch/perception_models` ; **lire le README et les exemples** ; audio, texte, vidéo. Si les poids exigent un accès, s'arrêter et prévenir.
4. `gemini.py` : `gemini-embedding-2` via le SDK `google-genai`. **Lire la doc officielle** pour : limite audio par requête, formats acceptés (mp3/wav), instructions de tâche (préfixes du type `task: … | query: …`), dimension réduite par MRL (`GEMINI_DIM`), API batch. Cache par hash, batch pour le catalogue, appels unitaires pour les requêtes.
5. `core/index.py` : un index FAISS par (embedder × modalité) — audio des extraits, texte des fiches — et un index BM25 sur les fiches. API : `search(query, embedder, modality, k)`.
6. `core/fingerprint.py` : empreintes Chromaprint des extraits du catalogue et recherche de la meilleure correspondance (baseline type Shazam).
7. Scripts : `build_embeddings.py --embedder … --subset {dev,full}`, `build_index.py`, `build_fingerprints.py`. Reprise si interrompu.

**Validation** : les trois embedders tournent sur le dev ; temps et coût (Gemini) du dev reportés ; projection du coût et du temps sur le catalogue complet avant de le lancer, **à valider avec moi**.

---

## Phase 4 — 🧪 Évaluation, fusion et baselines (≈ ½ jour)

Tâches :
1. `scripts/eval_retrieval.py` : sans agent, pour chaque jeu de test :
   - T1, T2 (texte → morceau) : BM25 fiches ; embedding texte → fiches ; embedding texte → audio ; pour chaque embedder.
   - T3, T4 (audio → morceau) : Chromaprint ; embedding audio → audio pour chaque embedder.
   - T5 (vidéo → morceau) : PE-AV et Gemini.
   Métriques : top-1, top-5, top-10, MRR.
2. `core/fusion.py` : RRF (`k` ∈ {10, 30, 60}, variante pondérée) entre toutes les listes disponibles d'une même requête.
3. `eval/retrieval.md` : un tableau par jeu, avec baselines, chaque embedder seul et la fusion ; 5 exemples qualitatifs ; recommandation de configuration.

**Validation** :
- Le rapport tranche la configuration par défaut de chaque modalité.
- **Décision scène 2** : si le meilleur top-5 sur T4 (reprises) est < 50 %, le signaler clairement ; la scène 2 sera remplacée (voir `DEMO.md`).
- Test unitaire de `rrf`.

---

## Phase 5 — 🧪 Réécriture de requête (≈ ½ jour)

Tâches :
1. `core/query_rewrite.py` : Llama produit, en JSON, à partir d'une description floue :
   - 3 à 5 **légendes audio** (instruments, tempo approximatif, voix, ambiance, production), en anglais ;
   - des **indices structurés** : décennie probable, langue des paroles, support (pub, série, film, jeu, radio), genre, fragments de paroles éventuels ;
   - une **requête texte** nettoyée pour BM25 et les fiches.
   Interdiction d'inventer des titres.
2. Comparer sur T1 et T2 : requête brute seule ; réécriture seule ; **fusion brute + réécriture**. L'article ToT_Music a observé que les reformulations LLM faisaient moins bien que la requête brute : on mesure, on ne suppose pas.
3. Les indices structurés servent de **filtres doux** (bonus de score sur la décennie et la langue), pas de filtres durs.

**Validation** : résultats ajoutés à `eval/retrieval.md` ; la configuration retenue est celle qui gagne, même si c'est la requête brute.

---

## Phase 6 — Agent baseline et reranking (≈ 1 jour)

### 6.1 Outils — `finder/tools.py`
- `catalog_search(text)` : recherche hybride (fiches BM25 + embeddings texte + texte → audio, avec réécriture), fusion, renvoie les 20 meilleurs candidats avec leur fiche.
- `audio_match(audio)` : embedding audio (+ isolation en phase 9), fusion, et Chromaprint en parallèle (si l'empreinte matche fort, c'est l'enregistrement exact : réponse immédiate).
- `web_search(query)` : Tavily ou Brave, cache ; pour les usages (pubs, séries) et les fragments de paroles.
- `ask_user(question, options)` : une question fermée, 2 à 4 options + « je ne sais pas ».
- `propose(candidates)` : 1 à 3 candidats `{titre, artiste, annee, confiance, raison, preview_url}`.

### 6.2 Boucle — `finder/agent.py`
- État explicite : entrée, indices, candidats, requêtes faites.
- À chaque tour, Llama choisit : chercher, demander, proposer. Garde-fous : 3 questions, 8 appels d'outils, pas de requête répétée, pas de titre sans source (catalogue ou web).
- Un candidat trouvé sur le web mais **absent du catalogue** est proposé avec un lien de recherche (pas d'extrait), et marqué comme tel.
- Événements émis pour le front : indice, recherche, candidats, question, proposition.

### 6.3 🧪 Reranking par Llama — `core/rerank.py`
- Les embeddings sont des bi-encodeurs : requête et morceau sont encodés séparément. Le reranking ajoute une étape où le LLM voit **la requête et la fiche ensemble**, comme un cross-encodeur.
- Entrée : description + indices + 20 candidats (fiches courtes). Sortie JSON : un score de 0 à 10 par candidat et une justification courte. Par lots de 5 pour tenir dans le contexte du 8B. Score final = combinaison de la fusion et du score Llama (poids réglé sur un sous-ensemble de T1, évalué sur le reste).
- **Validation** : gain en top-1 / top-3 sur T1 et T2 par rapport à la fusion seule, reporté dans `eval/retrieval.md`.

### 6.4 Évaluation de l'agent — `scripts/eval_finder.py`
- Rejoue T1 (200) et T2 (50) avec l'utilisateur simulé (deux variantes). Mesures : top-1, top-3, nombre de questions, appels d'outils, temps, part des réponses trouvées via le web contre via le catalogue.

**Validation** : pas de boucle infinie ; `eval/finder.md` rempli ; l'agent fait mieux que la meilleure recherche sans agent (phase 5) sur T1 en top-3, sinon le signaler.

---

## Phase 7 — 🧪 Questions par gain d'information (≈ 1 jour)

`finder/belief.py` :
1. **Croyance** : les candidats issus de la recherche (top-20 après reranking) + une hypothèse « autre ». Prior = scores normalisés (softmax avec température réglée sur T1), pas une estimation Llama libre.
2. **Questions candidates** : un **répertoire fixe** de questions dont la réponse se lit dans les fiches (décennie, langue, voix masculine/féminine/instrumental, genre, tempo lent/rapide, support pub/série/film) + 2 questions libres proposées par Llama.
3. **Modèle d'observation** :
   - questions du répertoire : `P(réponse | h)` calculé **depuis les métadonnées** de la fiche (exact, rapide, sans LLM), lissé par `ε` pour les erreurs de mémoire de l'utilisateur (les demandes ToT contiennent souvent de faux souvenirs) ;
   - questions libres : estimées par Llama en JSON, puis lissées.
   - « je ne sais pas » : peu informatif.
4. **Choix** : maximiser `IG(q) = H(P) − Σ_r P(r)·H(P | r)`.
5. **Mise à jour** : Bayes ; si « autre » dépasse 0,3, relancer une recherche (catalogue + web) avec les nouveaux indices et réinjecter les candidats.
6. **Arrêt** : `max P(h) > 0,7` ou 3 questions.
7. Exposer à chaque tour : entropie, distribution, question choisie et son gain attendu.
8. Régler `ε` et la température sur 50 cas de T1, évaluer sur les 150 autres.

**Validation** : comparaison baseline (6.2) contre gain d'information sur T1 et T2 (top-1, top-3, nombre moyen de questions) dans `eval/finder_ig.md`, avec intervalles de confiance par bootstrap. L'entropie baisse après au moins 80 % des réponses. Si le gain d'information ne fait pas mieux, garder la baseline et le noter.

---

## Phase 8 — API et front (≈ ½ à 1 jour)

API (`api/main.py`) :
- `POST /session` ; `POST /session/{id}/message` (texte ou réponse à une question), réponse en SSE ;
- `POST /session/{id}/audio` (enregistrement micro) ; `POST /session/{id}/video` ;
- `GET /preview/{track_id}` (extrait en cache local) ;
- `POST /replay/{nom}` : rejoue une session enregistrée (plan B de démo) ; toute session peut être enregistrée sous un nom.

Front (`web/index.html`, HTML + JS vanilla, thème clair/sombre, lisible au projecteur) :
- conversation ; questions en boutons ; bouton micro (MediaRecorder, 10 s) ; dépôt vidéo ;
- panneau « enquête » : indices, hypothèses en barres de probabilité, **courbe d'entropie**, requêtes effectuées, source de chaque candidat (catalogue, web, empreinte) ;
- cartes candidats avec lecteur d'extrait et bouton « c'est celui-là » / « non » (un « non » relance l'enquête) ;
- pour la scène 3 : lecteur avant/après isolation.

**Validation** : parcours complets des scènes 1 et 2 de `DEMO.md` sans erreur console, sur Chrome et Safari ; micro fonctionnel.

---

## Phase 9 — Scène 3 et options (≈ 1 jour)

1. **SAM Audio** (`core/isolate.py`) : isoler la musique (consigne « music ») d'un enregistrement bruité avant `audio_match`. Lire le README officiel ; mesurer la latence sur MPS. Évaluer sur T3 avec et sans isolation, par SNR → `eval/isolation.md`.
2. **Vidéo** : `embed_video` via PE-AV et Gemini sur T5 ; en complément, Llama décrit la scène (si un Llama vision est disponible) pour orienter la recherche web. → `eval/video.md`.
3. 🧪 **Adaptateur contrastif** (optionnel) : projection entraînée sur encodeur gelé (InfoNCE symétrique, température apprise), paires audio ↔ fiches + légendes, découpage par artiste, évalué sur T1/T2 jamais vus. À abandonner s'il ne gagne pas sur T1.
4. **Llama plus gros** via API hébergée : comparaison sur T1.
5. **Fredonnement** (expérimental) : contour de hauteur + DTW sur les extraits. Ne pas bloquer la démo dessus.

**Validation** : décision scène 3 (bruit ou vidéo) prise sur les chiffres de 9.1 et 9.2, notée dans `DEMO.md`.

---

## Phase 10 — Démo (≈ ½ jour)

1. Suivre le storyboard de `DEMO.md`. Enregistrer une session de secours par scène (`/replay`).
2. Préchauffer : Ollama, embedders locaux, SAM Audio ; préremplir le cache Gemini des cas répétés ; vérifier le fonctionnement **sans réseau** des cas de secours.
3. Diapo résultats générée depuis `eval/` (`scripts/make_results_slide.py` → un tableau PNG/SVG : SoundTrace contre BM25, embedding brut, Chromaprint).
4. `README.md` : installation, données externes et leurs conditions, ce qui est envoyé à Gemini, désactivation avec `USE_GEMINI=false`.

---

## Risques connus

| Risque | Parade |
|---|---|
| Conditions Deezer / iTunes incompatibles | Les vérifier en J1 ; usage local, non commercial, sans redistribution ; repli possible sur un catalogue réduit |
| Règles du hackathon défavorables aux modèles non-Meta | `USE_GEMINI=false` : PE-AV + CLAP prennent le relais ; les chiffres de la phase 4 montrent ce qu'on perd |
| Reprises mal retrouvées (T4 < 50 % en top-5) | Remplacer la scène 2 par la vidéo ; ne pas promettre les reprises |
| Réponses ToT_Music mal couvertes par le catalogue | Prioriser leur récupération en phase 1 ; évaluer seulement sur les cas couverts, en le disant |
| Llama 8B faible en appel d'outils ou en JSON | Sortie contrainte + parseur tolérant ; Llama plus gros via API |
| Faux souvenirs de l'utilisateur | Lissage `ε`, filtres doux, hypothèse « autre » |
| Réseau instable le jour J | Sessions de secours rejouables, cache Gemini prérempli |
| Latence SAM Audio sur Mac | Choisir la variante vidéo pour la scène 3 |
