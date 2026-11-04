# PLAN.md — Lalala

Execution plan. The **why** (demo scenario, data choices, test sets) lives in `DEMO.md`: read it before phases 1, 2 and 10.

Each phase has a goal, tasks and **validation criteria**. Do not move on to the next phase until they are met. When blocked (API differs from the docs, access denied, incompatible terms of use), stop and propose an alternative rather than silently working around it.

## Indicative schedule (1 week, 1 person)

| Day | Phases |
|---|---|
| D1 | 0, 1 (catalog) |
| D2 | 2 (test sets), 3 (embeddings + index) |
| D3 | 4 (evaluation + fusion), 5 (rewriting) |
| D4 | 6 (baseline agent + Llama reranking) |
| D5 | 7 (information gain) |
| D6 | 8 (API + front), 9.1 or 9.2 (scene 3) |
| D7 | 10 (demo, rehearsals, plan B) |

## Modeling building blocks (🧪)

| Building block | Phase | Status |
|---|---|---|
| Multi-embedder fusion (RRF) | 4 | Required |
| Query rewriting, compared against the raw query | 5 | Required |
| Llama reranking (LLM cross-encoder) | 6.3 | Required |
| Information-gain questions | 7 | Required (core of scene 1) |
| Contrastive adapter on a frozen encoder | 9.3 | Optional |

Each building block keeps a simple baseline, implemented first, which serves as comparison and fallback.

---

## Target layout

```
.
├── PLAN.md
├── DEMO.md
├── pyproject.toml
├── .env.example
├── core/
│   ├── device.py
│   ├── config.py
│   ├── llm.py              # OpenAI-compatible client to Llama
│   ├── audio_io.py         # loading, resampling, mixing with noise
│   ├── cache.py            # hash-keyed disk cache (API calls, embeddings)
│   ├── embedders/
│   │   ├── base.py
│   │   ├── clap.py
│   │   ├── peav.py
│   │   └── gemini.py
│   ├── index.py            # FAISS (audio, text) + BM25
│   ├── fingerprint.py      # Chromaprint (Shazam-style baseline)
│   ├── fusion.py           # 🧪 RRF
│   ├── query_rewrite.py    # 🧪
│   ├── rerank.py           # 🧪 Llama reranking
│   ├── isolate.py          # SAM Audio
│   └── adapter.py          # 🧪 optional
├── catalog/
│   ├── wasabi.py           # WASABI loading and filtering
│   ├── previews.py         # Deezer / iTunes previews
│   ├── cards.py            # text track cards
│   └── build.py            # catalog assembly
├── testsets/
│   ├── tot_music.py        # T1
│   ├── fr_memories.py      # T2
│   ├── noisy.py            # T3
│   ├── covers.py           # T4
│   └── video.py            # T5
├── finder/
│   ├── tools.py
│   ├── agent.py
│   ├── belief.py           # 🧪 information gain
│   ├── simulated_user.py
│   └── prompts.py
├── api/main.py
├── web/index.html
├── scripts/                # one script per replayable step
├── eval/                   # numeric reports (markdown + csv)
├── data/                   # git-ignored
└── tests/
```

---

## Phase 0 — Setup (≈ 1 h)

Tasks:
1. `uv` project (Python 3.11). Dependencies: torch, torchaudio, transformers, faiss-cpu, rank-bm25, librosa, soundfile, numpy, pandas, pyarrow, google-genai, openai, httpx, tenacity, fastapi, uvicorn, python-multipart, python-dotenv, pyacoustid, pytest. Chromaprint (`fpcalc`) installed via Homebrew (document it).
2. `core/device.py`, `core/config.py`, `.env.example`: `LLM_BASE_URL`, `LLM_MODEL=llama3.1:8b`, `LLM_API_KEY`, `USE_GEMINI=true`, `GEMINI_API_KEY`, `GEMINI_DIM=768`, `SEARCH_PROVIDER`, `SEARCH_API_KEY`, `EMBEDDERS=clap,peav,gemini`, `DATA_DIR=./data`.
3. `core/llm.py`: `chat(messages, tools=None, json_schema=None)`, lenient JSON parser + one retry on invalid JSON.
4. `core/cache.py`: disk cache keyed by hash(content + parameters), used by every expensive call.
5. `.gitignore`: `data/`, `.env`, caches.

**Validation**: `chat` answers; `@pytest.mark.llm` test on a dummy `tool_call` and a JSON output; a test Gemini call (short text) returns a vector of the requested dimension.

---

## Phase 1 — Mainstream catalog (≈ 1 day)

See `DEMO.md` § 3.2 for the rationale.

Tasks:
1. `catalog/wasabi.py`: download the **WASABI Song Corpus** (read the official page for the URL, format and license; do not guess). Extract: identifier, title, artist, album, year, genre, external identifiers (Deezer if present, MusicBrainz). **Flag it** if Deezer identifiers are missing: we will go through Deezer search instead.
2. `testsets/tot_music.py` (retrieval only at this stage): download **ToT_Music** (Bhargav et al., SIGIR 2023; link from the arXiv paper 2305.14072 or the associated code repository). Extract the list of answer tracks.
3. `catalog/build.py`: assemble the catalog, in priority order:
   a. every ToT_Music answer track;
   b. hits by decade 1970–2025, France and international (editorial playlists and charts via the Deezer API);
   c. `data/manual/ads_series_fr.csv`: ad music and TV themes (I will provide the list; prepare the empty file with the columns `titre, artiste, usage, annee`);
   d. distractors: other tracks by the same artists and genres.
   Deduplicate (same normalized title + artist, remastered versions). Target: 20,000 to 30,000 tracks. A **dev** subset of 2,000 tracks containing 300 ToT_Music answers.
4. `catalog/previews.py`: for each track, fetch the 30 s preview via the **Deezer API** (direct identifier or `artist:"…" track:"…"` search), with the **iTunes Search API** as fallback. Check the title/artist match (string similarity) to avoid mismatches. Respect rate limits (`tenacity`, backoff). Local cache, never committed. Log the coverage rate.
5. `catalog/cards.py`: text track card = title, artist, year, genre, short Wikipedia summary (Wikipedia API, FR then EN), known usages if present in the manual list, + an **audio description** generated once by Llama from the metadata (marked as generated). No lyrics.

**Validation**:
- Report `eval/catalog.md`: number of tracks, preview coverage (target > 85 %), breakdown by decade and language, share of ToT_Music answers covered (target > 70 %).
- Manual check of 30 preview ↔ metadata pairings: 0 gross errors tolerated on the sample, otherwise tighten the filter.

---

## Phase 2 — Test sets (≈ ½ day)

See `DEMO.md` § 3.3. All sets are frozen in `data/testsets/*.jsonl` with a fixed seed.

Tasks:
1. **T1 ToT_Music**: keep only the queries whose answer is in the catalog with a preview. Evaluation sample of 200 (stratified by decade), the rest held in reserve. Fields: `id, texte, reponse_id`.
2. **T2 French memories**: file `data/manual/fr_memories.csv` (`description, titre, artiste`) that I will fill in; validation script checking that every answer exists in the catalog.
3. **T3 Noise**: download **MUSAN** (read the official page); mix 100 catalog previews with noise and babble at SNR 0 / 5 / 10 dB, 10 s windows at random positions. Folder `data/manual/real_recordings/` for my 20 real recordings (named `<track_id>.m4a`).
4. **T4 Covers**: for 300 catalog hits, search Deezer for "title + piano", "title + acoustic cover", "title + live", "title + karaoke". Keep results whose title matches and whose artist **differs** (or with a live mention). Fetch their preview. Target: 200 cover → original pairs. Manual check of 20 pairs.
5. **T5 Video**: folder `data/manual/video/` (I will provide ~20 clips `<track_id>.mp4`).
6. `finder/simulated_user.py`: simulated user as specified in `DEMO.md` § 3.3 (sees only the original text; an "informed" variant that also knows the year and language). **Test** checking that the expected answer never appears in its prompt.

**Validation**: statistics for each set in `eval/testsets.md`; anti-leak test green.

---

## Phase 3 — Embeddings and index (≈ ½ day + compute)

Tasks:
1. `core/embedders/base.py`: `embed_audio(items)`, `embed_text(texts, task="query"|"document")`, `embed_video(path)` (optional depending on the embedder, otherwise `NotImplementedError`), L2-normalized outputs.
2. `clap.py`: `laion/larger_clap_music` via `transformers`, MPS, batching. Note: English only, French queries go through a Llama translation.
3. `peav.py`: PE-AV from `facebookresearch/perception_models`; **read the README and the examples**; audio, text, video. If the weights are gated, stop and report.
4. `gemini.py`: `gemini-embedding-2` via the `google-genai` SDK. **Read the official docs** for: audio limit per request, accepted formats (mp3/wav), task instructions (prefixes such as `task: … | query: …`), reduced dimension via MRL (`GEMINI_DIM`), batch API. Hash-keyed cache, batch for the catalog, single calls for queries.
5. `core/index.py`: one FAISS index per (embedder × modality) — preview audio, card text — plus a BM25 index on the cards. API: `search(query, embedder, modality, k)`.
6. `core/fingerprint.py`: Chromaprint fingerprints of the catalog previews and best-match lookup (Shazam-style baseline).
7. Scripts: `build_embeddings.py --embedder … --subset {dev,full}`, `build_index.py`, `build_fingerprints.py`. Resumable if interrupted.

**Validation**: all three embedders run on the dev subset; dev time and cost (Gemini) reported; projected cost and time for the full catalog before launching it, **to be validated with me**.

---

## Phase 4 — 🧪 Evaluation, fusion and baselines (≈ ½ day)

Tasks:
1. `scripts/eval_retrieval.py`: without the agent, for each test set:
   - T1, T2 (text → track): BM25 on cards; text embedding → cards; text embedding → audio; for each embedder.
   - T3, T4 (audio → track): Chromaprint; audio embedding → audio for each embedder.
   - T5 (video → track): PE-AV and Gemini.
   Metrics: top-1, top-5, top-10, MRR.
2. `core/fusion.py`: RRF (`k` ∈ {10, 30, 60}, weighted variant) across all available lists for a given query.
3. `eval/retrieval.md`: one table per set, with baselines, each embedder alone and the fusion; 5 qualitative examples; configuration recommendation.

**Validation**:
- The report settles the default configuration for each modality.
- **Scene 2 decision**: if the best top-5 on T4 (covers) is < 50 %, flag it clearly; scene 2 will be replaced (see `DEMO.md`).
- Unit test for `rrf`.

---

## Phase 5 — 🧪 Query rewriting (≈ ½ day)

Tasks:
1. `core/query_rewrite.py`: from a fuzzy description, Llama produces, as JSON:
   - 3 to 5 **audio captions** (instruments, approximate tempo, vocals, mood, production), in English;
   - **structured clues**: likely decade, lyrics language, medium (ad, series, film, game, radio), genre, possible lyric fragments;
   - a cleaned **text query** for BM25 and the cards.
   Inventing titles is forbidden.
2. Compare on T1 and T2: raw query alone; rewrite alone; **fusion of raw + rewrite**. The ToT_Music paper observed that LLM reformulations did worse than the raw query: we measure, we do not assume.
3. Structured clues act as **soft filters** (score bonus on decade and language), not hard filters.

**Validation**: results added to `eval/retrieval.md`; the configuration kept is the one that wins, even if it is the raw query.

---

## Phase 6 — Baseline agent and reranking (≈ 1 day)

### 6.1 Tools — `finder/tools.py`
- `catalog_search(text)`: hybrid search (BM25 cards + text embeddings + text → audio, with rewriting), fusion, returns the 20 best candidates with their card.
- `audio_match(audio)`: audio embedding (+ isolation in phase 9), fusion, and Chromaprint in parallel (if the fingerprint matches strongly, it is the exact recording: immediate answer).
- `web_search(query)`: Tavily or Brave, cached; for usages (ads, series) and lyric fragments.
- `ask_user(question, options)`: one closed question, 2 to 4 options + "I don't know".
- `propose(candidates)`: 1 to 3 candidates `{titre, artiste, annee, confiance, raison, preview_url}`.

### 6.2 Loop — `finder/agent.py`
- Explicit state: input, clues, candidates, queries made.
- Each turn, Llama chooses: search, ask, propose. Guardrails: 3 questions, 8 tool calls, no repeated query, no title without a source (catalog or web).
- A candidate found on the web but **absent from the catalog** is proposed with a search link (no preview) and marked as such.
- Events emitted for the front: clue, search, candidates, question, proposal.

### 6.3 🧪 Llama reranking — `core/rerank.py`
- Embeddings are bi-encoders: query and track are encoded separately. Reranking adds a step where the LLM sees **the query and the card together**, like a cross-encoder.
- Input: description + clues + 20 candidates (short cards). JSON output: a score from 0 to 10 per candidate and a short justification. In batches of 5 to fit the 8B context. Final score = combination of the fusion score and the Llama score (weight tuned on a subset of T1, evaluated on the rest).
- **Validation**: top-1 / top-3 gain on T1 and T2 compared with fusion alone, reported in `eval/retrieval.md`.

### 6.4 Agent evaluation — `scripts/eval_finder.py`
- Replays T1 (200) and T2 (50) with the simulated user (both variants). Measures: top-1, top-3, number of questions, tool calls, time, share of answers found via the web versus via the catalog.

**Validation**: no infinite loop; `eval/finder.md` filled in; the agent beats the best agent-less search (phase 5) on T1 in top-3, otherwise flag it.

---

## Phase 7 — 🧪 Information-gain questions (≈ 1 day)

`finder/belief.py`:
1. **Belief**: the candidates from the search (top-20 after reranking) + an "other" hypothesis. Prior = normalized scores (softmax with temperature tuned on T1), not a free Llama estimate.
2. **Candidate questions**: a **fixed repertoire** of questions whose answer can be read from the cards (decade, language, male/female/instrumental vocals, genre, slow/fast tempo, ad/series/film medium) + 2 free-form questions proposed by Llama.
3. **Observation model**:
   - repertoire questions: `P(answer | h)` computed **from the card metadata** (exact, fast, no LLM), smoothed by `ε` to account for the user's memory errors (ToT queries often contain false memories);
   - free-form questions: estimated by Llama as JSON, then smoothed.
   - "I don't know": weakly informative.
4. **Choice**: maximize `IG(q) = H(P) − Σ_r P(r)·H(P | r)`.
5. **Update**: Bayes; if "other" exceeds 0.3, run a new search (catalog + web) with the new clues and inject the candidates back.
6. **Stop**: `max P(h) > 0.7` or 3 questions.
7. Expose at each turn: entropy, distribution, chosen question and its expected gain.
8. Tune `ε` and the temperature on 50 T1 cases, evaluate on the other 150.

**Validation**: baseline (6.2) versus information gain compared on T1 and T2 (top-1, top-3, average number of questions) in `eval/finder_ig.md`, with bootstrap confidence intervals. Entropy decreases after at least 80 % of answers. If information gain does not do better, keep the baseline and note it.

---

## Phase 8 — API and front (≈ ½ to 1 day)

API (`api/main.py`):
- `POST /session`; `POST /session/{id}/message` (text or answer to a question), SSE response;
- `POST /session/{id}/audio` (microphone recording); `POST /session/{id}/video`;
- `GET /preview/{track_id}` (preview from the local cache);
- `POST /replay/{nom}`: replays a recorded session (demo plan B); any session can be recorded under a name.

Front (`web/index.html`, HTML + vanilla JS, light/dark theme, readable on a projector):
- conversation; questions as buttons; microphone button (MediaRecorder, 10 s); video drop;
- "investigation" panel: clues, hypotheses as probability bars, **entropy curve**, queries made, source of each candidate (catalog, web, fingerprint);
- candidate cards with a preview player and a "that's the one" / "no" button (a "no" restarts the investigation);
- for scene 3: before/after isolation player.

**Validation**: full runs of scenes 1 and 2 from `DEMO.md` without console errors, on Chrome and Safari; microphone working.

---

## Phase 9 — Scene 3 and options (≈ 1 day)

1. **SAM Audio** (`core/isolate.py`): isolate the music (prompt "music") from a noisy recording before `audio_match`. Read the official README; measure latency on MPS. Evaluate on T3 with and without isolation, per SNR → `eval/isolation.md`.
2. **Video**: `embed_video` via PE-AV and Gemini on T5; additionally, Llama describes the scene (if a vision Llama is available) to steer the web search. → `eval/video.md`.
3. 🧪 **Contrastive adapter** (optional): projection trained on a frozen encoder (symmetric InfoNCE, learned temperature), audio ↔ card + caption pairs, split by artist, evaluated on never-seen T1/T2. Drop it if it does not win on T1.
4. **Bigger Llama** via a hosted API: comparison on T1.
5. **Humming** (experimental): pitch contour + DTW on the previews. Do not let the demo depend on it.

**Validation**: scene 3 decision (noise or video) made on the numbers from 9.1 and 9.2, recorded in `DEMO.md`.

---

## Phase 10 — Demo (≈ ½ day)

1. Follow the storyboard in `DEMO.md`. Record a backup session per scene (`/replay`).
2. Warm up: Ollama, local embedders, SAM Audio; pre-fill the Gemini cache for the rehearsed cases; check that the backup cases work **without network**.
3. Results slide generated from `eval/` (`scripts/make_results_slide.py` → a PNG/SVG table: Lalala versus BM25, raw embedding, Chromaprint).
4. `README.md`: installation, external data and its terms, what is sent to Gemini, disabling with `USE_GEMINI=false`.

---

## Known risks

| Risk | Mitigation |
|---|---|
| Incompatible Deezer / iTunes terms | Check them on D1; local, non-commercial use, no redistribution; possible fallback to a reduced catalog |
| Hackathon rules unfavorable to non-Meta models | `USE_GEMINI=false`: PE-AV + CLAP take over; the phase 4 numbers show what is lost |
| Covers poorly retrieved (T4 < 50 % in top-5) | Replace scene 2 with video; do not promise covers |
| ToT_Music answers poorly covered by the catalog | Prioritize fetching them in phase 1; evaluate only on covered cases, and say so |
| Llama 8B weak at tool calling or JSON | Constrained output + lenient parser; bigger Llama via API |
| False memories from the user | `ε` smoothing, soft filters, "other" hypothesis |
| Unstable network on demo day | Replayable backup sessions, pre-filled Gemini cache |
| SAM Audio latency on Mac | Pick the video variant for scene 3 |
