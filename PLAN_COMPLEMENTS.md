# PLAN_COMPLEMENTS.md — Additions to the Lalala plan

This file **complements** `PLAN.md`, `DEMO.md` and `FRONT_PLAN.md` without replacing them. It builds on existing projects identified after the plan was written: the ToT_Music dataset from Spotify Research, the TREC Tip-of-the-Tongue track, the CoverHunter cover-song model, the CHAD dataset and the hum2song humming model.

**How to read this file**
- Read it **after** `PLAN.md`, before each phase mentioned below.
- Each addition states which phase it attaches to ("adds to" or "replaces").
- In case of contradiction, **this file wins**, only on the point concerned.
- The product name is **Lalala**.

## Summary

| # | Addition | Attached to | Effort | Status |
|---|---|---|---|---|
| C1 | Use every ToT_Music field (ISRC, types, split, reformulations) | adds to phases 1, 2, 5 | 2 h | Required |
| C2 | "Covers" audio branch with CoverHunter | adds to phases 3, 4, 6 | ½ day | Required |
| C3 | CHAD test set (covers and hummings) | adds to phase 2 | 2 h | Required |
| C4 | Humming with hum2song | **replaces** phase 9.5 | ½ day | Optional, recommended |
| C5 | TTMR++ as an additional text → music embedder | adds to phases 3, 4 | 2 h | Optional |
| C6 | Comparability with the literature (TREC ToT, ToT_Music) | adds to phases 4, 10 | 1 h | Required |
| C7 | Front impacts (steps, sources, input types) | adds to `FRONT_PLAN.md` § 3, 4 | 1 h | Required if C2/C4 |

Schedule impact: C1 and C3 on D2, C2 on D3, C4 on D6 (phase 9 slot), C5 only if time remains on D3, C6 on D3 and D7.

---

## C1 — Use every ToT_Music field

Source: `spotify-research/tot` repository (data only, CC0 license), file `data.jsonl.zip`, one query per line.

**Adds to phase 1 (catalog)**
- Fetch the answer tracks **by ISRC** rather than by title/artist: fields `wasabi_isrc` and `spotify_resolved.items[].isrc`. Look up the Deezer preview by ISRC (check the Deezer documentation for the ISRC lookup endpoint), with title + artist search as fallback.
- Several recordings can be valid for the same query (`spotify_resolved.items` is a list): store **all acceptable ISRCs** per query; a result is correct if it matches any of them.
- Report: share of answers found by ISRC, by text search, and not found.

**Adds to phase 2 (test sets)**
- Use the original `split` field: the T1 evaluation sample is drawn from `test`; hyperparameter tuning (ε, temperature, fusion weights) is done on `train`/`val` only.
- Keep `query_types` (`descriptive`, `user_created`, `extant_media`) and the `annotations` labels (genre, era, lyrics, comparison with an artist, uncertainty…) to **break down results by query type** in the reports.
- `user_created` most likely denotes queries that come with a user-made recording; check the paper (Table 1) and, if so, report how many of these queries still have accessible media.

**Adds to phase 5 (rewriting)**
- The `reformulations` field holds the paper's LLM reformulations (`keywords`, `reform_10`, `reform_25`, `reform_50`, `reform`). Add them as a **direct baseline** in the phase 5 table: raw query, paper reformulations, our rewrite, fusion.

**Validation**: `eval/catalog.md` reports coverage by ISRC; `eval/retrieval.md` contains a table per `query_types` and the "paper reformulations" row.

---

## C2 — "Covers" audio branch with CoverHunter

Observation: CLAP, PE-AV and Gemini Embedding 2 are **general-purpose** embeddings, not trained to recognize the same song across different versions (timbre, tempo, key, structure). That is the job of *cover song identification*. This branch directly targets scene 2 of `DEMO.md`.

**Adds to phase 3 (embeddings and index)**
1. `core/embedders/coverhunter.py`: audio → audio only (`embed_text` raises `NotImplementedError`).
   - Start from the official `Liu-Feng-deeplearning/CoverHunter` repository and its follow-up `alanngnet/CoverHunterMPS`. **Read both READMEs**: check whether the follow-up brings Apple Silicon (MPS) support, how to extract the features (CQT), and the license of the pre-trained model (trained on SHS100K).
   - If the installation fails on Mac, stop and report before looking for another model.
2. Dedicated FAISS index on the catalog previews.
3. Point of attention: these models are designed for full tracks; here, query and catalog are 10 to 30 s excerpts, which do not necessarily cover the same passage. Test two strategies: one embedding per full excerpt, and sliding-window embeddings (e.g. 10 s, 5 s hop) with score = maximum similarity.

**Adds to phase 4 (evaluation)**
- Evaluate CoverHunter alone, each general-purpose embedder alone, and the fusion, on T4 (covers), T3 (noise) and C3 (CHAD).
- The phase 4 **scene 2 decision** is made on the best configuration, CoverHunter included.

**Adds to phase 6 (agent tools)**
- `audio_match` queries CoverHunter in parallel with the other embedders. Decision order: strong Chromaprint fingerprint → exact recording; otherwise weighted fusion, with a higher CoverHunter weight when the fingerprint failed (the query is probably another version).
- Justification returned to the front: source `Version`, text such as "Same song, different version (cover or live)."

**Validation**: CoverHunter row in `eval/retrieval.md` for T3, T4 and CHAD; per-query inference time on Mac reported.

---

## C3 — CHAD test set (covers and hummings)

Source: CHAD (Covers and Hummings Aligned Dataset, ISMIR 2023), `amanteur/CHAD` repository. About 18 h of short fragments, each labeled `original`, `cover` or `humming`, grouped by `group_id` and time-aligned. The humming subset is published on Hugging Face; originals and covers are referenced by YouTube identifier.

**Adds to phase 2**
1. Download the **humming** subset from Hugging Face and the metadata (`metadata/dataset.csv`). **Do not download any YouTube audio** without checking the terms of use with me first.
2. Find the original tracks in our catalog (title/artist from the metadata, then Deezer preview). Keep only the groups whose original is covered.
3. Build two sets:
   - **T6a Hummings**: CHAD humming → original catalog track;
   - **T6b CHAD covers**: only if the cover fragments are accessible without downloading from YouTube; otherwise stick to T4.
4. Limitation to document: the hummed fragment does not necessarily fall within the 30 s of the Deezer preview. Measure this overlap when the intervals allow it, and report results separately for the cases where it is likely.

**Validation**: T6a statistics (and T6b if built) in `eval/testsets.md`.

---

## C4 — Humming with hum2song

**Replaces phase 9.5** ("pitch contour + DTW").

Source: `luigir5/hum2song` model on Hugging Face, code `luiginotmario/hum2song`; melodic contour encoder of about 5.2 M parameters, a single tower for queries and tracks, 256-dimensional embeddings, weights under CC BY 4.0 license. Handles humming, whistling and singing with wrong lyrics; does not use the lyrics and struggles on tracks without a clear vocal melody.

Tasks:
1. `core/embedders/hum2song.py`: melodic contour extraction (the model card uses RMVPE), then embedding. **Read the README and the model card** before coding; use the `e2b` checkpoint recommended for open retrieval.
2. Compute contours and embeddings for the catalog previews (test with and without prior vocal isolation by Demucs); dedicated FAISS index.
3. Routing: all branches run on an audio input; add a simple "solo monophonic voice" detection (high pitch confidence, low polyphony) that increases the hum2song weight in the fusion.
4. Evaluate on T6a (CHAD) and on 20 hummings recorded by me (`data/manual/hums/<track_id>.m4a`).

**Demo decision**: if top-5 on the real hummings exceeds 50 %, propose in `eval/` a **bonus scene** "the jury hums" (to be arbitrated with me against scene 3). Otherwise, keep humming out of the demo.

**Validation**: `eval/humming.md` with T6a and real-humming results, with and without Demucs.

---

## C5 — TTMR++ as a text → music embedder (optional)

TTMR++ (Doh et al.) is a text-to-music retrieval model, trained with LLM-enriched descriptions and metadata; code and demo published by the authors.

- `core/embedders/ttmr.py` (text and audio), to add to the phase 4 table and to the fusion.
- Only do this if CLAP is clearly the weak link on the text → audio side after phase 4.
- Check the license of the weights before use.

---

## C6 — Comparability with the literature

**Adds to phase 4**
- On the ToT_Music `test` split, reproduce the paper's BM25 baseline (full query) and place our results next to it, stating the corpus differences (ours is smaller than full WASABI: write it explicitly, as the task is easier there).

**Adds to phase 10 (results slide)**
- One positioning sentence: the TREC Tip-of-the-Tongue 2025 track (films, places, people) showed that hybrid approaches (BM25 + dense embeddings + LLM signals) followed by a reranker beat every isolated model; Lalala applies this pattern to music and adds audio (fingerprint, covers, humming).
- Cite ToT_Music (Bhargav, Schuth, Hauff, SIGIR 2023) as the source of the test set.

---

## C7 — Front impacts

**Adds to `FRONT_PLAN.md` § 3 and to the mock-mode fixtures**:
- New possible steps in *Search progress* (titles returned by the backend): **Version matching** (CoverHunter), **Melody matching** (hum2song).
- New sources in *Why this match*: **Version** ("Same song, different version (cover or live)."), **Melody** ("Your hummed melody matches the vocal line.").
- New input type in the *Your input* card: **Humming · 8 s** when the backend detects a solo voice (field `input.kind = "humming"` in the `input` SSE event).
- Add a `humming.json` mock fixture and the `?replay=humming` scene.

---

## Added risks

| Risk | Mitigation |
|---|---|
| CoverHunter does not run on Mac or its weights have an incompatible license | Keep the fusion of general-purpose embeddings; report before looking for another model |
| 30 s previews too short for cover identification | Sliding windows; pick demo covers whose played passage is within the preview |
| CHAD poorly covered by our catalog | Evaluate on the covered part, and say so; complement with our own hummings |
| Jury hummings too approximate | Bonus scene only if the real numbers are good; otherwise, out of the demo |
