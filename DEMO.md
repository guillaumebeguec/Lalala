# DEMO.md — Demo scenario and data strategy

## 1. The message to get across

> "Shazam recognizes a **recording**. Lalala finds a **song**: from a memory, a cover, or a sound captured in the noise."

The jury should leave with three images in mind:
1. an agent that **asks the right question**, with its uncertainty visibly melting away;
2. Shazam **failing live** on a cover, Lalala succeeding;
3. a table of hard numbers against serious baselines.

Target hackathon theme: **Build for Your World** (a problem everyone has had), with technical depth that touches "Push the Frontier" (cross-modal retrieval, Bayesian inference).

---

## 2. Storyboard (3 minutes)

| Time | Scene | What we show | What has to work |
|---|---|---|---|
| 0:00–0:20 | **The problem** | "You've all had a tune stuck in your head without being able to name it. Shazam is useless without the sound." Figure: thousands of requests of this kind on r/TipOfMyTongue. | One slide |
| 0:20–1:20 | **Scene 1 — The memory** | A jury member describes a track from memory. The agent asks 1 to 3 questions as buttons. The panel shows the hypotheses and **the entropy curve dropping**. A proposal with a 30 s clip to confirm. | Agent + information gain + mainstream catalogue |
| 1:20–2:10 | **Scene 2 — The cover** | We play a piano or acoustic cover of a hit from a phone. **Shazam live on a second phone: fails.** Lalala: the original title at the top. | Semantic audio search robust to covers |
| 2:10–2:35 | **Scene 3 — The noise** (or the video, whichever proves more reliable in rehearsal) | A track played with café chatter; SAM Audio isolates the music; we hear before/after; match. Variant: a 5 s video ad clip. | SAM Audio or PE-AV video |
| 2:35–3:00 | **Results and stack** | Table: Lalala against BM25, raw embedding, audio fingerprint (Shazam-style). Architecture slide: Llama, PE-AV, SAM Audio, FAISS (+ Gemini Embedding 2 in fusion). | `eval/` up to date |

**Golden rule**: every scene has a **rehearsed fallback** (a "replay" button that replays a recorded session) in case the live run goes wrong.

### Scene 1: how to make it safe
- The jury picks the track, so the catalogue has to be **broad on the mainstream side** (hits from 1970–2025, well-known ad and TV series music in France and internationally).
- Prepare 5 rehearsed cases that work every time, to suggest if the jury member draws a blank ("Think of an ad, or a TV series theme...").
- If the person thinks of a track outside the catalogue, the agent switches to web search: the demo still holds, the log shows the investigation.

### Scene 2: why it is the key scene
- It is the only moment where we **beat a product everyone knows, live**. Audio fingerprinting (Shazam, Chromaprint) compares exact recordings: a cover is a different recording. A semantic embedding compares musical content.
- Must be validated in phase 4: if no embedder finds the originals from covers (top-5 < 50 %), replace scene 2 with the video and do not promise covers.
- For the demo, pick 3 **instrumental** covers (piano, guitar) of very well-known hits, where the melody is intact but the timbre is completely different.

### Scene 3: video or noise
- Noise: spectacular if we let people hear the SAM Audio before/after. Risk: SAM Audio latency on Mac.
- Video: simple and visual. Risk: limited to 80 s with audio on the Gemini side, and dependent on the network.
- Decide in rehearsal (D6) based on measured reliability, not on what we would like.

---


### Decisions taken on the numbers (29/10/2026, dev panel of 2,124 tracks, `eval/retrieval.md`)

- **Scene 2 (cover)**: 50 % top-5 threshold **not reached**. T4 covers (44 cases): Gemma audio 25 % top-5, CoverHunter 18 % top-5 (11 % top-1), best fusion 32 % top-10; CoverHunter sliding windows: marginal gain. Structural cause: the cover and the original are only known through two 30 s clips taken at different positions in the song. Consequence: **do not promise covers live**; if we want to keep a cover in the demo, pick in advance a pair whose two clips overlap (checked by ear) and replay it.
- **Scene 3 (noise)**: solid without SAM Audio on synthetic MUSAN noise: fingerprint (audfprint or Chromaprint) 87 % top-1; with the PE-AV audio index (30/10, 2,063 clips: alone 30 % top-1 / 50 % top-10 on T3, 14 % / 23 % on T4, slightly below Gemma) the RRF k=10 fusion climbs to **93 % top-1 / 100 % top-5** on T3; previously fusion 97 % top-10 at 0/5/10 dB. Still to validate on the 20 real café recordings (`data/manual/real_recordings/`): that is the real judge of "Shazam fails, we succeed". **SAM Audio (29/10, `eval/sam_audio.md`, 20 cases ≤ 5 dB)**: generative separation is not faithful to the sample — fingerprint 80 % → 45 % on the isolated audio, Gemma 15 % → 5 % — but it rescues one case (85 % on raw *or* isolated); 38 s per clip (MLX). Kept as a **second chance for the fingerprint only**, the raw audio is used for similarity. On screen: an "Isolating the music" step is visible, honest about what it contributes.
- **Humming (bonus scene, PLAN_COMPLEMENTS C4)**: general-purpose embedders are at ~0 on 393 human hums (CHAD). hum2song: on 6 hums against 36 tracks, 2/6 in the top-5 with the contour of the mix; vocal separation with Demucs (54 s per clip) does **worse** (1/6) and sliding windows bring nothing → contour of the mix, no Demucs. On the 393 CHAD cases against the 2,124-track panel: hum2song **0.5 % top-1 / 2.3 % top-5 / 3.6 % top-10** (fragments at the start of the song: 3.5 % top-5), every other system at 0. **Decision: no "pure humming" scene** with 30 s references. Paths kept: (a) **sung lyrics** → Whisper transcription → lyric fingerprints (hashed n-grams, 21,015 tracks) + Llama + song cards: on 12 clips of 10 s of studio singing, 5 top-1, the 7 failures are all transcription failures; in production in the API (calibrated voice-only detection: recall 0.93, false positives 0.13 on CHAD); (b) possibly a reduced panel prepared for a bonus humming scene. **Update 29/10 16:40**: hum2song misses even *synthetic* hums rendered from the clip's melody (1/6 in top-3) while RMVPE tracks them to within 0.2 semitones → it is the learned embedder that fails, not the voice. Replaced by **direct melodic matching** (`core/melody.py`: contour alignment, tolerant to transposition, tempo 0.8–1.25 and silences, 3.2 s per query over 2,063 contours). On the **393 CHAD hums**: **16.0 % top-1 / 21.1 % top-5 / 24.9 % top-10, MRR 0.19** (30× hum2song); 6/6 top-1 on the synthetic ones. Structural ceiling: the hummed fragment is often outside the clip's 30 s. **Revised decision: a humming scene is possible** with tracks whose hummed passage is inside the clip (check in advance; `scripts/record_queries.py --mode hum` to practice), hum2song relegated to a secondary vote. **Update 29/10 21:50**: a second 30 s window per track (iTunes clip, 1,819 tracks out of 2,064; `scripts/build_contours.py --provider itunes`, merged stores, best alignment per track) → **19.6 % top-1 / 26.5 % top-5 / 29.8 % top-10, MRR 0.23** on the 393 CHAD (fragments at the end of the song: 23.5 % top-1, 37.3 % top-5), 6/6 synthetic; 7 s per query over 3,882 contours.
- **Scene 1 (memory, text)**: on 43 real ToT requests whose answer is in the panel, EmbeddingGemma 2 text→cards 14 % top-1 / 26 % top-10, fusion 30 % top-10; BM25 7 % top-10. A bigger Llama as the search engine (recommendation 1) is the expected lever.

---

## 3. Data strategy

### 3.1 Why not FMA
FMA is free but contains **no well-known track**. The whole point of the demo is that the jury searches for tracks they know. So we need a **mainstream catalogue**, of which we only store metadata, embeddings and locally cached clips.

### 3.2 Mainstream catalogue (core of the system)

| Building block | Source | Use | To check |
|---|---|---|---|
| Song metadata | **WASABI Song Corpus** (≈ 2 M songs, enriched metadata, links to external identifiers including Deezer according to its documentation) | Title, artist, year, genre, identifiers | Licence, actual presence of Deezer identifiers |
| 30 s audio clips | **Deezer API** (`preview` field of the search endpoint); **iTunes Search API** as fallback (`previewUrl`) | Audio embeddings, confirmation listening | Terms of use, rate limits |
| Track context | Wikipedia / Wikidata (summary, year, notable uses); MusicBrainz for deduplication | Text "song card" | — |
| Uses in ads / series / films | Web search at query time (Tunefind, ad music sites, forums); no mass scraping | The agent uses it live | — |
| Lyrics | **Not stored.** Lyric fragments searched on the web at query time | "I remember a bit of the chorus" clue | — |

**Target composition: 20,000 to 30,000 tracks.**
1. All answer tracks of the ToT_Music set (see 3.3) present in WASABI with an available clip: essential for evaluation.
2. Hits by decade (1970–2025), France and international: charts, editorial "best of" playlists by year.
3. Well-known ad and theme music in France (list partly built by hand, ~300 titles).
4. Distractors: tracks by the same artist or in the same genre as the above, so the search is not artificially easy.

**Song card** (indexed text, one document per track): title, artist, year, genre, short summary (Wikipedia), known uses, audio description generated once by Llama from the metadata (likely instruments, mood). Indexed with BM25 and a text embedding.

**Gemini cost (order of magnitude, to check against the official pricing)**: audio counts about 25 tokens per second, i.e. ~750 tokens per 30 s clip, i.e. ~20 M tokens for 25,000 clips. Use the batch API (half price) and the cache.

### 3.3 Test sets

| Set | Content | Size | Metric | Role |
|---|---|---|---|---|
| **T1 — ToT_Music** | Real requests from r/TipOfMyTongue with a verified answer, linked to the WASABI corpus (Bhargav et al., SIGIR 2023) | ~2,300, sample of 200 for the agent eval | top-1 / top-3 / top-10, number of questions | Scientific credibility: real human descriptions, not made-up cases |
| **T2 — French memories** | Fuzzy descriptions written by friends and colleagues, in French, about ads, theme songs, hits | 50 | same | The demo will probably be in French; CLAP only understands English |
| **T3 — Noise** | Catalogue clips mixed with chatter and noise (free **MUSAN** corpus) at SNR 0, 5 and 10 dB, + 20 real recordings (phone speaker in a café, captured by another phone) | 300 + 20 | top-5, with / without SAM Audio | Validate scene 3 |
| **T4 — Covers** | Piano, acoustic, live, karaoke covers of catalogue tracks, found through Deezer search ("title piano cover", "title acoustic") | 200 | top-1 / top-5 of the original; **Chromaprint fingerprint as baseline** | Prove "we beat Shazam" |
| **T5 — Video** | 5–10 s clips of ads or theme songs whose music is in the catalogue | 20 | top-5 | Validate the video variant |

**Simulated user** (to evaluate the agent on T1 and T2 without a human): a Llama distinct from the agent, which knows **only the text of the original request**. It answers the agent's questions from that text, and says "I don't know" otherwise. It never sees the expected answer. An "informed user" variant, which also knows the year and language of the real track, serves as an upper bound; both numbers are reported.

**Scientific point of attention**: in the ToT_Music paper, LLM query reformulations did **worse** than the full request used as is. Our rewriting (phase 5) must therefore be compared honestly against the raw query, and the fusion of the two is probably the right answer. It is an interesting result to show either way.

### 3.4 Baselines to beat (final table)
1. **BM25** on the song cards (the baseline of the ToT_Music paper).
2. **Raw embedding** of the query, without the agent.
3. **Chromaprint audio fingerprint** (Shazam proxy) for T3 and T4.
4. **Baseline agent** (questions freely chosen by Llama) against **information-gain agent**.

---

## 4. What to prepare ourselves (outside code)

- D1: Gemini key, web search key (Tavily or Brave), access requests for the PE-AV and SAM Audio weights.
- D1: read the Deezer / iTunes terms of use for clip usage; check the hackathon rules on the use of non-Meta models.
- D1–D3: collect the 50 French descriptions for T2 (a form sent to colleagues: "describe from memory a track from an ad or a series, and give the answer").
- D5: record the 20 real noisy recordings for T3 and choose the 3 covers for scene 2.
- D6: two full timed rehearsals, with Shazam installed on the second phone.
