# Lalala — hackathon journal

Meta Global AI Developer Hackathon, hosted by Meta and Cerebral Valley, October 26 – November 4, 2026, Indie track, theme "Build for Your World".
One MacBook M2 24 GB, Meta's open models (Llama 3.1, PE-AV, SAM Audio, FAISS), one week.

## Before starting

The starting idea: a Shazam that works when you don't have the recording. Everyone has had a song in their head without
finding the title: a tune you hum, three words of the chorus, "the car ad with the whistling, around 2008",
a cover in a bar, a noisy video.

First lead, the day before the kickoff: a RAG over a song catalogue. You type the memory, an embedding searches,
an LLM answers. Dropped within the hour: the jury doesn't want "basic RAG", and a RAG does nothing with a hum.
New idea: an agent that investigates. Several ways to search (text, sound, melody, sung lyrics), a vote,
a calibrated confidence, and a question to the user when it hesitates. The product is called Lalala, because that is what you do when you are looking for a title.

Rules we set ourselves: Meta models at the core, everything local, no audio stored, no lyrics stored, and test
sets the agent never sees. That last rule spared us several false joys.

## Thursday, October 29

No pipeline today, first something to measure with.

- Catalogue: WASABI 2.0, 24,361 songs, extracted as a stream without the lyrics. 30 s clips via the Deezer API,
  iTunes as fallback. Deezer's terms forbid storing the audio: clips are read in memory, embedded,
  discarded. We only keep the vectors. Signed URLs expire in under an hour, refreshed on demand.
- Test sets, one per situation: fuzzy memories (ToT_Music, real Reddit requests), noisy recordings (replayable
  MUSAN recipe, 0/5/10 dB), covers → original, 393 real hums (CHAD). Simulated user that doesn't know
  the answer.
- Dev subset of 2,000 songs to iterate on, the full set for the end.
- Idea noted for later: one card per song (Wikipedia summary + Llama description "what it sounds like, what it's
  about"), so text search has something other than titles.

## Friday, October 30

Plumbing day, goal: a complete chain, even a mediocre one.

- Embedders behind a common interface: PE-AV (audio, text↔audio), EmbeddingGemma (text↔cards), CLAP for
  comparison, Gemini as an option that can be disabled. FAISS for vectors, BM25 for words, Chromaprint for the fingerprint.
- Llama in three roles: rewrite the memory into a clean query, rerank the top 10, ask the most useful question
  (Bayesian belief, information gain).
- SSE API to display the investigation step by step with the model at work. Lalala front: Capture, Searching,
  Result. Tested on the phone over the local network to get a real microphone.
- macOS: faiss and torch each ship their own OpenMP, the API crashed with no message. OMP_NUM_THREADS=1, fixed.
- In the evening, first version of humming with hum2song (published contour embedder). 2 out of 6 in the top 5. We believe in it.

## Saturday, October 31

First eval report on the dev panel. It reorganized the week.

- Noise: fingerprint 87 % top-1, fusion 97 % top-10. Scene secured.
- Covers: 25 % top-5 at best. The idea of "play a piano cover and beat Shazam" doesn't hold on 30 s clips:
  the cover and the original don't overlap on the same window. Written in DEMO.md: scene dropped.
- Humming: hum2song at 0.5 % top-1 on the 393 real hums. Everything else at zero.

Eight ideas from the night, applied during the day: routing by input type (a text only queries the audio index
if it describes a sound), Llama as a knowledge-based search engine (titles proposed, then resolved in the catalogue),
per-branch thresholds, learned fusion instead of equal-weight RRF, lyric themes in the cards, audfprint
fingerprint more robust to the microphone than Chromaprint.

Pivot of the day: when people "hum", they often sing words. Words → Whisper. And we can search
the lyrics without storing them: non-reversible hashes of 4-grams, computed once from the corpus. 5 top-1 on
12 sung clips, the failures are all transcription failures. Humming becomes two problems: words,
or a melody.

End of day: learned fusion trained, 28 % top-5 on fuzzy memories against 21 % with RRF. The agent and its
questions on 15 cases: no gain with an 8B. Feature kept, not the selling point.

## Sunday, November 1

Morning question: does hum2song fail because of the voice or because of the embedder? To settle it, synthetic hums:
we extract the melody from the clip, render it as "mmm" with transposition, tempo, vibrato. RMVPE recovers it to within 0.2
semitones, hum2song still misses it. It's the embedder.

So we remove it. Direct contour alignment: the query melody slid over that of each clip, tolerance
to transposition, tempo (0.8–1.25) and silences. On the 393 hums: 16 % top-1, 25 % top-10, thirty times
better. And a clear diagnosis: by position of the hummed passage, song intros are the least recognized because
the 30 s clip often starts at the chorus. The ceiling is the window, not the method.

SAM Audio, access to the weights received. Painful install on Mac (the package installs a `core` module like ours, two
dependencies without a macOS wheel): separate venv, subprocess, MLX port three times faster. Measured on 20 noisy
cases: the separation is generative, the isolated audio degrades the fingerprint and the embeddings on average, but saves one
case nothing else catches. Policy: second chance for the fingerprint only, never instead of the raw audio.

Llama 8B: from "guitar?" it made up "rock, male vocals" and searched for those words in titles and
lyrics. Two fixes: a clue is only kept if it's in the user's words, and words that describe
a sound query the audio descriptions and the audio index, not the titles or the lyrics.

## Monday, November 2

On the phone, a hum without words almost always landed on "We couldn't pin it down". Three causes in the
logs: Whisper hallucinates lyrics ("Thank you for watching!") on an "mmm", RRF scores are always
"low" because they have no scale, and a 60 s Llama branch was still lingering in the API.

- Words or melody: detection of real words (language probability, transcription quality, hallucination list).
  Without words, the melody decides alone.
- Confidence: the melodic score doesn't separate a good result from a bad one (0.73 against 0.72). The margin between first
  and second does. Calibrated on the 393 hums.
- No more dead end: at low confidence, a top 10 to listen to instead of a failure.
- Second window per song: the iTunes clip often falls on a different passage than Deezer. 1,819 tracks have
  one. Humming: 16 % → 19.6 % top-1, 30 % top-10.

Night of computing, one model at a time so the Mac doesn't overheat: 2,000 Llama cards, text and sound indexes, full
eval, homemade cross-encoder, fusion retrained.

## Tuesday, November 3

Reading the night's results. PE-AV audio had saturated GPU memory halfway through: freeing between batches, resuming from the
cache. Homemade cross-encoder: degrades top-1, dropped. Lyric themes: no gain on fuzzy memories, kept
in the cards, not in the pitch. PE-AV alone slightly below Gemma on audio, but the fusion with it climbs to 93 %
top-1 on noise.

| Situation | Dev panel | In the demo |
|---|---|---|
| Fuzzy memory | 14 % top-1, 30 % top-5 (learned fusion) | short list, the agent asks a question |
| Noisy recording | 93 % top-1, 100 % top-5 | strong scene |
| Cover / live | 32 % top-10 | dropped |
| Humming | 19.6 % top-1, 30 % top-10 | holds if the passage is inside the window |

Repo cleanup, README for the jury, published on GitHub.

## What we take away

- Measuring first cost a day and saved three: hum2song, isolated audio for similarity and the
  cross-encoder were dropped on the numbers.
- A small LLM makes things up. It has to be anchored in the user's words, and "what it sounds like" kept separate from "what it's about".
- When a learned model fails, building an ideal input tells you right away whether it's the model or the data.
- The displayed confidence has to be calibrated on real cases, otherwise the interface lies.
- Naming the structural limit (the 30 s window) says where the next gain is.

## With one more week

- Full audio for the 15 demo tracks (DRM-free purchase): the window goes away. The code already accepts several
  windows per track.
- A Llama 70B via API for the knowledge-based search engine and listwise reranking: wired up, no model.
- Real microphone recordings instead of synthetic noise (the tool exists).
- Video with PE-AV frames, planned, not measured.
