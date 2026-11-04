# Test sets — phase 2 report

Catalogue: 24361 tracks, 23659 with a preview.

## T1 — ToT_Music

200 evaluation queries, 2006 in reserve (answer in the catalogue with a preview).
Paper split within the eval sample: {'test': 200} (C1: drawn from the `test` split). 0 queries have several acceptable recordings (multiple ISRCs).

| Decade | Eval |
|---|---|
| 1960s | 5 |
| 1970s | 9 |
| 1980s | 20 |
| 1990s | 46 |
| 2000s | 77 |
| 2010s | 43 |

Query types (eval): descriptive 200

`user_created` (C1): 514 queries across all of ToT_Music; 418 still contain a link to a user media file (vocaroo…), whose accessibility was not checked (no download).

## T2 — FR memories

0 descriptions matched to the catalogue (file `data/manual/fr_memories.csv`).

## T3 — Noise

300 cases (replayable recipe, audio mixed on the fly).

| Type | SNR | Cases |
|---|---|---|
| babble | 0 | 50 |
| babble | 10 | 50 |
| babble | 5 | 50 |
| noise | 0 | 50 |
| noise | 10 | 50 |
| noise | 5 | 50 |

## T4 — Covers

200 cover → original pairs (146 originals).

| Type | Pairs |
|---|---|
| live | 65 |
| karaoke | 56 |
| piano | 51 |
| acoustic | 28 |

Manual check of 20 pairs: to do (see `scripts/review_pairs.py`).

## T6a — CHAD hums

393 hums → original in the catalogue; CHAD groups 311, YouTube title read 294, title/artist parsed 268, in the catalogue 131, with a preview 131. Limitation: position of the fragment in the track ≠ position of the Deezer preview (30 s).

## T6b — Home hums

6 hums (`data/manual/hums/` real, `data/manual/hums_synth/` synthetic): synth 6.

## T5 — Video

0 clips in `data/manual/video/`.
