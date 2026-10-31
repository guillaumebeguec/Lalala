# Recherche sans agent — sous-ensemble dev

Catalogue : 2000 morceaux ; embedders (dernier run) : clap, gemma, coverhunter ; BM25 : oui ; Chromaprint : oui ; audfprint : oui.

## T3 Bruit

30 cas ; 100 s ; systèmes : chromaprint, audfprint, clap:audio→audio, gemma:audio→audio, coverhunter:audio→audio, rrf(k=10), rrf(k=30), rrf(k=60), rrf-embeddings(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.867 | 0.900 | 0.900 | 0.875 |
| audfprint | 0.867 | 0.867 | 0.867 | 0.867 |
| clap:audio→audio | 0.167 | 0.267 | 0.267 | 0.214 |
| gemma:audio→audio | 0.333 | 0.433 | 0.467 | 0.366 |
| coverhunter:audio→audio | 0.233 | 0.300 | 0.367 | 0.274 |
| rrf(k=10) | 0.867 | 0.933 | 0.967 | 0.889 |
| rrf(k=30) | 0.867 | 0.867 | 0.933 | 0.878 |
| rrf(k=60) | 0.833 | 0.867 | 0.900 | 0.861 |
| rrf-embeddings(k=60) | 0.533 | 0.633 | 0.633 | 0.574 |

Détail de `rrf(k=10)` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| 0 | 10 | 0.800 | 0.900 |
| 10 | 10 | 0.900 | 0.900 |
| 5 | 10 | 0.900 | 1.000 |

## T4 Reprises

44 cas ; 130 s ; systèmes : chromaprint, audfprint, clap:audio→audio, gemma:audio→audio, coverhunter:audio→audio, rrf(k=10), rrf(k=30), rrf(k=60), rrf-embeddings(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.068 | 0.091 | 0.091 | 0.077 |
| audfprint | 0.000 | 0.023 | 0.023 | 0.011 |
| clap:audio→audio | 0.000 | 0.023 | 0.023 | 0.008 |
| gemma:audio→audio | 0.182 | 0.250 | 0.250 | 0.206 |
| coverhunter:audio→audio | 0.114 | 0.182 | 0.205 | 0.143 |
| rrf(k=10) | 0.182 | 0.227 | 0.318 | 0.220 |
| rrf(k=30) | 0.136 | 0.205 | 0.273 | 0.186 |
| rrf(k=60) | 0.136 | 0.205 | 0.250 | 0.180 |
| rrf-embeddings(k=60) | 0.091 | 0.159 | 0.341 | 0.141 |

Détail de `gemma:audio→audio` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| acoustic | 6 | 0.333 | 0.333 |
| karaoke | 13 | 0.077 | 0.077 |
| live | 15 | 0.200 | 0.400 |
| piano | 10 | 0.200 | 0.200 |

## Exemples qualitatifs (fusion RRF k=60)

- **T3 Bruit** « t3_dz818256_0 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; We Can Work It Out — The Beatles; Thinking Of You — Sister Sledge
- **T3 Bruit** « t3_dz818256_5 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; Snakes And Ladders — Men At Work; We Can Work It Out — The Beatles
- **T3 Bruit** « t3_dz818256_10 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; Snakes And Ladders — Men At Work; We Can Work It Out — The Beatles
- **T3 Bruit** « t3_dz1015065_0 » → attendu *7 Seconds (feat. Neneh Cherry) — Youssou N'Dour* (rang 11) ; top-3 : While My Guitar Gently Weeps — Peter Frampton; DVNO — Justice; Kid OK — The Go Find
- **T3 Bruit** « t3_dz1015065_5 » → attendu *7 Seconds (feat. Neneh Cherry) — Youssou N'Dour* (rang 1) ; top-3 : 7 Seconds (feat. Neneh Cherry) — Youssou N'Dour; Rule The World — Walk Off The Earth; A.M. 180 — Grandaddy
- **T4 Reprises** « Me Gustas Tu » → attendu *Me Gustas Tu — Manu Chao* (rang 13) ; top-3 : Eyesdown — Bonobo; How R U Doin? — Aqua; Dutty Love — Don Omar
- **T4 Reprises** « Blinding Lights (Live) » → attendu *Blinding Lights — The Weeknd* (rang 12) ; top-3 : Not Afraid — Eminem; Thinkin Bout You — Frank Ocean; It Is What It Is — Lifehouse
- **T4 Reprises** « Sail (Originally Performed By Awol Nation) (Karaoke Version) » → attendu *Sail — Awolnation* (rang 1) ; top-3 : Sail — Awolnation; Starfish — Savant; Ghetto In The Sky — Aswad
- **T4 Reprises** « Stitches (Instrumental Version) » → attendu *Stitches — Shawn Mendes* (rang None) ; top-3 : Blacking Out The Friction — Death Cab For Cutie; Feels So Good — Chuck Mangione; Brothers On A Hotel Bed — Death Cab For Cutie
- **T4 Reprises** « when the party's over (Recorded at Spotify Studios NYC) » → attendu *when the party's over — Billie Eilish* (rang 1) ; top-3 : when the party's over — Billie Eilish; Latch — Disclosure; Tourner La Page — Zaho
