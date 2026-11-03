# Recherche sans agent — sous-ensemble dev

Catalogue : 2000 morceaux ; embedders (dernier run) : gemma, peav, coverhunter ; BM25 : oui ; Chromaprint : oui ; audfprint : oui.

## T1 ToT_Music

43 cas ; 23 s ; systèmes : bm25, bm25:reformulation-article, gemma:text→text, gemma:text→audio, peav:text→text, rrf(k=10), rrf(k=30), rrf(k=60), rrf-embeddings(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| bm25 | 0.070 | 0.093 | 0.093 | 0.080 |
| bm25:reformulation-article | 0.047 | 0.093 | 0.116 | 0.077 |
| gemma:text→text | 0.116 | 0.186 | 0.256 | 0.159 |
| gemma:text→audio | 0.047 | 0.070 | 0.116 | 0.073 |
| peav:text→text | 0.023 | 0.047 | 0.140 | 0.054 |
| rrf(k=10) | 0.116 | 0.302 | 0.349 | 0.191 |
| rrf(k=30) | 0.140 | 0.302 | 0.349 | 0.196 |
| rrf(k=60) | 0.140 | 0.256 | 0.326 | 0.192 |
| rrf-embeddings(k=60) | 0.140 | 0.256 | 0.326 | 0.193 |

Détail de `rrf(k=10)` par type de demande (`query_types`) :

| Type | n | top-1 | top-5 | top-10 |
|---|---|---|---|---|
| descriptive | 43 | 0.116 | 0.302 | 0.349 |

Comparabilité (C6) : la ligne `bm25` reproduit la baseline de l'article (requête complète) sur le split test de ToT_Music, mais sur notre catalogue de 2124 morceaux au lieu du corpus WASABI complet : la tâche y est plus facile, les chiffres ne sont pas directement comparables.

## T3 Bruit

30 cas ; 111 s ; systèmes : chromaprint, audfprint, gemma:audio→audio, peav:audio→audio, coverhunter:audio→audio, rrf(k=10), rrf(k=30), rrf(k=60), rrf-embeddings(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.867 | 0.900 | 0.900 | 0.875 |
| audfprint | 0.867 | 0.867 | 0.867 | 0.867 |
| gemma:audio→audio | 0.333 | 0.433 | 0.467 | 0.366 |
| peav:audio→audio | 0.300 | 0.467 | 0.500 | 0.371 |
| coverhunter:audio→audio | 0.233 | 0.300 | 0.367 | 0.274 |
| rrf(k=10) | 0.933 | 1.000 | 1.000 | 0.958 |
| rrf(k=30) | 0.933 | 0.967 | 0.967 | 0.947 |
| rrf(k=60) | 0.867 | 0.967 | 0.967 | 0.911 |
| rrf-embeddings(k=60) | 0.333 | 0.633 | 0.667 | 0.477 |

Détail de `rrf(k=10)` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| 0 | 10 | 0.800 | 1.000 |
| 10 | 10 | 1.000 | 1.000 |
| 5 | 10 | 1.000 | 1.000 |

## T4 Reprises

44 cas ; 462 s ; systèmes : chromaprint, audfprint, gemma:audio→audio, peav:audio→audio, coverhunter:audio→audio, rrf(k=10), rrf(k=30), rrf(k=60), rrf-embeddings(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.068 | 0.091 | 0.091 | 0.077 |
| audfprint | 0.000 | 0.023 | 0.023 | 0.011 |
| gemma:audio→audio | 0.182 | 0.250 | 0.250 | 0.206 |
| peav:audio→audio | 0.136 | 0.205 | 0.227 | 0.165 |
| coverhunter:audio→audio | 0.114 | 0.182 | 0.205 | 0.143 |
| rrf(k=10) | 0.182 | 0.295 | 0.318 | 0.221 |
| rrf(k=30) | 0.159 | 0.227 | 0.273 | 0.196 |
| rrf(k=60) | 0.159 | 0.205 | 0.250 | 0.189 |
| rrf-embeddings(k=60) | 0.114 | 0.205 | 0.250 | 0.164 |

Détail de `rrf(k=10)` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| acoustic | 6 | 0.000 | 0.167 |
| karaoke | 13 | 0.385 | 0.385 |
| live | 15 | 0.133 | 0.333 |
| piano | 10 | 0.100 | 0.200 |

## T6a Fredonnements CHAD

393 cas ; 2592 s ; systèmes : chromaprint, audfprint, melody, gemma:audio→audio, coverhunter:audio→audio, hum2song:audio→audio, rrf(k=10), rrf(k=30), rrf(k=60), rrf-embeddings(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.000 | 0.003 | 0.003 | 0.001 |
| audfprint | 0.000 | 0.000 | 0.000 | 0.000 |
| melody | 0.196 | 0.265 | 0.298 | 0.229 |
| gemma:audio→audio | 0.003 | 0.010 | 0.018 | 0.008 |
| coverhunter:audio→audio | 0.000 | 0.000 | 0.000 | 0.000 |
| hum2song:audio→audio | 0.005 | 0.023 | 0.036 | 0.014 |
| rrf(k=10) | 0.000 | 0.010 | 0.018 | 0.006 |
| rrf(k=30) | 0.000 | 0.005 | 0.010 | 0.004 |
| rrf(k=60) | 0.000 | 0.003 | 0.010 | 0.003 |
| rrf-embeddings(k=60) | 0.003 | 0.005 | 0.013 | 0.007 |

Détail de `melody` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| fragment en début (< 60 s) du morceau | 201 | 0.169 | 0.214 |
| fragment en fin (> 150 s) du morceau | 51 | 0.235 | 0.373 |
| fragment en milieu (60–150 s) du morceau | 141 | 0.220 | 0.298 |

## T6b Fredonnements maison

6 cas ; 48 s ; systèmes : chromaprint, audfprint, melody, gemma:audio→audio, coverhunter:audio→audio, hum2song:audio→audio, rrf(k=10), rrf(k=30), rrf(k=60), rrf-embeddings(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.167 | 0.333 | 0.333 | 0.222 |
| audfprint | 0.000 | 0.000 | 0.000 | 0.000 |
| melody | 1.000 | 1.000 | 1.000 | 1.000 |
| gemma:audio→audio | 0.000 | 0.000 | 0.000 | 0.000 |
| coverhunter:audio→audio | 0.000 | 0.000 | 0.000 | 0.000 |
| hum2song:audio→audio | 0.000 | 0.000 | 0.000 | 0.000 |
| rrf(k=10) | 0.000 | 0.000 | 0.167 | 0.027 |
| rrf(k=30) | 0.000 | 0.000 | 0.000 | 0.015 |
| rrf(k=60) | 0.000 | 0.000 | 0.000 | 0.014 |
| rrf-embeddings(k=60) | 0.000 | 0.000 | 0.000 | 0.000 |

Détail de `melody` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| synth | 6 | 1.000 | 1.000 |

## Exemples qualitatifs (fusion RRF k=60)

- **T3 Bruit** « t3_dz818256_0 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; Tom's Diner — Suzanne Vega; Let You Love Me — Rita Ora
- **T3 Bruit** « t3_dz818256_5 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; Tom's Diner — Suzanne Vega; Whenever — Cherish
- **T3 Bruit** « t3_dz818256_10 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; We Can Work It Out — The Beatles; Delta — C2C
- **T3 Bruit** « t3_dz1015065_0 » → attendu *7 Seconds (feat. Neneh Cherry) — Youssou N'Dour* (rang 12) ; top-3 : While My Guitar Gently Weeps — Peter Frampton; DVNO — Justice; Kid OK — The Go Find
- **T3 Bruit** « t3_dz1015065_5 » → attendu *7 Seconds (feat. Neneh Cherry) — Youssou N'Dour* (rang 1) ; top-3 : 7 Seconds (feat. Neneh Cherry) — Youssou N'Dour; Rule The World — Walk Off The Earth; Looking For Something — Era
- **T4 Reprises** « Me Gustas Tu » → attendu *Me Gustas Tu — Manu Chao* (rang 21) ; top-3 : Eyesdown — Bonobo; Pray — Lasgo; How R U Doin? — Aqua
- **T4 Reprises** « Blinding Lights (Live) » → attendu *Blinding Lights — The Weeknd* (rang 15) ; top-3 : Coming To Terms — Carolina Liar; Mr Writer — Stereophonics; Thinkin Bout You — Frank Ocean
- **T4 Reprises** « Sail (Originally Performed By Awol Nation) (Karaoke Version) » → attendu *Sail — Awolnation* (rang 1) ; top-3 : Sail — Awolnation; Pursuit — Gesaffelstein; Psycho — Post Malone
- **T4 Reprises** « Stitches (Instrumental Version) » → attendu *Stitches — Shawn Mendes* (rang None) ; top-3 : Feels So Good — Chuck Mangione; Burial On The Presidio Banks — This Will Destroy You; One — Apocalyptica
- **T4 Reprises** « when the party's over (Recorded at Spotify Studios NYC) » → attendu *when the party's over — Billie Eilish* (rang 21) ; top-3 : Latch — Disclosure; You & I — One Direction; Sheets — Damien Jurado
