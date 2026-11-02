# Recherche sans agent — sous-ensemble dev

Catalogue : 2000 morceaux ; embedders (dernier run) : hum2song ; BM25 : oui ; Chromaprint : oui ; audfprint : oui.

## T1 ToT_Music

43 cas ; 6 s ; systèmes : bm25, bm25:reformulation-article, gemma:text→text, gemma:text→audio, rrf(k=10), rrf(k=30), rrf(k=60), rrf-embeddings(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| bm25 | 0.023 | 0.047 | 0.070 | 0.043 |
| bm25:reformulation-article | 0.000 | 0.093 | 0.093 | 0.041 |
| gemma:text→text | 0.140 | 0.209 | 0.256 | 0.176 |
| gemma:text→audio | 0.047 | 0.070 | 0.116 | 0.073 |
| rrf(k=10) | 0.116 | 0.302 | 0.302 | 0.192 |
| rrf(k=30) | 0.116 | 0.302 | 0.302 | 0.191 |
| rrf(k=60) | 0.093 | 0.279 | 0.302 | 0.176 |
| rrf-embeddings(k=60) | 0.163 | 0.233 | 0.302 | 0.213 |

Détail de `rrf(k=10)` par type de demande (`query_types`) :

| Type | n | top-1 | top-5 | top-10 |
|---|---|---|---|---|
| descriptive | 43 | 0.116 | 0.302 | 0.302 |

Comparabilité (C6) : la ligne `bm25` reproduit la baseline de l'article (requête complète) sur le split test de ToT_Music, mais sur notre catalogue de 2124 morceaux au lieu du corpus WASABI complet : la tâche y est plus facile, les chiffres ne sont pas directement comparables.

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

## T6a Fredonnements CHAD

393 cas ; 4081 s ; systèmes : chromaprint, audfprint, melody, hum2song:audio→audio, rrf(k=10), rrf(k=30), rrf(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.000 | 0.003 | 0.003 | 0.001 |
| audfprint | 0.000 | 0.000 | 0.000 | 0.000 |
| melody | 0.196 | 0.265 | 0.298 | 0.229 |
| hum2song:audio→audio | 0.005 | 0.023 | 0.036 | 0.014 |
| rrf(k=10) | 0.003 | 0.018 | 0.025 | 0.009 |
| rrf(k=30) | 0.003 | 0.013 | 0.025 | 0.008 |
| rrf(k=60) | 0.003 | 0.013 | 0.025 | 0.008 |

Détail de `melody` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| fragment en début (< 60 s) du morceau | 201 | 0.169 | 0.214 |
| fragment en fin (> 150 s) du morceau | 51 | 0.235 | 0.373 |
| fragment en milieu (60–150 s) du morceau | 141 | 0.220 | 0.298 |

## T6b Fredonnements maison

6 cas ; 56 s ; systèmes : chromaprint, audfprint, melody, hum2song:audio→audio, rrf(k=10), rrf(k=30), rrf(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.167 | 0.333 | 0.333 | 0.222 |
| audfprint | 0.000 | 0.000 | 0.000 | 0.000 |
| melody | 1.000 | 1.000 | 1.000 | 1.000 |
| hum2song:audio→audio | 0.000 | 0.000 | 0.000 | 0.000 |
| rrf(k=10) | 0.000 | 0.167 | 0.333 | 0.111 |
| rrf(k=30) | 0.000 | 0.167 | 0.333 | 0.083 |
| rrf(k=60) | 0.000 | 0.167 | 0.333 | 0.083 |

Détail de `melody` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| synth | 6 | 1.000 | 1.000 |

## Exemples qualitatifs (fusion RRF k=60)

- **T6a Fredonnements CHAD** « t6a_616d2185f49090d4 » → attendu *Youngblood — 5 Seconds Of Summer* (rang None) ; top-3 : Tuesday's Gone — Lynyrd Skynyrd; The Host Of Seraphim — Dead Can Dance; Clear Air — Sevdaliza
- **T6a Fredonnements CHAD** « t6a_8ebf470b2c5dd9d9 » → attendu *Youngblood — 5 Seconds Of Summer* (rang None) ; top-3 : Heart-Shaped Box — Nirvana; I Like The Way — Jade; Poor Fool — Ike & Tina Turner
- **T6a Fredonnements CHAD** « t6a_9da4fcda15169156 » → attendu *Youngblood — 5 Seconds Of Summer* (rang None) ; top-3 : Pink Batman — Dan Deacon; Stubborn Love — The Lumineers; The Host Of Seraphim — Dead Can Dance
- **T6a Fredonnements CHAD** « t6a_0c9534df68e2f4ac » → attendu *I Will Survive (Single Version) — Gloria Gaynor* (rang None) ; top-3 : Say My Name — Odesza; Le baiser — Alain Souchon; Leaving The Past — Immortal Technique
- **T6a Fredonnements CHAD** « t6a_18fbe5b28d378b95 » → attendu *I Will Survive (Single Version) — Gloria Gaynor* (rang None) ; top-3 : Magdalene — Bear's Den; Bring It On — Nick Cave And The Bad Seeds; Only You — A. R. Rahman
- **T6b Fredonnements maison** « t6b_synth_dz585453_1 » → attendu *Crazy She Calls Me — Rod Stewart* (rang None) ; top-3 : Vámonos, Negrito — Natalia Lafourcade; Poor Fool — Ike & Tina Turner; Before You Go — Lewis Capaldi
- **T6b Fredonnements maison** « t6b_synth_dz585453_2 » → attendu *Crazy She Calls Me — Rod Stewart* (rang None) ; top-3 : Ma Liberté — Chimène Badi; Alibi — Thirty Seconds To Mars; Sail — Awolnation
- **T6b Fredonnements maison** « t6b_synth_dz638673_1 » → attendu *Greetings From Space — Amanda Jenssen* (rang None) ; top-3 : Story Of Your Life — Five For Fighting; Like A Wind — Bahamas; Get Home — Bastille
- **T6b Fredonnements maison** « t6b_synth_dz638673_2 » → attendu *Greetings From Space — Amanda Jenssen* (rang 3) ; top-3 : C'est dit — Calogero; We Can Work It Out — The Beatles; Greetings From Space — Amanda Jenssen
- **T6b Fredonnements maison** « t6b_synth_dz6640923_1 » → attendu *Jalopy Style — Citizen King* (rang None) ; top-3 : Savoir Qui Je Suis (Gotta Go My Own Way) — Lââm; Side To Side — Ariana Grande; If — Bread
