# Recherche sans agent — sous-ensemble dev

Catalogue : 2000 morceaux ; embedders (dernier run) : gemma, coverhunter, hum2song ; BM25 : oui ; Chromaprint : oui ; audfprint : oui.

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

393 cas ; 2203 s ; systèmes : chromaprint, audfprint, gemma:audio→audio, coverhunter:audio→audio, hum2song:audio→audio, rrf(k=10), rrf(k=30), rrf(k=60), rrf-embeddings(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.000 | 0.003 | 0.003 | 0.001 |
| audfprint | 0.000 | 0.000 | 0.000 | 0.000 |
| gemma:audio→audio | 0.003 | 0.010 | 0.018 | 0.008 |
| coverhunter:audio→audio | 0.000 | 0.000 | 0.000 | 0.000 |
| hum2song:audio→audio | 0.005 | 0.023 | 0.036 | 0.014 |
| rrf(k=10) | 0.000 | 0.010 | 0.018 | 0.006 |
| rrf(k=30) | 0.000 | 0.005 | 0.010 | 0.004 |
| rrf(k=60) | 0.000 | 0.003 | 0.010 | 0.003 |
| rrf-embeddings(k=60) | 0.003 | 0.005 | 0.013 | 0.007 |

Détail de `hum2song:audio→audio` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| fragment en début (< 60 s) du morceau | 201 | 0.010 | 0.035 |
| fragment en fin (> 150 s) du morceau | 51 | 0.000 | 0.000 |
| fragment en milieu (60–150 s) du morceau | 141 | 0.000 | 0.014 |

## Exemples qualitatifs (fusion RRF k=60)

- **T1 ToT_Music** « Alt/Electro song with few vocals – starts with a deep male voice singing "Go!"  After the "Go!" there's an instrumental part, after which the male voice joins a » → attendu *'64 Aka Go — Lemon Jelly* (rang 3) ; top-3 : From East To West — Enigma; The Voice Of Enigma — Enigma; '64 Aka Go — Lemon Jelly
- **T1 ToT_Music** « Sad country song  Sad country song about a 2 kids in school where one of them is being abused. I think story had to do with one of the kids giving away his lunc » → attendu *Alyssa Lies — Jason Michael Carroll* (rang 1) ; top-3 : Alyssa Lies — Jason Michael Carroll; Youth Of The Nation — P.O.D.; Mama's Broken Heart — Miranda Lambert
- **T1 ToT_Music** « 00s(?) Turkish Rock Song  I remember hearing this song in a Turkish music channel while vacationing in Istanbul back in 2006. The channel played Turkish and int » → attendu *Seninle Bir Dakika — Gece Yolcuları* (rang 1) ; top-3 : Seninle Bir Dakika — Gece Yolcuları; Something Good Can Work — Two Door Cinema Club; Space Lord — Monster Magnet
- **T1 ToT_Music** « looking for a song I’ve been hearing on this 2000s tv show  it goes something like:. I know it wasn’t you, yeah, yeah , yeah . And it probably something somethi » → attendu *Mercy — Duffy* (rang None) ; top-3 : I've Been Thinking About You — Londonbeat; Looking For Something — Era; Sweet Dreams (Are Made Of This) — Marilyn Manson
- **T1 ToT_Music** « Sounds like Me & Mr. Jones by Amy Winehouse  Particularly when she says “you made me miss the slick rick gig”. This is bothering me and I don’t even wanna sleep » → attendu *Chasing Pavements — Adele* (rang None) ; top-3 : She Wants To Dance With Me — Rick Astley; Me And Mrs. Jones — Michael Bublé; Believe In Me — Rick Springfield
- **T6a Fredonnements CHAD** « t6a_616d2185f49090d4 » → attendu *Youngblood — 5 Seconds Of Summer* (rang None) ; top-3 : The Host Of Seraphim — Dead Can Dance; Seasons (Waiting On You) — Future Islands; Brothers On A Hotel Bed — Death Cab For Cutie
- **T6a Fredonnements CHAD** « t6a_8ebf470b2c5dd9d9 » → attendu *Youngblood — 5 Seconds Of Summer* (rang None) ; top-3 : ObZen — Meshuggah; Dragon Chasers — Wax Tailor; Seasons (Waiting On You) — Future Islands
- **T6a Fredonnements CHAD** « t6a_9da4fcda15169156 » → attendu *Youngblood — 5 Seconds Of Summer* (rang None) ; top-3 : Pink Batman — Dan Deacon; Kid OK — The Go Find; Rakim — Dead Can Dance
- **T6a Fredonnements CHAD** « t6a_0c9534df68e2f4ac » → attendu *I Will Survive (Single Version) — Gloria Gaynor* (rang None) ; top-3 : Say My Name — Odesza; 100 — The Game; A Little Time — The Beautiful South
- **T6a Fredonnements CHAD** « t6a_18fbe5b28d378b95 » → attendu *I Will Survive (Single Version) — Gloria Gaynor* (rang None) ; top-3 : Say My Name — Odesza; Magdalene — Bear's Den; Bring It On — Nick Cave And The Bad Seeds
