# Recherche sans agent — sous-ensemble dev

Catalogue : 2000 morceaux ; embedders (dernier run) : gemma, peav, coverhunter, hum2song ; BM25 : oui ; Chromaprint : oui ; audfprint : oui.

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

30 cas ; 226 s ; systèmes : chromaprint, audfprint, melody, gemma:audio→audio, coverhunter:audio→audio, hum2song:audio→audio, rrf(k=10), rrf(k=30), rrf(k=60), rrf-embeddings(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.867 | 0.900 | 0.900 | 0.875 |
| audfprint | 0.867 | 0.867 | 0.867 | 0.867 |
| melody | 0.333 | 0.500 | 0.533 | 0.400 |
| gemma:audio→audio | 0.333 | 0.433 | 0.467 | 0.366 |
| coverhunter:audio→audio | 0.233 | 0.300 | 0.367 | 0.274 |
| hum2song:audio→audio | 0.000 | 0.000 | 0.000 | 0.002 |
| rrf(k=10) | 0.867 | 0.967 | 0.967 | 0.888 |
| rrf(k=30) | 0.833 | 0.867 | 0.933 | 0.862 |
| rrf(k=60) | 0.800 | 0.867 | 0.933 | 0.843 |
| rrf-embeddings(k=60) | 0.200 | 0.333 | 0.500 | 0.286 |

Détail de `rrf(k=10)` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| 0 | 10 | 0.800 | 0.900 |
| 10 | 10 | 0.900 | 1.000 |
| 5 | 10 | 0.900 | 1.000 |

## T4 Reprises

44 cas ; 370 s ; systèmes : chromaprint, audfprint, melody, gemma:audio→audio, coverhunter:audio→audio, hum2song:audio→audio, rrf(k=10), rrf(k=30), rrf(k=60), rrf-embeddings(k=60).

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.068 | 0.091 | 0.091 | 0.077 |
| audfprint | 0.000 | 0.023 | 0.023 | 0.011 |
| melody | 0.068 | 0.091 | 0.091 | 0.076 |
| gemma:audio→audio | 0.182 | 0.250 | 0.250 | 0.206 |
| coverhunter:audio→audio | 0.114 | 0.182 | 0.205 | 0.143 |
| hum2song:audio→audio | 0.023 | 0.068 | 0.068 | 0.038 |
| rrf(k=10) | 0.182 | 0.250 | 0.341 | 0.224 |
| rrf(k=30) | 0.159 | 0.227 | 0.250 | 0.197 |
| rrf(k=60) | 0.136 | 0.227 | 0.227 | 0.181 |
| rrf-embeddings(k=60) | 0.091 | 0.136 | 0.273 | 0.134 |

Détail de `gemma:audio→audio` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| acoustic | 6 | 0.333 | 0.333 |
| karaoke | 13 | 0.077 | 0.077 |
| live | 15 | 0.200 | 0.400 |
| piano | 10 | 0.200 | 0.200 |

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

- **T1 ToT_Music** « Alt/Electro song with few vocals – starts with a deep male voice singing "Go!"  After the "Go!" there's an instrumental part, after which the male voice joins a » → attendu *'64 Aka Go — Lemon Jelly* (rang 7) ; top-3 : From East To West — Enigma; The Voice Of Enigma — Enigma; I Shot The Sheriff — Warren G
- **T1 ToT_Music** « Sad country song  Sad country song about a 2 kids in school where one of them is being abused. I think story had to do with one of the kids giving away his lunc » → attendu *Alyssa Lies — Jason Michael Carroll* (rang 1) ; top-3 : Alyssa Lies — Jason Michael Carroll; Why Me — Kris Kristofferson; Tears Don't Care Who Cries Them — K.d. Lang
- **T1 ToT_Music** « 00s(?) Turkish Rock Song  I remember hearing this song in a Turkish music channel while vacationing in Istanbul back in 2006. The channel played Turkish and int » → attendu *Seninle Bir Dakika — Gece Yolcuları* (rang 7) ; top-3 : Something Good Can Work — Two Door Cinema Club; Beni Al — Pinhani; Do You Remember The First Time? — Pulp
- **T1 ToT_Music** « looking for a song I’ve been hearing on this 2000s tv show  it goes something like:. I know it wasn’t you, yeah, yeah , yeah . And it probably something somethi » → attendu *Mercy — Duffy* (rang None) ; top-3 : What I Am — Emma Bunton; Pull Up The Cork — Ini Kamoze; Think Of Me — Lloyd
- **T1 ToT_Music** « Sounds like Me & Mr. Jones by Amy Winehouse  Particularly when she says “you made me miss the slick rick gig”. This is bothering me and I don’t even wanna sleep » → attendu *Chasing Pavements — Adele* (rang None) ; top-3 : She Wants To Dance With Me — Rick Astley; Monkey Man — Amy Winehouse; Miss Independent — Ne-Yo
- **T3 Bruit** « t3_dz818256_0 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; Aucune Fille Au Monde — G-Squad; Rock That Body — Black Eyed Peas
- **T3 Bruit** « t3_dz818256_5 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; We Can Work It Out — The Beatles; Cola Song — Inna
- **T3 Bruit** « t3_dz818256_10 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; We Can Work It Out — The Beatles; Cola Song — Inna
- **T3 Bruit** « t3_dz1015065_0 » → attendu *7 Seconds (feat. Neneh Cherry) — Youssou N'Dour* (rang 16) ; top-3 : Collapse — Eminem; While My Guitar Gently Weeps — Peter Frampton; DVNO — Justice
- **T3 Bruit** « t3_dz1015065_5 » → attendu *7 Seconds (feat. Neneh Cherry) — Youssou N'Dour* (rang 1) ; top-3 : 7 Seconds (feat. Neneh Cherry) — Youssou N'Dour; Rule The World — Walk Off The Earth; Say My Name — Odesza
