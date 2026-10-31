# Recherche sans agent — sous-ensemble dev

Catalogue : 2000 morceaux ; embedders : clap, gemma ; BM25 : oui ; Chromaprint : oui.

## T1 ToT_Music

43 cas ; 68 s.

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| bm25 | 0.023 | 0.047 | 0.070 | 0.043 |
| bm25:reformulation-article | 0.000 | 0.093 | 0.093 | 0.041 |
| clap:text→text | 0.000 | 0.000 | 0.000 | 0.002 |
| clap:text→audio | 0.000 | 0.000 | 0.023 | 0.004 |
| gemma:text→text | 0.140 | 0.209 | 0.256 | 0.176 |
| gemma:text→audio | 0.047 | 0.070 | 0.116 | 0.073 |
| rrf(k=10) | 0.070 | 0.256 | 0.279 | 0.150 |
| rrf(k=30) | 0.047 | 0.233 | 0.302 | 0.141 |
| rrf(k=60) | 0.023 | 0.233 | 0.279 | 0.125 |
| rrf-embeddings(k=60) | 0.047 | 0.209 | 0.302 | 0.127 |

Détail de `rrf(k=30)` par type de demande (`query_types`) :

| Type | n | top-1 | top-5 | top-10 |
|---|---|---|---|---|
| descriptive | 43 | 0.047 | 0.233 | 0.302 |

Comparabilité (C6) : la ligne `bm25` reproduit la baseline de l'article (requête complète) sur le split test de ToT_Music, mais sur notre catalogue de 2124 morceaux au lieu du corpus WASABI complet : la tâche y est plus facile, les chiffres ne sont pas directement comparables.

## T3 Bruit

30 cas ; 191 s.

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.867 | 0.900 | 0.900 | 0.875 |
| clap:audio→audio | 0.167 | 0.267 | 0.267 | 0.214 |
| gemma:audio→audio | 0.333 | 0.433 | 0.467 | 0.366 |
| rrf(k=10) | 0.700 | 0.967 | 0.967 | 0.791 |
| rrf(k=30) | 0.633 | 0.867 | 0.967 | 0.715 |
| rrf(k=60) | 0.600 | 0.833 | 0.967 | 0.696 |
| rrf-embeddings(k=60) | 0.233 | 0.400 | 0.500 | 0.318 |

Détail de `rrf(k=10)` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| 0 | 10 | 0.500 | 0.900 |
| 10 | 10 | 0.800 | 1.000 |
| 5 | 10 | 0.800 | 1.000 |

## T4 Reprises

44 cas ; 324 s.

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.068 | 0.091 | 0.091 | 0.077 |
| clap:audio→audio | 0.000 | 0.023 | 0.023 | 0.008 |
| gemma:audio→audio | 0.182 | 0.250 | 0.250 | 0.206 |
| rrf(k=10) | 0.114 | 0.250 | 0.295 | 0.166 |
| rrf(k=30) | 0.068 | 0.159 | 0.295 | 0.116 |
| rrf(k=60) | 0.068 | 0.136 | 0.273 | 0.114 |
| rrf-embeddings(k=60) | 0.068 | 0.205 | 0.250 | 0.124 |

Détail de `gemma:audio→audio` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| acoustic | 6 | 0.333 | 0.333 |
| karaoke | 13 | 0.077 | 0.077 |
| live | 15 | 0.200 | 0.400 |
| piano | 10 | 0.200 | 0.200 |

## T6a Fredonnements CHAD

393 cas ; 1382 s.

| Système | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| chromaprint | 0.000 | 0.003 | 0.003 | 0.001 |
| clap:audio→audio | 0.000 | 0.005 | 0.008 | 0.003 |
| gemma:audio→audio | 0.003 | 0.010 | 0.018 | 0.008 |
| rrf(k=10) | 0.000 | 0.005 | 0.013 | 0.004 |
| rrf(k=30) | 0.000 | 0.003 | 0.005 | 0.003 |
| rrf(k=60) | 0.000 | 0.003 | 0.010 | 0.003 |
| rrf-embeddings(k=60) | 0.000 | 0.013 | 0.020 | 0.006 |

Détail de `rrf-embeddings(k=60)` par sous-groupe (SNR dB ou type) :

| Groupe | n | top-1 | top-5 |
|---|---|---|---|
| fragment à 0-14 s (début) | 6 | 0.000 | 0.000 |
| fragment à 1-14 s (début) | 3 | 0.000 | 0.000 |
| fragment à 1-15 s (début) | 3 | 0.000 | 0.000 |
| fragment à 10-22 s (début) | 3 | 0.000 | 0.000 |
| fragment à 10-23 s (début) | 3 | 0.000 | 0.000 |
| fragment à 10-26 s (début) | 3 | 0.000 | 0.000 |
| fragment à 100-109 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 103-115 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 105-119 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 105-120 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 108-122 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 108-123 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 11-20 s (début) | 3 | 0.000 | 0.000 |
| fragment à 11-21 s (début) | 3 | 0.000 | 0.000 |
| fragment à 112-123 s (milieu) | 3 | 0.000 | 0.333 |
| fragment à 116-126 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 122-133 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 13-25 s (début) | 3 | 0.000 | 0.000 |
| fragment à 13-29 s (début) | 3 | 0.000 | 0.000 |
| fragment à 130-140 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 134-143 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 136-150 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 139-151 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 14-24 s (début) | 3 | 0.000 | 0.000 |
| fragment à 14-27 s (début) | 6 | 0.000 | 0.000 |
| fragment à 145-159 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 146-155 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 148-163 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 150-160 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 152-166 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 155-163 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 16-26 s (début) | 3 | 0.000 | 0.000 |
| fragment à 16-28 s (début) | 3 | 0.000 | 0.000 |
| fragment à 161-173 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 170-178 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 176-185 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 178-190 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 18-28 s (début) | 3 | 0.000 | 0.000 |
| fragment à 184-200 s (fin) | 3 | 0.000 | 0.333 |
| fragment à 19-32 s (début) | 3 | 0.000 | 0.000 |
| fragment à 191-201 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 199-206 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 2-12 s (début) | 3 | 0.000 | 0.000 |
| fragment à 2-14 s (début) | 3 | 0.000 | 0.000 |
| fragment à 20-36 s (début) | 3 | 0.000 | 0.000 |
| fragment à 202-212 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 204-216 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 21-37 s (début) | 3 | 0.000 | 0.000 |
| fragment à 211-226 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 216-232 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 23-30 s (début) | 3 | 0.000 | 0.000 |
| fragment à 23-31 s (début) | 3 | 0.000 | 0.000 |
| fragment à 23-33 s (début) | 3 | 0.000 | 0.000 |
| fragment à 23-36 s (début) | 3 | 0.000 | 0.000 |
| fragment à 237-253 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 25-37 s (début) | 6 | 0.000 | 0.000 |
| fragment à 26-41 s (début) | 3 | 0.000 | 0.000 |
| fragment à 27-43 s (début) | 3 | 0.000 | 0.000 |
| fragment à 28-41 s (début) | 3 | 0.000 | 0.000 |
| fragment à 3-17 s (début) | 3 | 0.000 | 0.000 |
| fragment à 30-41 s (début) | 3 | 0.000 | 0.000 |
| fragment à 307-316 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 31-40 s (début) | 3 | 0.000 | 0.000 |
| fragment à 31-41 s (début) | 3 | 0.000 | 0.000 |
| fragment à 317-327 s (fin) | 3 | 0.000 | 0.000 |
| fragment à 32-40 s (début) | 3 | 0.000 | 0.000 |
| fragment à 32-45 s (début) | 3 | 0.000 | 0.000 |
| fragment à 33-49 s (début) | 3 | 0.000 | 0.000 |
| fragment à 36-47 s (début) | 3 | 0.000 | 0.000 |
| fragment à 36-48 s (début) | 3 | 0.000 | 0.000 |
| fragment à 36-52 s (début) | 3 | 0.000 | 0.000 |
| fragment à 37-45 s (début) | 3 | 0.000 | 0.000 |
| fragment à 37-46 s (début) | 3 | 0.000 | 0.000 |
| fragment à 38-46 s (début) | 3 | 0.000 | 0.000 |
| fragment à 39-55 s (début) | 3 | 0.000 | 0.000 |
| fragment à 4-13 s (début) | 3 | 0.000 | 0.000 |
| fragment à 4-20 s (début) | 3 | 0.000 | 0.000 |
| fragment à 41-52 s (début) | 3 | 0.000 | 0.000 |
| fragment à 50-61 s (début) | 9 | 0.000 | 0.111 |
| fragment à 50-63 s (début) | 3 | 0.000 | 0.000 |
| fragment à 50-65 s (début) | 3 | 0.000 | 0.000 |
| fragment à 51-60 s (début) | 3 | 0.000 | 0.000 |
| fragment à 52-60 s (début) | 3 | 0.000 | 0.000 |
| fragment à 52-61 s (début) | 3 | 0.000 | 0.333 |
| fragment à 53-66 s (début) | 3 | 0.000 | 0.000 |
| fragment à 54-64 s (début) | 3 | 0.000 | 0.000 |
| fragment à 55-66 s (début) | 3 | 0.000 | 0.000 |
| fragment à 57-68 s (début) | 3 | 0.000 | 0.000 |
| fragment à 58-71 s (début) | 3 | 0.000 | 0.000 |
| fragment à 59-68 s (début) | 3 | 0.000 | 0.000 |
| fragment à 59-73 s (début) | 6 | 0.000 | 0.167 |
| fragment à 6-19 s (début) | 3 | 0.000 | 0.000 |
| fragment à 62-71 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 62-75 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 62-77 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 63-75 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 64-73 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 64-77 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 68-78 s (milieu) | 6 | 0.000 | 0.000 |
| fragment à 70-84 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 71-80 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 71-86 s (milieu) | 6 | 0.000 | 0.000 |
| fragment à 72-86 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 74-82 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 74-84 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 75-83 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 77-87 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 78-86 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 78-87 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 78-89 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 8-16 s (début) | 3 | 0.000 | 0.000 |
| fragment à 8-22 s (début) | 3 | 0.000 | 0.000 |
| fragment à 80-92 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 82-90 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 84-92 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 84-97 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 87-98 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 88-98 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 90-100 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 91-101 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 92-100 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 92-102 s (milieu) | 3 | 0.000 | 0.000 |
| fragment à 95-105 s (milieu) | 3 | 0.000 | 0.000 |

## Exemples qualitatifs (fusion RRF k=60)

- **T1 ToT_Music** « Alt/Electro song with few vocals – starts with a deep male voice singing "Go!"  After the "Go!" there's an instrumental part, after which the male voice joins a » → attendu *'64 Aka Go — Lemon Jelly* (rang 4) ; top-3 : From East To West — Enigma; Homecoming — Kanye West; The Voice Of Enigma — Enigma
- **T1 ToT_Music** « Sad country song  Sad country song about a 2 kids in school where one of them is being abused. I think story had to do with one of the kids giving away his lunc » → attendu *Alyssa Lies — Jason Michael Carroll* (rang 3) ; top-3 : Record Year — Eric Church; Youth Of The Nation — P.O.D.; Alyssa Lies — Jason Michael Carroll
- **T1 ToT_Music** « 00s(?) Turkish Rock Song  I remember hearing this song in a Turkish music channel while vacationing in Istanbul back in 2006. The channel played Turkish and int » → attendu *Seninle Bir Dakika — Gece Yolcuları* (rang 2) ; top-3 : Love Is All — Roger Glover; Seninle Bir Dakika — Gece Yolcuları; Have You Any Sympathy — Cock Robin
- **T1 ToT_Music** « looking for a song I’ve been hearing on this 2000s tv show  it goes something like:. I know it wasn’t you, yeah, yeah , yeah . And it probably something somethi » → attendu *Mercy — Duffy* (rang None) ; top-3 : I've Been Thinking About You — Londonbeat; Pieces Of Me — Ashlee Simpson; Looking For Something — Era
- **T1 ToT_Music** « Sounds like Me & Mr. Jones by Amy Winehouse  Particularly when she says “you made me miss the slick rick gig”. This is bothering me and I don’t even wanna sleep » → attendu *Chasing Pavements — Adele* (rang None) ; top-3 : Mary Jane — Rick James; She Wants To Dance With Me — Rick Astley; Me And Mrs. Jones — Michael Bublé
- **T3 Bruit** « t3_dz818256_0 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; We Can Work It Out — The Beatles; Thinking Of You — Sister Sledge
- **T3 Bruit** « t3_dz818256_5 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; Snakes And Ladders — Men At Work; We Can Work It Out — The Beatles
- **T3 Bruit** « t3_dz818256_10 » → attendu *Lucky — Jason Mraz* (rang 1) ; top-3 : Lucky — Jason Mraz; Snakes And Ladders — Men At Work; We Can Work It Out — The Beatles
- **T3 Bruit** « t3_dz1015065_0 » → attendu *7 Seconds (feat. Neneh Cherry) — Youssou N'Dour* (rang 3) ; top-3 : Do You Want To Dance — McFadden & Whitehead; False Advertising — Bright Eyes; 7 Seconds (feat. Neneh Cherry) — Youssou N'Dour
- **T3 Bruit** « t3_dz1015065_5 » → attendu *7 Seconds (feat. Neneh Cherry) — Youssou N'Dour* (rang 5) ; top-3 : Rule The World — Walk Off The Earth; IV. Sweatpants — Childish Gambino; I Like The Way — Jade
