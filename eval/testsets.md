# Jeux de test — rapport de phase 2

Catalogue : 24361 morceaux, 22473 avec extrait.

## T1 — ToT_Music

200 demandes d'évaluation, 1971 en réserve (réponse dans le catalogue avec extrait).
Split de l'article dans l'échantillon d'éval : {'test': 200} (C1 : tiré du split `test`). 0 demandes ont plusieurs enregistrements acceptables (ISRC multiples).

| Décennie | Éval |
|---|---|
| 1960s | 5 |
| 1970s | 9 |
| 1980s | 19 |
| 1990s | 46 |
| 2000s | 78 |
| 2010s | 43 |

Types de demande (éval) : descriptive 200

`user_created` (C1) : 514 demandes dans tout ToT_Music ; 418 contiennent encore un lien vers un média de l'utilisateur (vocaroo…), dont l'accessibilité n'a pas été vérifiée (pas de téléchargement).

## T2 — Souvenirs FR

0 descriptions appariées au catalogue (fichier `data/manual/fr_memories.csv`).

## T3 — Bruit

300 cas (recette rejouable, audio mixé à la volée).

| Type | SNR | Cas |
|---|---|---|
| babble | 0 | 50 |
| babble | 10 | 50 |
| babble | 5 | 50 |
| noise | 0 | 50 |
| noise | 10 | 50 |
| noise | 5 | 50 |

## T4 — Reprises

200 paires reprise → original (146 originaux).

| Type | Paires |
|---|---|
| live | 65 |
| karaoke | 56 |
| piano | 51 |
| acoustic | 28 |

Contrôle manuel de 20 paires : à faire (voir `scripts/review_pairs.py`).

## T6a — Fredonnements CHAD

390 fredonnements → original du catalogue ; groupes CHAD 311, titre YouTube lu 294, titre/artiste parsés 268, dans le catalogue 131, avec extrait 130. Limite : position du fragment dans le morceau ≠ position de l'extrait Deezer (30 s).

## T5 — Vidéo

0 extraits dans `data/manual/video/`.
