# Fusion apprise contre RRF — T1 test (43 demandes, sous-ensemble dev)

Entraînement : 252 demandes des splits train/val ; 61371 paires, 148 positives. Branches : bm25:brut, bm25:réécrit, clap:légendes→audio, clap:texte→fiches, gemma:légendes→audio, gemma:texte→fiches.

| Système | top-1 [IC 95 %] | top-5 | top-10 | MRR |
|---|---|---|---|---|
| RRF k=60 | 0.070 [0.00, 0.16] | 0.209 | 0.256 | 0.139 |
| fusion apprise (LR) | 0.093 [0.02, 0.19] | 0.279 | 0.279 | 0.182 |
| seule : bm25:réécrit | 0.070 [0.00, 0.16] | 0.140 | 0.186 | 0.107 |
| seule : clap:légendes→audio | 0.000 [0.00, 0.00] | 0.000 | 0.000 | 0.002 |
| seule : bm25:brut | 0.023 [0.00, 0.07] | 0.047 | 0.070 | 0.043 |
| seule : gemma:texte→fiches | 0.140 [0.05, 0.26] | 0.209 | 0.256 | 0.176 |
| seule : clap:texte→fiches | 0.000 [0.00, 0.00] | 0.000 | 0.047 | 0.009 |
| seule : gemma:légendes→audio | 0.000 [0.00, 0.00] | 0.023 | 0.047 | 0.015 |

Coefficients (régression logistique, caractéristiques standardisées par construction) :

| Caractéristique | Coefficient |
|---|---|
| gemma:texte→fiches:score | 3.506999969482422 |
| clap:légendes→audio:score | -2.680000066757202 |
| bm25:brut:score | 2.25 |
| bm25:réécrit:inv_rank | 2.135999917984009 |
| gemma:légendes→audio:inv_rank | 1.8919999599456787 |
| bm25:réécrit:score | 1.4129999876022339 |
| clap:texte→fiches:inv_rank | 1.4129999876022339 |
| gemma:légendes→audio:score | 1.3880000114440918 |
| decade_match | 1.3020000457763672 |
| clap:légendes→audio:inv_rank | -0.9800000190734863 |
| input:text | -0.8659999966621399 |
| gemma:texte→fiches:present | 0.7670000195503235 |
| clap:texte→fiches:score | -0.4410000145435333 |
| bm25:brut:present | -0.35199999809265137 |
| clap:texte→fiches:present | -0.3310000002384186 |
| gemma:texte→fiches:inv_rank | -0.3199999928474426 |
| bm25:réécrit:present | 0.28299999237060547 |
| decade_adjacent | 0.2329999953508377 |
| clap:légendes→audio:present | -0.22100000083446503 |
| language_match | 0.2199999988079071 |
