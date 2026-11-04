# Learned fusion vs RRF — T1 test (43 queries, dev subset)

Training: 252 queries from the train/val splits; 52040 pairs, 144 positive. Branches: bm25:brut, bm25:réécrit, gemma:légendes→audio, gemma:texte→fiches, peav:texte→fiches.

| System | top-1 [95 % CI] | top-5 | top-10 | MRR |
|---|---|---|---|---|
| RRF k=60 | 0.093 [0.02, 0.19] | 0.209 | 0.279 | 0.151 |
| learned fusion (LR) | 0.140 [0.05, 0.26] | 0.302 | 0.302 | 0.206 |
| alone: bm25:réécrit | 0.070 [0.00, 0.16] | 0.116 | 0.163 | 0.100 |
| alone: bm25:brut | 0.070 [0.00, 0.16] | 0.093 | 0.093 | 0.080 |
| alone: gemma:légendes→audio | 0.023 [0.00, 0.07] | 0.023 | 0.047 | 0.029 |
| alone: peav:texte→fiches | 0.023 [0.00, 0.07] | 0.023 | 0.116 | 0.043 |
| alone: gemma:texte→fiches | 0.116 [0.02, 0.21] | 0.186 | 0.256 | 0.159 |

Coefficients (logistic regression, features standardised by construction):

| Feature | Coefficient |
|---|---|
| bm25:brut:inv_rank | -2.372999906539917 |
| bm25:brut:score | 2.2109999656677246 |
| bm25:réécrit:score | 2.13700008392334 |
| gemma:texte→fiches:score | 2.002000093460083 |
| gemma:légendes→audio:score | 1.437999963760376 |
| decade_match | 1.2300000190734863 |
| gemma:texte→fiches:inv_rank | 1.159000039100647 |
| input:text | -0.8790000081062317 |
| gemma:texte→fiches:present | 0.8479999899864197 |
| peav:texte→fiches:present | 0.8169999718666077 |
| bm25:réécrit:inv_rank | 0.6970000267028809 |
| gemma:légendes→audio:inv_rank | 0.40700000524520874 |
| peav:texte→fiches:score | -0.3919999897480011 |
| n_branches | 0.3630000054836273 |
| peav:texte→fiches:inv_rank | -0.3100000023841858 |
| gemma:légendes→audio:present | 0.18299999833106995 |
| bm25:brut:present | -0.14100000262260437 |
| bm25:réécrit:present | 0.1080000028014183 |
| language_match | 0.03099999949336052 |
| decade_adjacent | 0.028999999165534973 |
