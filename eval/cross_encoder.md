# Cross-encoder maison contre fusion — T1 test (43 demandes, sous-ensemble dev)

Base : `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` ; 252 groupes (1 positif + ≤ 7 négatifs BM25/Gemma), 2 époques, perte listwise. Candidats rerankés : top-20 de BM25 + Gemma.

| Système | top-1 [IC 95 %] | top-3 | top-5 | top-10 | MRR |
|---|---|---|---|---|---|
| fusion BM25 + Gemma (top-20) | 0.116 [0.05, 0.21] | 0.163 | 0.186 | 0.209 | 0.148 |
| + cross-encoder maison | 0.093 [0.02, 0.19] | 0.116 | 0.140 | 0.209 | 0.118 |
