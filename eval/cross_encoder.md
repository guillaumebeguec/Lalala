# In-house cross-encoder vs fusion — T1 test (43 queries, dev subset)

Base: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`; 252 groups (1 positive + ≤ 7 BM25/Gemma negatives), 2 epochs, listwise loss. Reranked candidates: top-20 from BM25 + Gemma.

| System | top-1 [95 % CI] | top-3 | top-5 | top-10 | MRR |
|---|---|---|---|---|---|
| fusion BM25 + Gemma (top-20) | 0.116 [0.05, 0.21] | 0.163 | 0.186 | 0.209 | 0.148 |
| + in-house cross-encoder | 0.093 [0.02, 0.19] | 0.116 | 0.140 | 0.209 | 0.118 |
