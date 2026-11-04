# SAM Audio on T3 (dev subset, mlx engine)

20 synthetic cases (SNR ≤ 5 dB), 10 s. Isolation: 46 s per case on average (base model, Apple Silicon).

| Measure | Mixture | Isolated music |
|---|---|---|
| Mean SI-SDR vs clean preview | 2.4 dB | -0.7 dB |
| Exact fingerprint found | 80 % | 45 % |
| Fingerprint found (mixture **or** isolated) | | 85 % |
| Gemma top-1 | 15 % | 5 % |
| Gemma top-10 | 20 % | 15 % |

**Reading.** The separation is generative (flow matching): it produces plausible audio, not the exact recording,
which shifts the fingerprint landmarks (80 % → 45 %) and pushes the Gemma embedding away (top-1 15 % → 5 %).
It mainly gains on ambient noise (free-sound) and loses on speech (librivox, us-gov), where the singing voice is
sometimes treated as speech to be removed; one case collapses (−57 dB). The net gain exists for "raw **or** isolated"
(+1 case, 85 %), so isolation remains a **second chance for the fingerprint** and never replaces the raw audio for
similarity. Engine comparison on one case: official PyTorch/MPS 145 s, MLX 38 s, same quality (5,2 vs 5,3 dB).

App policy: isolation only runs when the recording is detected as noisy and has no exact fingerprint match; the isolated audio gives the fingerprint a second chance, then feeds the similarity search.

| Case | SNR | Noise | SI-SDR mix | SI-SDR iso | FP mix | FP iso | Gemma mix | Gemma iso |
|---|---|---|---|---|---|---|---|---|
| t3_dz818256_0 | 0 | free-sound | -0.1 | 8.1 | ✓ | ✓ | 1 | 1 |
| t3_dz818256_5 | 5 | free-sound | 4.9 | 0.8 | ✓ | – | 1 | – |
| t3_dz1015065_0 | 0 | free-sound | -0.0 | -57.5 | – | – | – | – |
| t3_dz1015065_5 | 5 | free-sound | 4.9 | 8.4 | ✓ | ✓ | – | – |
| t3_dz3090998_0 | 0 | us-gov | -0.0 | 4.6 | ✓ | ✓ | – | – |
| t3_dz3090998_5 | 5 | us-gov | 4.9 | 4.6 | ✓ | ✓ | – | – |
| t3_dz983830_0 | 0 | librivox | 0.0 | -1.0 | ✓ | – | – | – |
| t3_dz983830_5 | 5 | librivox | 5.0 | -0.0 | ✓ | – | – | – |
| t3_dz1240062762_0 | 0 | free-sound | -0.1 | 8.4 | – | ✓ | 11 | 2 |
| t3_dz1240062762_5 | 5 | free-sound | 4.9 | 7.3 | ✓ | ✓ | – | – |
| t3_dz936739_0 | 0 | free-sound | -0.1 | 0.7 | ✓ | ✓ | 5 | – |
| t3_dz936739_5 | 5 | free-sound | 4.9 | 1.7 | ✓ | ✓ | 1 | – |
| t3_dz1169689_0 | 0 | us-gov | -0.0 | 0.9 | ✓ | ✓ | – | 6 |
| t3_dz1169689_5 | 5 | us-gov | 5.0 | -3.6 | ✓ | – | – | – |
| t3_dz424565872_0 | 0 | librivox | -0.2 | 0.3 | ✓ | – | – | – |
| t3_dz424565872_5 | 5 | librivox | 4.8 | 1.1 | ✓ | – | – | – |
| t3_dz7856347_0 | 0 | us-gov | -0.1 | 0.6 | ✓ | – | – | – |
| t3_dz7856347_5 | 5 | us-gov | 4.9 | 1.6 | ✓ | – | – | – |
| t3_dz69419038_0 | 0 | free-sound | -0.3 | -1.1 | – | – | – | – |
| t3_dz69419038_5 | 5 | free-sound | 4.8 | 0.7 | – | – | – | – |
