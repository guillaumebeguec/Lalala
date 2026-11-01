# SAM Audio sur T3 (sous-ensemble dev, moteur mlx)

20 cas synthétiques (SNR ≤ 5 dB), 10 s. Isolation : 46 s par cas en moyenne (modèle base, Apple Silicon).

| Mesure | Mélange | Musique isolée |
|---|---|---|
| SI-SDR moyen vs extrait propre | 2.4 dB | -0.7 dB |
| Empreinte exacte retrouvée | 80 % | 45 % |
| Empreinte retrouvée (mélange **ou** isolé) | | 85 % |
| Gemma top-1 | 15 % | 5 % |
| Gemma top-10 | 20 % | 15 % |

**Lecture.** La séparation est générative (flow matching) : elle produit un audio plausible, pas l'enregistrement
exact, ce qui déplace les points de repère de l'empreinte (80 % → 45 %) et éloigne l'embedding Gemma (top-1 15 % → 5 %).
Elle gagne surtout sur les bruits d'ambiance (free-sound) et perd sur la parole (librivox, us-gov), où la voix chantée est
parfois traitée comme de la parole à retirer ; un cas s'effondre (−57 dB). Le gain net existe en « brut **ou** isolé »
(+1 cas, 85 %), donc l'isolation reste une **seconde chance pour l'empreinte** et ne remplace jamais l'audio brut pour la
similarité. Comparaison moteur sur un cas : officiel PyTorch/MPS 145 s, MLX 38 s, même qualité (5,2 vs 5,3 dB).

Politique de l'app : l'isolation n'est lancée que si l'enregistrement est détecté bruité et sans empreinte exacte ; l'audio isolé donne une seconde chance à l'empreinte puis sert à la similarité.

| Cas | SNR | Bruit | SI-SDR mél. | SI-SDR iso | Empr. mél. | Empr. iso | Gemma mél. | Gemma iso |
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
