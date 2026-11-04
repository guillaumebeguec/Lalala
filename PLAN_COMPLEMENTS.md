# PLAN_COMPLEMENTS.md — Ajouts au plan Lalala

Ce fichier **complète** `PLAN.md`, `DEMO.md` et `FRONT_PLAN.md` sans les remplacer. Il s'appuie sur des projets existants repérés après la rédaction du plan : le jeu ToT_Music de Spotify Research, la piste TREC Tip-of-the-Tongue, le modèle de reprises CoverHunter, le jeu CHAD et le modèle de fredonnement hum2song.

**Règles de lecture**
- Lire ce fichier **après** `PLAN.md`, avant chaque phase citée ci-dessous.
- Chaque complément indique la phase à laquelle il se rattache (« s'ajoute à » ou « remplace »).
- En cas de contradiction, **ce fichier l'emporte**, uniquement sur le point concerné.
- Le nom du produit est **Lalala**. Les mentions « SoundTrace » dans les autres fichiers désignent le même projet ; ne pas renommer le code ni le dépôt.

## Récapitulatif

| # | Complément | Rattachement | Effort | Statut |
|---|---|---|---|---|
| C1 | Exploiter tous les champs de ToT_Music (ISRC, types, découpage, reformulations) | s'ajoute aux phases 1, 2, 5 | 2 h | Obligatoire |
| C2 | Branche audio « reprises » avec CoverHunter | s'ajoute aux phases 3, 4, 6 | ½ jour | Obligatoire |
| C3 | Jeu de test CHAD (reprises et fredonnements) | s'ajoute à la phase 2 | 2 h | Obligatoire |
| C4 | Fredonnement avec hum2song | **remplace** la phase 9.5 | ½ jour | Optionnel, recommandé |
| C5 | TTMR++ comme embedder texte → musique supplémentaire | s'ajoute aux phases 3, 4 | 2 h | Optionnel |
| C6 | Comparabilité avec la littérature (TREC ToT, ToT_Music) | s'ajoute aux phases 4, 10 | 1 h | Obligatoire |
| C7 | Impacts sur le front (étapes, sources, types d'entrée) | s'ajoute à `FRONT_PLAN.md` § 3, 4 | 1 h | Obligatoire si C2/C4 |

Impact calendrier : C1 et C3 en J2, C2 en J3, C4 en J6 (créneau de la phase 9), C5 seulement s'il reste du temps en J3, C6 en J3 et J7.

---

## C1 — Exploiter tous les champs de ToT_Music

Source : dépôt `spotify-research/tot` (données seules, licence CC0), fichier `data.jsonl.zip`, une demande par ligne.

**S'ajoute à la phase 1 (catalogue)**
- Récupérer les morceaux réponses **par ISRC** plutôt que par titre/artiste : champs `wasabi_isrc` et `spotify_resolved.items[].isrc`. Chercher l'extrait Deezer par ISRC (vérifier dans la documentation Deezer l'endpoint de recherche par ISRC), avec repli sur la recherche titre + artiste.
- Plusieurs enregistrements peuvent être valides pour une même demande (`spotify_resolved.items` est une liste) : stocker **tous les ISRC acceptables** par demande ; un résultat est correct s'il correspond à l'un d'eux.
- Rapport : part des réponses trouvées par ISRC, par recherche textuelle, et non trouvées.

**S'ajoute à la phase 2 (jeux de test)**
- Utiliser le champ `split` d'origine : l'échantillon d'évaluation T1 est tiré du `test` ; le réglage des hyperparamètres (ε, température, poids de fusion) se fait sur `train`/`val` uniquement.
- Conserver `query_types` (`descriptive`, `user_created`, `extant_media`) et les étiquettes de `annotations` (genre, époque, paroles, comparaison avec un artiste, incertitude…) pour **découper les résultats par type de demande** dans les rapports.
- `user_created` désigne vraisemblablement des demandes accompagnées d'un enregistrement fait par l'utilisateur ; vérifier dans l'article (Table 1) et, si c'est le cas, signaler combien de ces demandes ont un média encore accessible.

**S'ajoute à la phase 5 (réécriture)**
- Le champ `reformulations` contient les reformulations LLM de l'article (`keywords`, `reform_10`, `reform_25`, `reform_50`, `reform`). Les ajouter comme **baseline directe** dans le tableau de la phase 5 : requête brute, reformulations de l'article, notre réécriture, fusion.

**Validation** : `eval/catalog.md` indique la couverture par ISRC ; `eval/retrieval.md` contient un tableau par `query_types` et la ligne « reformulations de l'article ».

---

## C2 — Branche audio « reprises » avec CoverHunter

Constat : CLAP, PE-AV et Gemini Embedding 2 sont des embeddings **généralistes**, pas entraînés à reconnaître une même chanson sous des versions différentes (timbre, tempo, tonalité, structure). C'est l'objet de la *cover song identification*. Cette branche vise directement la scène 2 de `DEMO.md`.

**S'ajoute à la phase 3 (embeddings et index)**
1. `core/embedders/coverhunter.py` : audio → audio uniquement (`embed_text` lève `NotImplementedError`).
   - Partir du dépôt officiel `Liu-Feng-deeplearning/CoverHunter` et de sa suite `alanngnet/CoverHunterMPS`. **Lire les deux README** : vérifier si la suite apporte le support Apple Silicon (MPS), comment extraire les caractéristiques (CQT), et la licence du modèle pré-entraîné (entraîné sur SHS100K).
   - Si l'installation échoue sur Mac, s'arrêter et le signaler avant de chercher un autre modèle.
2. Index FAISS dédié sur les extraits du catalogue.
3. Point de vigilance : ces modèles sont pensés pour des morceaux entiers ; ici, requête et catalogue sont des extraits de 10 à 30 s, qui ne couvrent pas forcément le même passage. Tester deux stratégies : un embedding par extrait entier, et des embeddings par fenêtres glissantes (ex. 10 s, pas de 5 s) avec score = maximum des similarités.

**S'ajoute à la phase 4 (évaluation)**
- Évaluer CoverHunter seul, chaque embedder généraliste seul, et la fusion, sur T4 (reprises), T3 (bruit) et C3 (CHAD).
- La **décision scène 2** de la phase 4 se prend sur la meilleure configuration, CoverHunter inclus.

**S'ajoute à la phase 6 (outils de l'agent)**
- `audio_match` interroge CoverHunter en parallèle des autres embedders. Ordre de décision : empreinte Chromaprint forte → enregistrement exact ; sinon fusion pondérée, avec un poids CoverHunter plus fort quand l'empreinte a échoué (la requête est probablement une autre version).
- Justification renvoyée au front : source `Version`, texte du type « Same song, different version (cover or live). »

**Validation** : ligne CoverHunter dans `eval/retrieval.md` pour T3, T4 et CHAD ; temps d'inférence par requête sur Mac reporté.

---

## C3 — Jeu de test CHAD (reprises et fredonnements)

Source : CHAD (Covers and Hummings Aligned Dataset, ISMIR 2023), dépôt `amanteur/CHAD`. Environ 18 h de courts fragments, chacun marqué `original`, `cover` ou `humming`, regroupés par `group_id` et alignés dans le temps. Le sous-ensemble des fredonnements est publié sur Hugging Face ; les originaux et reprises sont référencés par identifiant YouTube.

**S'ajoute à la phase 2**
1. Télécharger le sous-ensemble **fredonnements** depuis Hugging Face et les métadonnées (`metadata/dataset.csv`). **Ne pas télécharger d'audio YouTube** sans avoir vérifié avec moi les conditions d'utilisation.
2. Retrouver les morceaux originaux dans notre catalogue (titre/artiste depuis les métadonnées, puis extrait Deezer). Ne garder que les groupes dont l'original est couvert.
3. Construire deux jeux :
   - **T6a Fredonnements** : fredonnement CHAD → morceau original du catalogue ;
   - **T6b Reprises CHAD** : seulement si les fragments de reprises sont accessibles sans téléchargement YouTube ; sinon, s'en tenir à T4.
4. Limite à documenter : le fragment fredonné ne tombe pas forcément dans les 30 s de l'extrait Deezer. Mesurer ce recouvrement quand les intervalles le permettent, et reporter les résultats séparément pour les cas où il est probable.

**Validation** : statistiques de T6a (et T6b si construit) dans `eval/testsets.md`.

---

## C4 — Fredonnement avec hum2song

**Remplace la phase 9.5** (« contour de hauteur + DTW »).

Source : modèle `luigir5/hum2song` sur Hugging Face, code `luiginotmario/hum2song` ; encodeur de contour mélodique d'environ 5,2 M paramètres, une seule tour pour les requêtes et les morceaux, embeddings de dimension 256, poids sous licence CC BY 4.0. Gère fredonnement, sifflement et chant avec de fausses paroles ; ne se sert pas des paroles et peine sur les morceaux sans mélodie vocale claire.

Tâches :
1. `core/embedders/hum2song.py` : extraction du contour mélodique (la carte du modèle utilise RMVPE), puis embedding. **Lire le README et la carte du modèle** avant de coder ; utiliser le checkpoint `e2b` recommandé pour la recherche ouverte.
2. Calculer les contours et embeddings des extraits du catalogue (tester avec et sans isolation préalable de la voix par Demucs) ; index FAISS dédié.
3. Routage : toutes les branches tournent sur une entrée audio ; ajouter une détection simple de « voix seule monophonique » (confiance de hauteur élevée, faible polyphonie) qui augmente le poids de hum2song dans la fusion.
4. Évaluer sur T6a (CHAD) et sur 20 fredonnements enregistrés par moi (`data/manual/hums/<track_id>.m4a`).

**Décision démo** : si le top-5 sur les fredonnements réels dépasse 50 %, proposer dans `eval/` une **scène bonus** « le jury fredonne » (à arbitrer avec moi contre la scène 3). Sinon, garder le fredonnement hors de la démo.

**Validation** : `eval/humming.md` avec résultats T6a et fredonnements réels, avec et sans Demucs.

---

## C5 — TTMR++ comme embedder texte → musique (optionnel)

TTMR++ (Doh et al.) est un modèle de recherche de musique par le texte, entraîné avec des descriptions enrichies par LLM et des métadonnées ; code et démo publiés par les auteurs.

- `core/embedders/ttmr.py` (texte et audio), à ajouter au tableau de la phase 4 et à la fusion.
- À faire seulement si CLAP est clairement le maillon faible côté texte → audio après la phase 4.
- Vérifier la licence des poids avant usage.

---

## C6 — Comparabilité avec la littérature

**S'ajoute à la phase 4**
- Sur le `split` test de ToT_Music, reproduire la baseline BM25 de l'article (requête complète) et placer nos résultats à côté, en précisant les différences de corpus (le nôtre est plus petit que WASABI complet : l'écrire explicitement, car la tâche y est plus facile).

**S'ajoute à la phase 10 (diapo résultats)**
- Une phrase de positionnement : la piste TREC Tip-of-the-Tongue 2025 (films, lieux, personnes) a montré que les approches hybrides (BM25 + embeddings denses + signaux LLM) suivies d'un reranker battent chaque modèle isolé ; Lalala applique ce schéma à la musique et y ajoute l'audio (empreinte, reprises, fredonnement).
- Citer ToT_Music (Bhargav, Schuth, Hauff, SIGIR 2023) comme source du jeu de test.

---

## C7 — Impacts sur le front

**S'ajoute à `FRONT_PLAN.md` § 3 et aux fixtures du mode mock** :
- Nouvelles étapes possibles dans *Search progress* (titres renvoyés par le backend) : **Version matching** (CoverHunter), **Melody matching** (hum2song).
- Nouvelles sources dans *Why this match* : **Version** (« Same song, different version (cover or live). »), **Melody** (« Your hummed melody matches the vocal line. »).
- Nouveau type d'entrée dans la carte *Your input* : **Humming · 8 s** quand le backend détecte une voix seule (champ `input.kind = "humming"` dans l'événement SSE `input`).
- Ajouter une fixture mock `humming.json` et la scène `?replay=humming`.

---

## Risques ajoutés

| Risque | Parade |
|---|---|
| CoverHunter ne tourne pas sur Mac ou ses poids ont une licence incompatible | Garder la fusion des embeddings généralistes ; signaler avant de chercher un autre modèle |
| Extraits de 30 s trop courts pour l'identification de reprises | Fenêtres glissantes ; choisir pour la démo des reprises dont le passage joué figure dans l'extrait |
| CHAD peu couvert par notre catalogue | Évaluer sur la partie couverte en le disant ; compléter par nos propres fredonnements |
| Fredonnements du jury trop approximatifs | Scène bonus seulement si les chiffres réels sont bons ; sinon, hors démo |
