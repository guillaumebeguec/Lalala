# DEMO.md — Scénario de démo et stratégie de données

## 1. Le message à faire passer

> « Shazam reconnaît un **enregistrement**. SoundTrace retrouve une **chanson** : à partir d'un souvenir, d'une reprise ou d'un son capté dans le bruit. »

Le jury doit repartir avec trois images :
1. un agent qui **pose la bonne question** et dont on voit l'incertitude fondre ;
2. Shazam qui **échoue en direct** sur une reprise, SoundTrace qui réussit ;
3. un tableau de résultats chiffrés contre des baselines sérieuses.

Thème du hackathon visé : **Build for Your World** (problème vécu par tout le monde), avec une profondeur technique qui touche à « Push the Frontier » (recherche cross-modale, inférence bayésienne).

---

## 2. Storyboard (3 minutes)

| Temps | Scène | Ce qu'on montre | Ce qui doit marcher |
|---|---|---|---|
| 0:00–0:20 | **Le problème** | « Vous avez tous déjà eu un air en tête sans pouvoir le nommer. Shazam ne sert à rien sans le son. » Chiffre : des milliers de demandes de ce type sur r/TipOfMyTongue. | Une diapo |
| 0:20–1:20 | **Scène 1 — Le souvenir** | Une personne du jury décrit un morceau de mémoire. L'agent pose 1 à 3 questions en boutons. Le panneau montre les hypothèses et **la courbe d'entropie qui chute**. Proposition avec extrait de 30 s pour confirmer. | Agent + gain d'information + catalogue grand public |
| 1:20–2:10 | **Scène 2 — La reprise** | On joue une reprise piano ou acoustique d'un tube depuis un téléphone. **Shazam en direct sur un second téléphone : échec.** SoundTrace : le titre original en tête. | Recherche audio sémantique robuste aux reprises |
| 2:10–2:35 | **Scène 3 — Le bruit** (ou la vidéo, selon ce qui est le plus fiable en répétition) | Morceau joué avec du brouhaha de café ; SAM Audio isole la musique ; on entend l'avant/après ; match. Variante : extrait de pub vidéo de 5 s. | SAM Audio ou PE-AV vidéo |
| 2:35–3:00 | **Résultats et stack** | Tableau : SoundTrace contre BM25, embedding brut, empreinte audio (type Shazam). Diapo d'architecture : Llama, PE-AV, SAM Audio, FAISS (+ Gemini Embedding 2 en fusion). | `eval/` à jour |

**Règle d'or** : chaque scène a un **cas de secours répété** (bouton « rejouer » qui rejoue une session enregistrée) si le direct déraille.

### Scène 1 : comment la sécuriser
- Le jury choisit son morceau, donc le catalogue doit être **large côté grand public** (tubes 1970–2025, musiques de pubs et de séries connues en France et à l'international).
- Préparer 5 cas répétés qui marchent à coup sûr, à proposer si la personne du jury sèche (« Pensez à une pub, ou à un générique de série… »).
- Si la personne pense à un morceau hors catalogue, l'agent bascule sur la recherche web : la démo reste valable, le journal montre l'enquête.

### Scène 2 : pourquoi c'est la scène clé
- C'est le seul moment où l'on **bat un produit que tout le monde connaît, en direct**. L'empreinte audio (Shazam, Chromaprint) compare des enregistrements exacts : une reprise est un autre enregistrement. Un embedding sémantique compare des contenus musicaux.
- À valider absolument en phase 4 : si aucun embedder ne retrouve les originaux à partir de reprises (top-5 < 50 %), remplacer la scène 2 par la vidéo et ne pas promettre les reprises.
- Choisir pour la démo 3 reprises **instrumentales** (piano, guitare) de tubes très connus, où la mélodie est intacte mais le timbre totalement différent.

### Scène 3 : vidéo ou bruit
- Bruit : spectaculaire si on fait entendre l'avant/après SAM Audio. Risque : latence de SAM Audio sur Mac.
- Vidéo : simple et visuel. Risque : limité à 80 s avec audio côté Gemini, et dépendance au réseau.
- Décider en répétition (J6) sur la base de la fiabilité mesurée, pas de l'envie.

---


### Décisions prises sur les chiffres (29/10/2026, panel dev de 2 124 morceaux, `eval/retrieval.md`)

- **Scène 2 (reprise)** : seuil de 50 % en top-5 **non atteint**. Reprises T4 (44 cas) : Gemma audio 25 % top-5, CoverHunter 18 % top-5 (11 % top-1), meilleure fusion 32 % top-10 ; fenêtres glissantes CoverHunter : gain marginal. Cause structurelle : la reprise et l'original ne sont connus que par deux extraits de 30 s à des positions différentes du morceau. Conséquence : **ne pas promettre les reprises en direct** ; si on veut garder une reprise en démo, choisir à l'avance une paire dont les deux extraits se recouvrent (vérifiée à l'écoute) et la rejouer.
- **Scène 3 (bruit)** : solide sans SAM Audio sur le bruit synthétique MUSAN : empreinte (audfprint ou Chromaprint) 87 % top-1, fusion 97 % top-10 à 0/5/10 dB. Reste à valider sur les 20 vrais enregistrements de café (`data/manual/real_recordings/`) : c'est le vrai juge de « Shazam échoue, nous réussissons ».
- **Fredonnement (scène bonus, PLAN_COMPLEMENTS C4)** : les embedders généralistes sont à ~0 sur 393 fredonnements humains (CHAD). hum2song : sur 6 fredonnements face à 36 morceaux, 2/6 en top-5 avec le contour du mixage ; la séparation de voix par Demucs (54 s par extrait) fait **moins bien** (1/6) et les fenêtres glissantes n'apportent rien → contour du mixage, sans Demucs. Sur les 393 cas CHAD face au panel de 2 124 morceaux : hum2song **0,5 % top-1 / 2,3 % top-5 / 3,6 % top-10** (fragments en début de morceau : 3,5 % top-5), tous les autres systèmes à 0. **Décision : pas de scène « fredonnement pur »** avec des références de 30 s. Voies retenues : (a) **paroles chantées** → transcription Whisper → empreintes de paroles (hachés de n-grammes, 21 015 morceaux) + Llama + fiches : sur 12 extraits de 10 s de chant studio, 5 top-1, les 7 échecs sont des échecs de transcription ; en production dans l'API (détection voix seule calibrée : rappel 0,93, faux positifs 0,13 sur CHAD) ; (b) éventuel panel réduit préparé pour une scène bonus fredonnement.
- **Scène 1 (souvenir, texte)** : sur 43 demandes réelles ToT dont la réponse est dans le panel, EmbeddingGemma 2 texte→fiches 14 % top-1 / 26 % top-10, fusion 30 % top-10 ; BM25 7 % top-10. Un Llama plus gros comme moteur de recherche (recommandation 1) est le levier attendu.

---

## 3. Stratégie de données

### 3.1 Pourquoi pas FMA
FMA est libre mais ne contient **aucun morceau connu**. Or tout l'intérêt de la démo, c'est que le jury cherche des morceaux qu'il connaît. Il faut donc un **catalogue grand public**, dont on ne stocke que des métadonnées, des embeddings et des extraits mis en cache localement.

### 3.2 Catalogue grand public (cœur du système)

| Brique | Source | Usage | À vérifier |
|---|---|---|---|
| Métadonnées de chansons | **WASABI Song Corpus** (≈ 2 M chansons, métadonnées enrichies, liens vers des identifiants externes dont Deezer d'après sa documentation) | Titre, artiste, année, genre, identifiants | Licence, présence effective des identifiants Deezer |
| Extraits audio de 30 s | **API Deezer** (champ `preview` de l'endpoint de recherche) ; **iTunes Search API** en repli (`previewUrl`) | Embeddings audio, écoute de confirmation | Conditions d'utilisation, limites de débit |
| Contexte des morceaux | Wikipédia / Wikidata (résumé, année, usages notables) ; MusicBrainz pour dédoublonner | « Fiche morceau » texte | — |
| Usages pubs / séries / films | Recherche web au moment de la requête (Tunefind, sites de musiques de pub, forums) ; pas de scraping massif | L'agent s'en sert en direct | — |
| Paroles | **Pas stockées.** Fragments de paroles cherchés via le web au moment de la requête | Indice « je me souviens d'un bout de refrain » | — |

**Composition visée : 20 000 à 30 000 morceaux.**
1. Tous les morceaux réponses du jeu ToT_Music (voir 3.3) présents dans WASABI avec un extrait disponible : indispensable pour évaluer.
2. Tubes par décennie (1970–2025), France et international : charts, playlists éditoriales « best of » par année.
3. Musiques de pubs et de génériques connues en France (liste construite à la main en partie, ~300 titres).
4. Distracteurs : morceaux du même artiste ou du même genre que les précédents, pour que la recherche ne soit pas artificiellement facile.

**Fiche morceau** (texte indexé, un document par morceau) : titre, artiste, année, genre, résumé court (Wikipédia), usages connus, description audio générée une fois par Llama à partir des métadonnées (instruments probables, ambiance). Indexée en BM25 et en embedding texte.

**Coût Gemini (ordre de grandeur, à vérifier sur la grille officielle)** : l'audio compte environ 25 tokens par seconde, soit ~750 tokens par extrait de 30 s, soit ~20 M tokens pour 25 000 extraits. Utiliser l'API batch (moitié prix) et le cache.

### 3.3 Jeux de test

| Jeu | Contenu | Taille | Mesure | Rôle |
|---|---|---|---|---|
| **T1 — ToT_Music** | Demandes réelles de r/TipOfMyTongue avec réponse vérifiée, liées au corpus WASABI (Bhargav et al., SIGIR 2023) | ~2 300, échantillon de 200 pour l'éval agent | top-1 / top-3 / top-10, nb de questions | Crédibilité scientifique : de vraies descriptions humaines, pas des cas inventés |
| **T2 — Souvenirs FR** | Descriptions floues écrites par des proches et collègues, en français, sur des pubs, génériques, tubes | 50 | idem | La démo sera probablement en français ; CLAP ne comprend que l'anglais |
| **T3 — Bruit** | Extraits du catalogue mixés avec du brouhaha et du bruit (corpus libre **MUSAN**) à SNR 0, 5 et 10 dB, + 20 vrais enregistrements (enceinte de téléphone dans un café, captés par un autre téléphone) | 300 + 20 | top-5, avec / sans SAM Audio | Valider la scène 3 |
| **T4 — Reprises** | Reprises piano, acoustiques, live, karaoké de morceaux du catalogue, trouvées via la recherche Deezer (« titre piano cover », « titre acoustic ») | 200 | top-1 / top-5 de l'original ; **empreinte Chromaprint en baseline** | Prouver « on bat Shazam » |
| **T5 — Vidéo** | Extraits de 5–10 s de pubs ou génériques dont la musique est dans le catalogue | 20 | top-5 | Valider la variante vidéo |

**Utilisateur simulé** (pour évaluer l'agent sur T1 et T2 sans humain) : un Llama distinct de l'agent, qui ne connaît **que le texte de la demande d'origine**. Il répond aux questions de l'agent à partir de ce texte, et dit « je ne sais pas » sinon. Il ne voit jamais la réponse attendue. Une variante « utilisateur informé », qui connaît aussi l'année et la langue du vrai morceau, sert de borne haute ; les deux chiffres sont reportés.

**Point d'attention scientifique** : dans l'article ToT_Music, les reformulations de requêtes par LLM faisaient **moins bien** que la demande complète utilisée telle quelle. Notre réécriture (phase 5) doit donc être comparée honnêtement à la requête brute, et la fusion des deux est probablement la bonne réponse. C'est un résultat intéressant à montrer dans les deux sens.

### 3.4 Baselines à battre (tableau final)
1. **BM25** sur les fiches morceaux (la baseline de l'article ToT_Music).
2. **Embedding brut** de la requête, sans agent.
3. **Empreinte audio Chromaprint** (proxy de Shazam) pour T3 et T4.
4. **Agent baseline** (questions choisies librement par Llama) contre **agent à gain d'information**.

---

## 4. Ce qu'il faut préparer soi-même (hors Claude Code)

- J1 : clé Gemini, clé de recherche web (Tavily ou Brave), demandes d'accès aux poids PE-AV et SAM Audio.
- J1 : lire les conditions d'utilisation Deezer / iTunes pour l'usage des extraits ; vérifier les règles du hackathon sur l'usage de modèles non-Meta.
- J1–J3 : collecter les 50 descriptions françaises de T2 (un formulaire envoyé aux collègues : « décrivez de mémoire un morceau de pub ou de série, et donnez la réponse »).
- J5 : enregistrer les 20 vrais enregistrements bruités de T3 et choisir les 3 reprises de la scène 2.
- J6 : deux répétitions complètes chronométrées, avec Shazam installé sur le second téléphone.
