# Lalala — journal du hackathon

Meta Global AI Developer Hackathon, 26 octobre – 4 novembre 2026, piste Indie, thème « Build for Your World ».
Un MacBook M2 24 Go, les modèles ouverts de Meta (Llama 3.1, PE-AV, SAM Audio, FAISS), une semaine.

## Avant de commencer

L'idée de départ : un Shazam qui marche quand on n'a pas l'enregistrement. Tout le monde a eu une chanson en tête sans
retrouver le titre : un air qu'on fredonne, trois mots du refrain, « la pub de voiture avec un sifflement, vers 2008 »,
une reprise dans un bar, une vidéo bruyante.

Première piste, la veille du kickoff : un RAG sur un catalogue de chansons. On tape le souvenir, un embedding cherche,
un LLM répond. Abandonnée dans l'heure : le jury ne veut pas de « RAG basique », et un RAG ne fait rien d'un fredonnement.
Nouvelle idée : un agent qui enquête. Plusieurs façons de chercher (texte, son, mélodie, paroles chantées), un vote,
une confiance calibrée, et une question à l'utilisateur quand ça hésite. Nom de code SoundTrace, nom du produit Lalala.

Règles qu'on se donne : modèles Meta au cœur, tout en local, pas d'audio stocké, pas de paroles stockées, et des jeux de
test que l'agent ne voit jamais. Cette dernière règle a évité plusieurs fausses joies.

## Jeudi 29 octobre

Pas de pipeline aujourd'hui, d'abord de quoi mesurer.

- Catalogue : WASABI 2.0, 24 361 chansons, extrait en flux sans les paroles. Extraits de 30 s via l'API Deezer,
  iTunes en repli. Les conditions Deezer interdisent de stocker l'audio : les extraits sont lus en mémoire, embarqués,
  jetés. On ne garde que les vecteurs. Les URL signées expirent en moins d'une heure, rafraîchissement à la demande.
- Jeux de test, un par situation : souvenirs flous (ToT_Music, vraies demandes Reddit), enregistrements bruités (recette
  MUSAN rejouable, 0/5/10 dB), reprises → original, 393 fredonnements réels (CHAD). Utilisateur simulé qui ne connaît
  pas la réponse.
- Sous-ensemble dev de 2 000 chansons pour itérer, complet pour la fin.
- Idée notée pour plus tard : des fiches par chanson (résumé Wikipédia + description Llama « ça sonne comment, ça parle
  de quoi »), pour que la recherche texte ait autre chose que des titres.

## Vendredi 30 octobre

Journée de plomberie, objectif : une chaîne complète même médiocre.

- Embedders derrière une interface commune : PE-AV (audio, texte↔audio), EmbeddingGemma (texte↔fiches), CLAP pour
  comparer, Gemini en option désactivable. FAISS pour les vecteurs, BM25 pour les mots, Chromaprint pour l'empreinte.
- Llama en trois rôles : réécrire le souvenir en requête propre, reclasser le top 10, poser la question la plus utile
  (croyance bayésienne, gain d'information).
- API en SSE pour afficher l'enquête étape par étape avec le modèle qui travaille. Front Lalala : Capture, Searching,
  Result. Test au téléphone sur le réseau local pour avoir un vrai micro.
- macOS : faiss et torch embarquent chacun leur OpenMP, l'API plantait sans message. OMP_NUM_THREADS=1, réglé.
- Le soir, première version du fredonnement avec hum2song (embedder de contour publié). 2 sur 6 dans le top 5. On y croit.

## Samedi 31 octobre

Premier rapport d'éval sur le panel dev. Il a réorganisé la semaine.

- Bruit : empreinte 87 % top-1, fusion 97 % top-10. Scène acquise.
- Reprises : 25 % top-5 au mieux. L'idée « jouer une reprise au piano et battre Shazam » ne tient pas sur des extraits de
  30 s, la reprise et l'original ne se recouvrent pas sur la même fenêtre. Écrit dans DEMO.md : scène non retenue.
- Fredonnement : hum2song à 0,5 % top-1 sur les 393 fredonnements réels. Tout le reste à zéro.

Huit idées de la nuit, appliquées dans la journée : routage par type d'entrée (un texte n'interroge l'index audio que
s'il décrit un son), Llama comme moteur de recherche par connaissance (titres proposés puis résolus dans le catalogue),
seuils par branche, fusion apprise à la place du RRF à poids égaux, thème des paroles dans les fiches, empreinte
audfprint plus robuste au micro que Chromaprint.

Pivot du jour : quand les gens « fredonnent », ils chantent souvent des mots. Des mots → Whisper. Et on peut chercher
dans les paroles sans les stocker : hachés non réversibles de 4-grammes, calculés une fois depuis le corpus. 5 top-1 sur
12 extraits chantés, les échecs sont tous des échecs de transcription. Le fredonnement devient deux problèmes : des mots,
ou une mélodie.

Fin de journée : fusion apprise entraînée, 28 % top-5 sur les souvenirs flous contre 21 % en RRF. L'agent et ses
questions sur 15 cas : aucun gain avec un 8B. Fonctionnalité gardée, pas l'argument.

## Dimanche 1er novembre

Question du matin : hum2song échoue à cause de la voix ou de l'embedder ? Pour trancher, fredonnements synthétiques :
on extrait la mélodie de l'extrait, on la rend en « mmm » avec transposition, tempo, vibrato. RMVPE la retrouve à 0,2
demi-ton près, hum2song la rate quand même. C'est l'embedder.

Donc on l'enlève. Alignement direct des contours : la mélodie de la requête glissée sur celle de chaque extrait, tolérance
à la transposition, au tempo (0,8–1,25) et aux silences. Sur les 393 fredonnements : 16 % top-1, 25 % top-10, trente fois
mieux. Et un diagnostic clair : par position du passage fredonné, les débuts de morceau sont les moins reconnus parce que
l'extrait de 30 s démarre souvent au refrain. Le plafond est la fenêtre, pas la méthode.

SAM Audio, accès aux poids reçu. Installation pénible sur Mac (le paquet installe un module `core` comme le nôtre, deux
dépendances sans roue macOS) : venv séparé, sous-processus, portage MLX trois fois plus rapide. Mesure sur 20 cas
bruités : la séparation est générative, l'audio isolé dégrade l'empreinte et les embeddings en moyenne, mais sauve un
cas que rien d'autre n'attrape. Politique : seconde chance pour l'empreinte seulement, jamais à la place de l'audio brut.

Llama 8B : à partir de « guitare ? » il inventait « rock, voix masculine » et cherchait ces mots dans les titres et les
paroles. Deux corrections : un indice n'est gardé que s'il est dans les mots de l'utilisateur, et les mots qui décrivent
un son interrogent les descriptions audio et l'index audio, pas les titres ni les paroles.

## Lundi 2 novembre

Au téléphone, un fredonnement sans mots tombait presque toujours sur « We couldn't pin it down ». Trois causes dans les
journaux : Whisper hallucine des paroles (« Thank you for watching! ») sur un « mmm », les scores RRF sont toujours
« bas » parce qu'ils n'ont pas d'échelle, et une branche Llama à 60 s traînait encore dans l'API.

- Mots ou mélodie : détection de mots réels (probabilité de langue, qualité de transcription, liste d'hallucinations).
  Sans mots, la mélodie décide seule.
- Confiance : le score mélodique ne sépare pas un bon d'un mauvais résultat (0,73 contre 0,72). La marge entre le premier
  et le deuxième, si. Calibrée sur les 393 fredonnements.
- Plus d'impasse : en confiance basse, top 10 à écouter au lieu d'un échec.
- Seconde fenêtre par chanson : l'extrait iTunes tombe souvent sur un autre passage que Deezer. 1 819 morceaux en ont
  une. Fredonnement : 16 % → 19,6 % top-1, 30 % top-10.

Nuit de calcul, un modèle à la fois pour ne pas faire chauffer le Mac : 2 000 fiches Llama, index texte et son, éval
complète, cross-encoder maison, fusion réentraînée.

## Mardi 3 novembre

Lecture de la nuit. PE-AV audio avait saturé la mémoire GPU à mi-chemin : libération entre les lots, reprise depuis le
cache. Cross-encoder maison : dégrade le top-1, écarté. Thèmes de paroles : aucun gain sur les souvenirs flous, gardés
dans les fiches, pas dans l'argumentaire. PE-AV seul un peu sous Gemma sur l'audio, mais la fusion avec lui monte à 93 %
top-1 sur le bruit.

| Situation | Panel dev | En démo |
|---|---|---|
| Souvenir flou | 14 % top-1, 30 % top-5 (fusion apprise) | liste courte, l'agent pose une question |
| Enregistrement bruité | 93 % top-1, 100 % top-5 | scène forte |
| Reprise / live | 32 % top-10 | non retenue |
| Fredonnement | 19,6 % top-1, 30 % top-10 | tient si le passage est dans la fenêtre |

Nettoyage du dépôt, README pour le jury, publication sur GitHub.

## Ce qu'on retient

- Mesurer d'abord a coûté une journée et en a économisé trois : hum2song, l'audio isolé pour la similarité et le
  cross-encoder sont partis sur les chiffres.
- Un petit LLM invente. Il faut l'ancrer dans les mots de l'utilisateur et séparer « ça sonne comment » de « ça parle de quoi ».
- Quand un modèle appris échoue, fabriquer une entrée idéale dit tout de suite si c'est le modèle ou la donnée.
- La confiance affichée se calibre sur des cas réels, sinon l'interface ment.
- Nommer la limite structurelle (la fenêtre de 30 s) dit où est le prochain gain.

## Avec une semaine de plus

- L'audio complet des 15 morceaux de démo (achat DRM-free) : la fenêtre disparaît. Le code accepte déjà plusieurs
  fenêtres par morceau.
- Un Llama 70B par API pour le moteur de recherche par connaissance et le reranking listwise : branché, pas de modèle.
- Des enregistrements réels au micro à la place du bruit synthétique (l'outil existe).
- La vidéo avec les images PE-AV, prévue, pas mesurée.
