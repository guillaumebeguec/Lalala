# Lalala — journal de bord du hackathon

*Meta Global AI Developer Hackathon, 26 octobre – 4 novembre 2026. Piste Indie, thème « Build for Your World ».*
*Un MacBook M2, une semaine, les modèles ouverts de Meta. Ce journal raconte le cheminement : la première idée, les fausses pistes, ce qui a été trouvé en cours de route et pourquoi.*

---

## Le point de départ

Tout le monde a vécu ce moment : une chanson dans la tête, impossible de retrouver son nom. Shazam ne sert à rien, il lui faut l'enregistrement exact. On a un air qu'on fredonne, trois mots de refrain, un souvenir de pub de voiture « vers 2008, avec un sifflement », une reprise entendue dans un bar, un bout de vidéo bruyant.

**La première piste**, un soir avant le kickoff : « un Shazam amélioré », un RAG sur un catalogue de chansons. On tape un souvenir, un modèle d'embedding cherche, un LLM répond. Elle a tenu une heure. Le jury dit explicitement qu'il ne veut pas de « RAG basique » ni de tableau de bord, et surtout un RAG ne sait rien faire d'un fredonnement. L'idée a basculé vers **un agent qui enquête** : plusieurs façons de chercher (le texte, le son, la mélodie, les paroles chantées), un vote entre elles, une confiance honnête, et une question posée à l'utilisateur quand ça hésite. Le nom de code est resté *SoundTrace* ; le produit s'est appelé *Lalala* parce que c'est ce qu'on fait quand on cherche un titre.

**Les contraintes qu'on s'est fixées dès le départ** : les modèles Meta au cœur (Llama, PE-AV, SAM Audio, FAISS), tout en local sur le Mac, jamais d'audio stocké, jamais de paroles stockées, et des mesures sur des jeux de test que l'agent ne voit jamais. La dernière règle a sauvé le projet plusieurs fois : sans elle, on aurait cru à des choses qui ne marchaient pas.

---

## Jeudi 29 octobre — « d'abord, de quoi mesurer »

La tentation du premier jour est d'écrire le pipeline. On a d'abord construit **ce qui permet de se tromper vite** : un catalogue et des jeux de test.

- **Le catalogue** : le corpus WASABI, 24 361 chansons avec métadonnées, extrait en flux sans les paroles. Les extraits audio de 30 s viennent de Deezer, avec iTunes en repli. Première surprise : les conditions Deezer interdisent de stocker l'audio. Décision prise en dix minutes et jamais regrettée : les extraits sont lus **en mémoire**, embarqués, puis oubliés. Seuls les vecteurs restent. Deuxième surprise : les URL d'extrait expirent en moins d'une heure, d'où un rafraîchissement à la demande.
- **Les jeux de test**, un par situation réelle : souvenirs flous (ToT_Music, des vraies demandes de Reddit), enregistrements bruités (recette rejouable avec les bruits MUSAN), reprises vers original, et 393 **vrais fredonnements** du jeu CHAD. Un utilisateur simulé, avec une cloison : il ne connaît pas la réponse attendue.
- Un sous-ensemble de développement de 2 000 chansons pour itérer vite, le complet pour la fin.

Idée venue ce jour-là, gardée pour plus tard : des **fiches** par chanson, mélange d'un résumé Wikipédia et d'une description générée par Llama (« ça sonne comment ? », « ça parle de quoi ? »), pour que la recherche texte ait autre chose que des titres à lire.

---

## Vendredi 30 octobre — tout brancher, bout à bout

Journée de construction. Le but était une chaîne complète, même médiocre, pour la voir fonctionner devant soi.

- Les **embedders** derrière une seule interface : PE-AV pour l'audio et le texte↔audio, EmbeddingGemma pour texte↔fiches, CLAP en comparaison, Gemini en option désactivable. FAISS pour les vecteurs, BM25 pour les mots, Chromaprint pour l'empreinte exacte.
- **Llama** en trois rôles : réécrire le souvenir en requête propre, reclasser les dix meilleurs candidats, et poser la question la plus utile quand l'agent hésite (croyance bayésienne, gain d'information).
- L'**API** en événements SSE, pour que l'écran montre l'enquête étape par étape avec le nom du modèle qui travaille. Le **front Lalala** : trois écrans, Capture, Searching, Result, testable au téléphone sur le réseau local pour que le micro soit un vrai micro.
- Première confrontation avec macOS : faiss et torch embarquent chacun leur OpenMP, l'API plantait en silence. `OMP_NUM_THREADS=1`, et on n'y touche plus.

Le soir, le fredonnement a eu sa première version : hum2song, un embedder de contour mélodique publié par des chercheurs. Sur six fredonnements, deux dans le top 5. On y croyait encore.

---

## Samedi 31 octobre — les chiffres parlent, on tranche

Premier rapport d'évaluation sur le panel. Il a réorganisé la semaine.

- **Le bruit** : l'empreinte retrouve 87 % des enregistrements bruités en premier, la fusion 97 % dans le top 10. Scène de démo acquise.
- **Les reprises** : 25 % dans le top 5 au mieux. L'idée de « jouer une reprise au piano et battre Shazam » ne tient pas sur des extraits de 30 s : une reprise et son original ne se recouvrent pas forcément sur la même fenêtre. Plutôt que forcer, on l'a écrit noir sur blanc dans le scénario de démo : scène non retenue.
- **Le fredonnement** : hum2song à 0,5 % de top-1 sur les 393 fredonnements réels. Tous les autres systèmes à zéro. Coup dur, mais c'était la raison d'avoir un jeu de test.

Huit idées de la nuit précédente, appliquées dans la journée : router par type d'entrée (un texte n'interroge l'index audio que s'il décrit un son), utiliser **Llama comme moteur de recherche par connaissance** (« quelles chansons correspondent à cette description ? », titres ensuite résolus dans le catalogue), des seuils par branche pour qu'une branche faible ne vote pas, une **fusion apprise** à la place du RRF à poids égaux, un thème de paroles dans les fiches, et une **empreinte audfprint** plus robuste au micro que Chromaprint.

**Le pivot du jour** est venu d'une observation simple : quand les gens « fredonnent », ils chantent souvent des mots. Si on a des mots, on a Whisper. Et si on a les mots, on peut chercher dans les paroles sans les stocker : des hachés non réversibles de 4-grammes de mots, calculés une fois depuis le corpus. Cinq top-1 sur douze extraits chantés, et tous les échecs sont des échecs de transcription, pas de recherche. Le fredonnement est devenu deux problèmes : **des mots** ou **une mélodie**.

En fin de journée, la fusion apprise entraînée : 28 % de top-5 sur les souvenirs flous contre 21 % en RRF. Pas spectaculaire, mais réel, et défendable. L'agent et ses questions, évalués sur quinze cas : les questions ne changent pas le résultat avec un Llama 8B. On a gardé la fonctionnalité, pas l'argument.

---

## Dimanche 1er novembre — pourquoi le fredonnement échoue, vraiment

La question du matin : hum2song échoue-t-il à cause de la voix humaine, ou de l'embedder ? Pour y répondre, on a fabriqué des **fredonnements synthétiques** : on extrait la mélodie de l'extrait lui-même, on la rend en « mmm » avec transposition, tempo, vibrato. Un fredonnement parfait, sans l'humain. Résultat : le suiveur de hauteur (RMVPE) retrouve la mélodie à 0,2 demi-ton près, et hum2song la rate quand même. **C'est l'embedder appris qui échoue, pas la voix.**

Alors on a enlevé l'embedder. **Alignement direct des contours** : la mélodie de la requête est glissée sur celle de chaque extrait, avec tolérance à la transposition (on chante dans sa tonalité), au tempo (0,8 à 1,25) et aux silences. Sur les 393 fredonnements réels : **16 % de top-1, 25 % de top-10**. Trente fois mieux. Et une lecture claire de ce qui manque : quand on regroupe par position du passage fredonné dans la chanson, les fragments de début de morceau sont les moins bien reconnus, parce que l'extrait de 30 s commence souvent au refrain. Le plafond, c'est la fenêtre, pas la méthode.

Même journée, **SAM Audio** : l'accès aux poids est arrivé. L'installation sur Mac a été un petit combat (le paquet officiel installe un module nommé `core`, comme le nôtre ; pas de roue macOS pour deux dépendances), réglé par un venv séparé et un portage MLX trois fois plus rapide. Puis la mesure, et une leçon : la séparation est générative, elle produit un audio plausible, pas l'enregistrement exact. Sur vingt cas bruités, l'audio isolé **dégrade** l'empreinte et les embeddings en moyenne, mais sauve un cas que rien d'autre n'attrape. D'où la politique : SAM Audio n'intervient que comme **seconde chance** pour l'empreinte, jamais comme remplaçant de l'audio brut. On aurait pu afficher « SAM Audio dans la boucle » et s'arrêter là ; les chiffres ont dit non.

Et une leçon sur **Llama 8B** : à partir du seul mot « guitare ? », il inventait « rock, voix masculine, tempo moyen » et cherchait ces mots dans les titres et les paroles. Deux corrections : un indice n'est gardé que s'il est dans les mots de l'utilisateur, et les mots qui décrivent un son (instruments, voix, tempo) interrogent les descriptions audio et l'index audio, jamais les titres ni les paroles. « Guitare » cherche désormais des morceaux qui sonnent guitare, pas des chansons qui s'appellent *Guitar*.

---

## Lundi 2 novembre — rendre l'app honnête

En testant au téléphone, un fredonnement sans mots tombait presque toujours sur « We couldn't pin it down ». Trois causes empilées, trouvées dans les journaux : Whisper hallucinait des paroles (« Thank you for watching ! ») sur un simple « mmm », les scores fusionnés étaient toujours « bas » parce qu'un score RRF n'a pas d'échelle, et une branche Llama à 60 s traînait encore dans l'API.

- **Des mots ou une mélodie** : détection de mots réels (probabilité de langue, qualité de transcription, liste d'hallucinations connues). Sans mots, la mélodie décide seule.
- **Une confiance qui veut dire quelque chose** : le score mélodique seul ne distingue pas un bon d'un mauvais résultat (0,73 contre 0,72). La **marge** entre le premier et le deuxième candidat, si. Calibrée sur les 393 fredonnements : au-delà d'une certaine marge, l'app a raison trois fois sur quatre et peut le dire.
- **Plus d'impasse** : en confiance basse, l'app propose le **top 10 à écouter** au lieu d'avouer l'échec. C'est souvent là que l'utilisateur dit « c'est celle-là ! ».
- **Une seconde fenêtre par chanson** : l'extrait iTunes tombe souvent sur un autre passage que l'extrait Deezer. 1 819 morceaux en ont une. Fredonnement : **16 % → 19,6 % de top-1**, 30 % de top-10. Le plafond de la fenêtre recule.

La nuit a été laissée au Mac, un modèle à la fois pour éviter la surchauffe : 2 000 fiches Llama avec thème des paroles, index texte et son, évaluation complète, un cross-encoder maison, la fusion apprise réentraînée.

---

## Mardi 3 novembre — bilan, dépôt, et ce qu'on ferait avec une semaine de plus

Le matin, lecture des résultats de la nuit. PE-AV audio avait saturé la mémoire du GPU à mi-chemin : libération entre les lots, reprise depuis le cache, index complet. Le cross-encoder maison **dégrade** le top-1 : écarté. Les thèmes de paroles n'ont rien changé sur les souvenirs flous : gardés dans les fiches, pas dans l'argumentaire. PE-AV seul est un peu sous Gemma sur l'audio, mais la fusion avec lui monte à **93 % de top-1 sur le bruit**.

| Situation | Résultat sur le panel dev | Ce qu'on en dit en démo |
|---|---|---|
| Souvenir flou | 14 % top-1, 30 % top-5 (fusion apprise) | une liste courte, et l'agent pose une question |
| Enregistrement bruité | 93 % top-1, 100 % top-5 | la scène forte |
| Reprise / live | 32 % top-10 | non retenue, dit tel quel |
| Fredonnement | 19,6 % top-1, 30 % top-10 | tient si le passage fredonné est dans la fenêtre |

**Ce qu'on a appris en une semaine**

1. Mesurer avant de construire a coûté une journée et en a économisé trois : trois idées séduisantes (hum2song, l'audio isolé pour la similarité, le cross-encoder) ont été écartées par les chiffres, pas par le goût.
2. Un petit LLM invente ; il faut l'ancrer dans les mots de l'utilisateur et séparer « ça sonne comment » de « ça parle de quoi ».
3. Quand un modèle appris échoue, fabriquer une entrée idéale (les fredonnements synthétiques) dit immédiatement si le problème est le modèle ou la donnée.
4. La confiance affichée doit être calibrée sur des cas réels, sinon l'interface ment dans les deux sens.
5. Les limites structurelles (la fenêtre de 30 s) méritent d'être nommées : c'est ce qui dit où chercher le prochain gain.

**Avec une semaine de plus**

- L'audio complet des quinze morceaux de démo (achat DRM-free) : la fenêtre disparaît, le fredonnement change de régime. Le code accepte déjà plusieurs fenêtres par morceau.
- Un **Llama 70B** par API pour le moteur de recherche par connaissance et le reranking listwise : tout est branché, il manque le modèle.
- Des enregistrements réels au micro (l'outil existe) à la place du bruit synthétique.
- La vidéo avec les images PE-AV, prévue, non mesurée.
