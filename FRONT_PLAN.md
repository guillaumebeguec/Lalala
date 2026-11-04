# FRONT_PLAN.md — Lalala, application de démo

Plan d'exécution du **front**. Le backend est décrit dans `PLAN.md`, le scénario de démo dans `DEMO.md`.

Référence visuelle : la maquette « Lalala — demo mockup » (3 écrans : Capture, Searching, Result). **Ce fichier en est la description fidèle : en cas de doute, il fait foi.**

Principe : le front se développe **en parallèle du backend**, sur un mode mock qui simule l'API (phases F0–F4). Le branchement sur la vraie API arrive ensuite (F5).

## Calendrier indicatif

| Jour | Phases front | Dépendance backend |
|---|---|---|
| J2 | F0, F1 | aucune |
| J3 | F2, F3 | aucune (mock) |
| J4 | F4 | aucune (mock) |
| J5 | F5 | `/search/stream` (PLAN.md phase 8) |
| J6 | F6, F7 | backend stable |
| J7 | F8 + répétitions | — |

---

## 1. Stack

- **Vite + React 18 + TypeScript** (strict).
- **Tailwind CSS 3**, avec les tokens ci-dessous dans `tailwind.config.ts` (aucune couleur en dur dans les composants).
- **framer-motion** pour les animations.
- Polices **auto-hébergées via `@fontsource`** (Bricolage Grotesque, DM Sans, DM Mono) : la démo doit fonctionner sans réseau.
- Pas de librairie de composants : une dizaine de composants maison suffisent.
- État : `useReducer` + une machine à états simple (§ 5), pas de Redux.
- Tests : Vitest + Testing Library (composants), Playwright (parcours complets en mode mock).
- Dossier `web/` à la racine du repo ; le backend FastAPI sert le build (`web/dist`) en production de démo.

---

## 2. Design tokens (repris de la maquette)

### Couleurs

| Token | Valeur | Usage |
|---|---|---|
| `bg` | `#15122B` | fond de page |
| `surface` | `#211C3D` | cartes, champ de saisie |
| `surface-strong` | `#2E2347` | carte « Original recording » |
| `chip` | `#2A2450` | pochettes vides, pastilles inactives |
| `border` | `#342D5C` | bordures standard, piste du lecteur |
| `border-strong` | `#4A3F7A` | boutons secondaires, cercle pointillé |
| `track` | `#3C3466` | partie non jouée de la forme d'onde |
| `text` | `#F7F3FF` | texte principal |
| `text-2` | `#DCD6F2` | sous-titres (artiste, année) |
| `muted` | `#B9B2D6` | libellés de section, détails |
| `dim` | `#9C94BF` | métadonnées (noms de modèles), étapes à venir |
| `source` | `#B4A9FF` | étiquettes de source (Audio, Web…) |
| `accent` | `#FF6FB5` | bouton d'écoute, lecture, mise en avant |
| `accent-soft` | `#FFC2E0` | libellé sur fond `surface-strong` |
| `yellow` | `#FFE066` | actions principales secondaires (Find it, That's the one), étapes terminées, badges |
| `on-accent` | `#15122B` | texte et icônes sur `accent` et `yellow` |

L'accent doit rester configurable (`--accent` en variable CSS) : alternatives prévues `#7CE0C3` et `#8B7CFF`.

### Typographie

| Rôle | Police | Graisse | Taille |
|---|---|---|---|
| Logo | Bricolage Grotesque | 800 | 28 px, letter-spacing −0,03em |
| H1 Capture | Bricolage Grotesque | 800 | 64 px, interligne 1,02, letter-spacing −0,04em |
| H1 Result | Bricolage Grotesque | 800 | 54 px |
| H1 Searching | Bricolage Grotesque | 800 | 46 px |
| Corps | DM Sans | 400 / 500 / 700 | 15–17 px |
| Libellés de section | DM Mono | 400, majuscules, letter-spacing 0,1em | 13 px |
| Métadonnées, badges | DM Mono | 400 / 500 | 11–13 px |

### Formes et espacements

- Rayons : boutons et champs en **pilule** (`999px`) ; cartes `24px` ; pochette principale `28px` ; lignes de liste `16–20px` ; vignettes `14px`.
- Bouton d'écoute : disque de 168 px dans une zone de 300 px (cercle pointillé `border-strong` + cercle `yellow` intérieur + 3 petits points décoratifs).
- Cibles tactiles ≥ 44 px partout.
- Largeur de contenu max : 1140–1180 px ; marges latérales 24–40 px.

### Motif de marque

Les **trois « la » en escalier** (rose, jaune, blanc, décalés de 0 / 7 / 14 px vers le haut) forment le logo. Le même motif en **trois points** sert d'animation d'attente (§ 6).

---

## 3. Écrans et textes

Les textes ci-dessous sont **définitifs** pour la démo ; les placer dans `src/copy.ts` (un seul endroit à modifier).

### 3.1 Capture (`/`)

- En-tête : logo seul.
- H1 : **You know… that one song?**
- Bouton d'écoute (icône micro, sans texte visible, `aria-label="Listen"`).
- Ligne de saisie : champ pilule, placeholder **“whistling, car ad, ~2008”**, `aria-label="Describe the song"`, bouton rond jaune flèche (`aria-label="Search"`) ; à droite deux boutons ronds icône : vidéo (`Add a video`), fichier audio (`Add an audio file`).
- Les entrées se **combinent** : on peut enregistrer puis ajouter une note texte avant d'envoyer. Dès qu'un enregistrement ou un fichier est présent, il s'affiche en pastille au-dessus du champ (« Recording · 10 s ✕ », « clip.mp4 ✕ »).
- Rien d'autre à l'écran : pas de sous-titre, pas de menu, pas de pied de page.

### 3.2 Searching (`/search/:id`)

- En-tête : logo + bouton texte **Cancel**.
- Colonne gauche :
  - trois points animés ;
  - H1 **Finding your song…** ; sous-titre **Usually takes a few seconds.** ;
  - carte **YOUR INPUT** : badge jaune (ex. **Background noise detected**, affiché seulement si le backend le signale), forme d'onde de l'enregistrement (partie « analysée » en accent), ligne de type (**Microphone recording · 10 s** / **Video clip · 6 s** / **Text description**), note de l'utilisateur (**Note: “…”**) si présente.
- Colonne droite, **SEARCH PROGRESS** : liste d'étapes alimentée par le backend (§ 4). Chaque étape : pastille d'état, titre, modèle (mono, `dim`), phrase de détail. États : terminé (pastille jaune ✓), en cours (fond `surface`, pastille accent ♪), à venir (titre `dim`, pastille `chip` ·), sauté (titre barré, détail « Skipped »).
- Titres attendus pour une entrée audio : *Isolating the music* (SAM Audio), *Exact match check* (Chromaprint), *Audio similarity search* (PE-AV · Gemini · CLAP), *Ranking candidates* (Llama), *Verification* (Web). Les titres viennent du backend ; ceux-ci servent aux fixtures mock.

### 3.3 Result (`/result/:id`)

- En-tête : logo + bouton texte **New search**.
- Bloc principal :
  - pochette 240 px (bordure accent) ;
  - badge jaune **Best match · high confidence** (ou **Best match · medium confidence** / **Possible match · low confidence** selon le score) ;
  - H1 titre ; ligne **Artiste · Album · Année** ;
  - lecteur de l'extrait (bouton accent, barre de progression, **0:10 / 0:30 preview**), **lecture automatique** à l'arrivée sur l'écran ;
  - **WHY THIS MATCH** : liste de justifications, une par ligne, étiquette de source à gauche (Audio, Fingerprint, Context, Web, Lyrics…) et phrase complète à droite ;
  - comparaison **Your recording** / **Original recording** (seulement si l'entrée est audio ou vidéo) : deux cartes avec lecteur ; une seule lecture à la fois ;
  - actions : **That’s the one** (jaune), **Not this one** (contour), lien **View score details**.
- Colonne droite **OTHER CANDIDATES** : 3 lignes (vignette, titre, artiste · année, bouton lecture). Un clic sur une ligne la promeut en résultat principal (animation d'échange).
- **View score details** ouvre un panneau latéral : tableau des scores et rangs par source pour chaque candidat, temps par étape, analyse du brief (réécriture, indices). C'est l'écran « preuve technique » pour le jury.

### 3.4 États supplémentaires (non dessinés, même langage visuel)

- **Recording** (sur l'écran Capture) : le bouton d'écoute devient un anneau de progression de 10 s, les cercles pulsent au rythme du niveau sonore, compteur **0:07** sous le bouton, bouton **Stop**.
- **No confident match** (écran Result) : H1 **We couldn’t pin it down.** ; sous-titre **Here are the closest songs we found.** ; liste des candidats ; bouton **Add a detail** qui renvoie à la Capture avec l'entrée conservée.
- **Error** : bandeau en haut, **Something went wrong. Try again.**, bouton **Retry** ; jamais de pile d'erreur à l'écran.
- **Not this one** : marque le candidat rejeté, promeut le suivant ; s'il n'y en a plus, bascule sur *No confident match*.

---

## 4. Contrat d'API attendu par le front

Le front n'appelle que ces routes. Si le backend (PLAN.md phase 8) diffère, **adapter le backend**, pas le front, ou le signaler.

- `POST /api/search/stream` (multipart : `text?`, `audio?`, `video?`, `options?`) → flux **SSE** :
  - `event: input` → `{ kind: "audio"|"video"|"text"|"mixed", duration_s?, noise_detected?, note?, waveform: number[] }` (forme d'onde réduite à ~64 valeurs 0–1, calculée côté backend) ;
  - `event: step` → `{ id, title, model, status: "pending"|"running"|"done"|"skipped", detail, elapsed_ms? }` (chaque mise à jour renvoie l'étape complète) ;
  - `event: result` → `{ query_id, confidence: "high"|"medium"|"low", best: Track, alternatives: Track[], proofs: {source, text}[], scores: ScoreRow[], timings: {step, ms}[], analysis: {...} }` ;
  - `event: error` → `{ message }`.
- `POST /api/search` : même entrée, réponse JSON finale seule (repli si SSE indisponible).
- `GET /api/preview/{track_id}` : extrait mp3 ; `GET /api/cover/{track_id}` : pochette (mise en cache locale côté backend, pour le mode hors ligne).
- `POST /api/feedback` → `{ query_id, track_id | null, verdict: "match"|"reject" }`.
- `GET /api/replay/{scene}` : rejoue une session enregistrée, avec le même flux SSE (§ 7).

Types TypeScript dans `src/api/types.ts` : `Track { id, title, artist, album?, year?, cover_url, preview_url, source: "catalog"|"web"|"fingerprint" }`, `ScoreRow { track_id, bm25?, emb: Record<string, {score, rank}>, fingerprint?, rerank?, final }`.

---

## 5. Architecture front

```
web/
├── index.html
├── tailwind.config.ts
├── src/
│   ├── main.tsx
│   ├── App.tsx              # routes : /, /search/:id, /result/:id
│   ├── copy.ts              # tous les textes
│   ├── state/
│   │   └── machine.ts       # idle → capturing → searching → result | noMatch | error
│   ├── api/
│   │   ├── types.ts
│   │   ├── client.ts        # fetch + parsing SSE (EventSource ne gère pas le POST : utiliser fetch + ReadableStream)
│   │   └── mock.ts          # simulateur d'API à partir des fixtures
│   ├── mocks/               # 3 scénarios : memory.json, cover.json, noisy.json + pochettes et extraits libres de droits
│   ├── audio/
│   │   ├── recorder.ts      # MediaRecorder + AnalyserNode (niveau en direct)
│   │   └── player.ts        # un seul lecteur actif à la fois
│   ├── components/
│   │   ├── Logo.tsx
│   │   ├── ListenButton.tsx # idle / recording / disabled
│   │   ├── BriefBar.tsx     # champ + bouton + icônes + pastilles d'entrées
│   │   ├── ThreeDots.tsx
│   │   ├── InputCard.tsx
│   │   ├── Waveform.tsx
│   │   ├── StepList.tsx
│   │   ├── TrackHero.tsx    # pochette + titre + lecteur
│   │   ├── PreviewPlayer.tsx
│   │   ├── ProofList.tsx
│   │   ├── CompareCards.tsx
│   │   ├── CandidateRow.tsx
│   │   ├── ScoreDrawer.tsx
│   │   └── Banner.tsx
│   └── screens/
│       ├── Capture.tsx
│       ├── Searching.tsx
│       └── Result.tsx
└── tests/
```

- `VITE_API_MODE=mock|live` choisit `mock.ts` ou `client.ts` ; même interface.
- Le mock rejoue les événements SSE des fixtures avec des délais réalistes (300–1 200 ms par étape).

---

## 6. Animations (framer-motion)

Sobres et rapides : rien ne doit ralentir la démo.

- **Bouton d'écoute** : au repos, légère respiration des cercles (échelle 1 → 1,03, 3 s) ; en enregistrement, les cercles suivent le niveau du micro, anneau de progression de 10 s.
- **Trois points** : rebond en séquence (rose, jaune, blanc), 900 ms par cycle.
- **Étapes** : chaque étape passe de « à venir » à « en cours » avec un fondu du fond ; la pastille ✓ apparaît avec un petit rebond.
- **Transition Searching → Result** : la pochette entre en scale 0,9 → 1 avec fondu (300 ms), puis le titre, puis les justifications en cascade (60 ms d'écart).
- **Promotion d'un autre candidat** : échange animé (`layoutId`) entre la ligne et le bloc principal.
- Respecter `prefers-reduced-motion` : fondus seuls.

---

## 7. Mode démo

- `?replay=memory|cover|noisy` : lance directement la session de secours correspondante via `/api/replay/{scene}` (ou les fixtures en mode mock). Aucune différence visible avec une vraie recherche.
- Menu discret de rejeu : un appui long (1 s) sur le logo ouvre une petite liste des scènes. Pas de raccourci clavier global.
- `?scale=1.25` : agrandit toute l'interface pour la projection.
- Tous les assets (polices, pochettes et extraits des cas de démo) sont servis en local : **la démo de secours fonctionne sans réseau**.
- Préchargement au démarrage : pochettes et extraits des scènes de démo.

---

## 8. Phases

### F0 — Mise en place (≈ 1 h)
Vite + React + TS, Tailwind avec les tokens du § 2, `@fontsource`, framer-motion, Vitest, Playwright, ESLint/Prettier. Proxy Vite `/api` → `http://localhost:8000`.
**Validation** : page vide avec le logo, aux bonnes couleurs et polices, hors ligne.

### F1 — Composants (≈ ½ jour)
Tous les composants du § 5 en version statique, avec une page `/_kit` (accessible seulement en dev) qui les montre dans tous leurs états.
**Validation** : comparaison visuelle avec la maquette ; contrastes vérifiés ; navigation clavier possible sur chaque bouton.

### F2 — Écrans statiques (≈ ½ jour)
Les trois écrans assemblés avec les données des fixtures, + états *No confident match* et *Error*.
**Validation** : rendu fidèle à la maquette à 1280 px, lisible à 390 px (colonnes empilées).

### F3 — Machine à états et mock (≈ ½ jour)
`machine.ts`, `mock.ts`, parsing SSE commun. Parcours complet en mode mock pour les trois scénarios.
**Validation** : test Playwright par scénario (Capture → Searching → Result), y compris *Not this one* et *No confident match*.

### F4 — Capture réelle (≈ ½ jour)
Enregistrement micro 10 s (MediaRecorder, format webm/opus sur Chrome, mp4/aac sur Safari : envoyer tel quel, le backend convertit), niveau en direct, dépôt de fichier audio/vidéo (glisser-déposer + sélecteur), combinaison avec la note texte, limites (vidéo ≤ 60 s, fichier ≤ 25 Mo) avec message clair.
**Validation** : enregistrement fonctionnel sur Chrome et Safari (macOS) ; permission refusée gérée proprement.

### F5 — Branchement API (≈ ½ jour)
`client.ts` sur `/api/search/stream`, `/api/feedback`, `/api/preview`, `/api/cover`. Repli sur `POST /api/search` si le flux échoue. Annulation (`AbortController`) sur **Cancel**.
**Validation** : les trois scènes de `DEMO.md` passent en mode live ; feedback écrit côté backend.

### F6 — Animations et finitions (≈ ½ jour)
Tout le § 6 ; lecture automatique de l'extrait ; un seul lecteur actif ; panneau **View score details**.
**Validation** : aucune animation ne retarde l'affichage du résultat de plus de 400 ms.

### F7 — Mode démo et robustesse (≈ ½ jour)
Tout le § 7 ; build servi par FastAPI ; test complet réseau coupé.
**Validation** : les trois `?replay=` fonctionnent hors ligne, sur la machine de démo, en plein écran.

### F8 — Tests et répétition (≈ ½ jour)
Suite Playwright verte en mock ; check-list de démo dans `DEMO.md` (navigateur, plein écran, volume, micro autorisé, onglet préchauffé).

---

## 9. Règles

- Aucun texte en dur dans les composants : tout passe par `copy.ts`.
- Aucune couleur en dur : tokens Tailwind ou variables CSS.
- Pas de `localStorage` pour des données importantes ; seulement l'accent choisi et l'échelle d'affichage.
- Les pochettes et extraits réels ne sont jamais commités ; les fixtures utilisent des visuels et sons libres de droits ou générés.
- Accessibilité : vrais `<button>` et `<label>`, `aria-label` sur les boutons icônes, focus visible (anneau jaune), contraste ≥ 4,5:1.
