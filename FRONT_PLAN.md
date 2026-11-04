# FRONT_PLAN.md — Lalala, demo application

Execution plan for the **front end**. The backend is described in `PLAN.md`, the demo scenario in `DEMO.md`.

Visual reference: the "Lalala — demo mockup" (3 screens: Capture, Searching, Result). **This file is its faithful description: when in doubt, this file is authoritative.**

Principle: the front end is developed **in parallel with the backend**, on a mock mode that simulates the API (phases F0–F4). Wiring to the real API comes afterwards (F5).

## Indicative schedule

| Day | Front phases | Backend dependency |
|---|---|---|
| D2 | F0, F1 | none |
| D3 | F2, F3 | none (mock) |
| D4 | F4 | none (mock) |
| D5 | F5 | `/search/stream` (PLAN.md phase 8) |
| D6 | F6, F7 | stable backend |
| D7 | F8 + rehearsals | — |

---

## 1. Stack

- **Vite + React 18 + TypeScript** (strict).
- **Tailwind CSS 3**, with the tokens below in `tailwind.config.ts` (no hard-coded colors in components).
- **framer-motion** for animations.
- Fonts **self-hosted via `@fontsource`** (Bricolage Grotesque, DM Sans, DM Mono): the demo must work without network.
- No component library: about ten in-house components are enough.
- State: `useReducer` + a simple state machine (§ 5), no Redux.
- Tests: Vitest + Testing Library (components), Playwright (full flows in mock mode).
- `web/` folder at the repo root; the FastAPI backend serves the build (`web/dist`) in demo production.

---

## 2. Design tokens (taken from the mockup)

### Colors

| Token | Value | Usage |
|---|---|---|
| `bg` | `#15122B` | page background |
| `surface` | `#211C3D` | cards, input field |
| `surface-strong` | `#2E2347` | "Original recording" card |
| `chip` | `#2A2450` | empty cover art, inactive chips |
| `border` | `#342D5C` | standard borders, player track |
| `border-strong` | `#4A3F7A` | secondary buttons, dotted circle |
| `track` | `#3C3466` | unplayed part of the waveform |
| `text` | `#F7F3FF` | main text |
| `text-2` | `#DCD6F2` | subtitles (artist, year) |
| `muted` | `#B9B2D6` | section labels, details |
| `dim` | `#9C94BF` | metadata (model names), upcoming steps |
| `source` | `#B4A9FF` | source labels (Audio, Web…) |
| `accent` | `#FF6FB5` | listen button, playback, highlight |
| `accent-soft` | `#FFC2E0` | label on `surface-strong` background |
| `yellow` | `#FFE066` | secondary primary actions (Find it, That's the one), completed steps, badges |
| `on-accent` | `#15122B` | text and icons on `accent` and `yellow` |

The accent must remain configurable (`--accent` as a CSS variable): planned alternatives `#7CE0C3` and `#8B7CFF`.

### Typography

| Role | Font | Weight | Size |
|---|---|---|---|
| Logo | Bricolage Grotesque | 800 | 28 px, letter-spacing −0.03em |
| H1 Capture | Bricolage Grotesque | 800 | 64 px, line-height 1.02, letter-spacing −0.04em |
| H1 Result | Bricolage Grotesque | 800 | 54 px |
| H1 Searching | Bricolage Grotesque | 800 | 46 px |
| Body | DM Sans | 400 / 500 / 700 | 15–17 px |
| Section labels | DM Mono | 400, uppercase, letter-spacing 0.1em | 13 px |
| Metadata, badges | DM Mono | 400 / 500 | 11–13 px |

### Shapes and spacing

- Radii: buttons and fields as **pills** (`999px`); cards `24px`; main cover art `28px`; list rows `16–20px`; thumbnails `14px`.
- Listen button: 168 px disc inside a 300 px area (dotted `border-strong` circle + inner `yellow` circle + 3 small decorative dots).
- Touch targets ≥ 44 px everywhere.
- Max content width: 1140–1180 px; side margins 24–40 px.

### Brand motif

The **three staggered "la"** (pink, yellow, white, offset upward by 0 / 7 / 14 px) form the logo. The same motif as **three dots** serves as the waiting animation (§ 6).

---

## 3. Screens and copy

The copy below is **final** for the demo; put it in `src/copy.ts` (a single place to edit).

### 3.1 Capture (`/`)

- Header: logo only.
- H1: **You know… that one song?**
- Listen button (microphone icon, no visible text, `aria-label="Listen"`).
- Input row: pill field, placeholder **“whistling, car ad, ~2008”**, `aria-label="Describe the song"`, round yellow arrow button (`aria-label="Search"`); on the right two round icon buttons: video (`Add a video`), audio file (`Add an audio file`).
- Inputs **combine**: you can record and then add a text note before sending. As soon as a recording or a file is present, it shows as a chip above the field ("Recording · 10 s ✕", "clip.mp4 ✕").
- Nothing else on screen: no subtitle, no menu, no footer.

### 3.2 Searching (`/search/:id`)

- Header: logo + **Cancel** text button.
- Left column:
  - three animated dots;
  - H1 **Finding your song…**; subtitle **Usually takes a few seconds.**;
  - **YOUR INPUT** card: yellow badge (e.g. **Background noise detected**, shown only if the backend reports it), waveform of the recording ("analyzed" part in accent), type line (**Microphone recording · 10 s** / **Video clip · 6 s** / **Text description**), user note (**Note: “…”**) if present.
- Right column, **SEARCH PROGRESS**: list of steps fed by the backend (§ 4). Each step: status dot, title, model (mono, `dim`), detail sentence. States: done (yellow ✓ dot), running (`surface` background, accent ♪ dot), upcoming (`dim` title, `chip` · dot), skipped (struck-through title, detail "Skipped").
- Expected titles for an audio input: *Isolating the music* (SAM Audio), *Exact match check* (Chromaprint), *Audio similarity search* (PE-AV · Gemini · CLAP), *Ranking candidates* (Llama), *Verification* (Web). Titles come from the backend; these are used for the mock fixtures.

### 3.3 Result (`/result/:id`)

- Header: logo + **New search** text button.
- Main block:
  - 240 px cover art (accent border);
  - yellow badge **Best match · high confidence** (or **Best match · medium confidence** / **Possible match · low confidence** depending on the score);
  - H1 title; **Artist · Album · Year** line;
  - preview player (accent button, progress bar, **0:10 / 0:30 preview**), **autoplay** on arriving at the screen;
  - **WHY THIS MATCH**: list of justifications, one per line, source label on the left (Audio, Fingerprint, Context, Web, Lyrics…) and full sentence on the right;
  - **Your recording** / **Original recording** comparison (only if the input is audio or video): two cards with a player; only one plays at a time;
  - actions: **That’s the one** (yellow), **Not this one** (outline), **View score details** link.
- Right column **OTHER CANDIDATES**: 3 rows (thumbnail, title, artist · year, play button). Clicking a row promotes it to the main result (swap animation).
- **View score details** opens a side panel: table of scores and ranks per source for each candidate, time per step, brief analysis (rewrite, clues). This is the "technical proof" screen for the jury.

### 3.4 Additional states (not drawn, same visual language)

- **Recording** (on the Capture screen): the listen button becomes a 10 s progress ring, the circles pulse with the sound level, **0:07** counter under the button, **Stop** button.
- **No confident match** (Result screen): H1 **We couldn’t pin it down.**; subtitle **Here are the closest songs we found.**; candidate list; **Add a detail** button that goes back to Capture with the input preserved.
- **Error**: banner at the top, **Something went wrong. Try again.**, **Retry** button; never a stack trace on screen.
- **Not this one**: marks the rejected candidate, promotes the next one; if there are none left, switches to *No confident match*.

---

## 4. API contract expected by the front end

The front end only calls these routes. If the backend (PLAN.md phase 8) differs, **adapt the backend**, not the front end, or flag it.

- `POST /api/search/stream` (multipart: `text?`, `audio?`, `video?`, `options?`) → **SSE** stream:
  - `event: input` → `{ kind: "audio"|"video"|"text"|"mixed", duration_s?, noise_detected?, note?, waveform: number[] }` (waveform reduced to ~64 values 0–1, computed on the backend side);
  - `event: step` → `{ id, title, model, status: "pending"|"running"|"done"|"skipped", detail, elapsed_ms? }` (each update sends the full step);
  - `event: result` → `{ query_id, confidence: "high"|"medium"|"low", best: Track, alternatives: Track[], proofs: {source, text}[], scores: ScoreRow[], timings: {step, ms}[], analysis: {...} }`;
  - `event: error` → `{ message }`.
- `POST /api/search`: same input, final JSON response only (fallback if SSE is unavailable).
- `GET /api/preview/{track_id}`: mp3 preview; `GET /api/cover/{track_id}`: cover art (cached locally on the backend side, for offline mode).
- `POST /api/feedback` → `{ query_id, track_id | null, verdict: "match"|"reject" }`.
- `GET /api/replay/{scene}`: replays a recorded session, with the same SSE stream (§ 7).

TypeScript types in `src/api/types.ts`: `Track { id, title, artist, album?, year?, cover_url, preview_url, source: "catalog"|"web"|"fingerprint" }`, `ScoreRow { track_id, bm25?, emb: Record<string, {score, rank}>, fingerprint?, rerank?, final }`.

---

## 5. Front-end architecture

```
web/
├── index.html
├── tailwind.config.ts
├── src/
│   ├── main.tsx
│   ├── App.tsx              # routes: /, /search/:id, /result/:id
│   ├── copy.ts              # all copy
│   ├── state/
│   │   └── machine.ts       # idle → capturing → searching → result | noMatch | error
│   ├── api/
│   │   ├── types.ts
│   │   ├── client.ts        # fetch + SSE parsing (EventSource does not support POST: use fetch + ReadableStream)
│   │   └── mock.ts          # API simulator built from the fixtures
│   ├── mocks/               # 3 scenarios: memory.json, cover.json, noisy.json + royalty-free cover art and previews
│   ├── audio/
│   │   ├── recorder.ts      # MediaRecorder + AnalyserNode (live level)
│   │   └── player.ts        # only one active player at a time
│   ├── components/
│   │   ├── Logo.tsx
│   │   ├── ListenButton.tsx # idle / recording / disabled
│   │   ├── BriefBar.tsx     # field + button + icons + input chips
│   │   ├── ThreeDots.tsx
│   │   ├── InputCard.tsx
│   │   ├── Waveform.tsx
│   │   ├── StepList.tsx
│   │   ├── TrackHero.tsx    # cover art + title + player
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

- `VITE_API_MODE=mock|live` selects `mock.ts` or `client.ts`; same interface.
- The mock replays the fixtures' SSE events with realistic delays (300–1,200 ms per step).

---

## 6. Animations (framer-motion)

Restrained and fast: nothing should slow the demo down.

- **Listen button**: at rest, slight breathing of the circles (scale 1 → 1.03, 3 s); while recording, the circles follow the microphone level, 10 s progress ring.
- **Three dots**: sequential bounce (pink, yellow, white), 900 ms per cycle.
- **Steps**: each step goes from "upcoming" to "running" with a background fade; the ✓ dot appears with a small bounce.
- **Searching → Result transition**: the cover art enters at scale 0.9 → 1 with a fade (300 ms), then the title, then the justifications in cascade (60 ms stagger).
- **Promoting another candidate**: animated swap (`layoutId`) between the row and the main block.
- Respect `prefers-reduced-motion`: fades only.

---

## 7. Demo mode

- `?replay=memory|cover|noisy`: directly launches the corresponding backup session via `/api/replay/{scene}` (or the fixtures in mock mode). No visible difference from a real search.
- Discreet replay menu: a long press (1 s) on the logo opens a small list of scenes. No global keyboard shortcut.
- `?scale=1.25`: scales up the whole interface for projection.
- All assets (fonts, cover art and previews for the demo cases) are served locally: **the backup demo works without network**.
- Preloading at startup: cover art and previews of the demo scenes.

---

## 8. Phases

### F0 — Setup (≈ 1 h)
Vite + React + TS, Tailwind with the § 2 tokens, `@fontsource`, framer-motion, Vitest, Playwright, ESLint/Prettier. Vite proxy `/api` → `http://localhost:8000`.
**Validation**: empty page with the logo, in the right colors and fonts, offline.

### F1 — Components (≈ ½ day)
All § 5 components in static form, with a `/_kit` page (accessible only in dev) that shows them in all their states.
**Validation**: visual comparison with the mockup; contrasts checked; keyboard navigation possible on every button.

### F2 — Static screens (≈ ½ day)
The three screens assembled with fixture data, + *No confident match* and *Error* states.
**Validation**: rendering faithful to the mockup at 1280 px, readable at 390 px (stacked columns).

### F3 — State machine and mock (≈ ½ day)
`machine.ts`, `mock.ts`, shared SSE parsing. Full flow in mock mode for the three scenarios.
**Validation**: one Playwright test per scenario (Capture → Searching → Result), including *Not this one* and *No confident match*.

### F4 — Real capture (≈ ½ day)
10 s microphone recording (MediaRecorder, webm/opus on Chrome, mp4/aac on Safari: send as is, the backend converts), live level, audio/video file drop (drag-and-drop + picker), combination with the text note, limits (video ≤ 60 s, file ≤ 25 MB) with a clear message.
**Validation**: recording works on Chrome and Safari (macOS); denied permission handled cleanly.

### F5 — API wiring (≈ ½ day)
`client.ts` on `/api/search/stream`, `/api/feedback`, `/api/preview`, `/api/cover`. Fallback to `POST /api/search` if the stream fails. Cancellation (`AbortController`) on **Cancel**.
**Validation**: the three scenes from `DEMO.md` pass in live mode; feedback written on the backend side.

### F6 — Animations and polish (≈ ½ day)
All of § 6; preview autoplay; only one active player; **View score details** panel.
**Validation**: no animation delays the display of the result by more than 400 ms.

### F7 — Demo mode and robustness (≈ ½ day)
All of § 7; build served by FastAPI; full test with the network off.
**Validation**: the three `?replay=` work offline, on the demo machine, in full screen.

### F8 — Tests and rehearsal (≈ ½ day)
Playwright suite green in mock mode; demo checklist in `DEMO.md` (browser, full screen, volume, microphone allowed, warmed-up tab).

---

## 9. Rules

- No hard-coded copy in components: everything goes through `copy.ts`.
- No hard-coded colors: Tailwind tokens or CSS variables.
- No `localStorage` for important data; only the chosen accent and the display scale.
- Real cover art and previews are never committed; fixtures use royalty-free or generated visuals and sounds.
- Accessibility: real `<button>` and `<label>` elements, `aria-label` on icon buttons, visible focus (yellow ring), contrast ≥ 4.5:1.
