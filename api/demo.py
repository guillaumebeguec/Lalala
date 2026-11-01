"""Routes `/api/*` attendues par le front « Lalala » (FRONT_PLAN.md § 4).

Flux d'une recherche (`POST /api/search/stream`, SSE) :
  event: input  → {kind, duration_s?, noise_detected?, note?, waveform[64]}
  event: step   → {id, title, model, status: pending|running|done|skipped, detail, elapsed_ms?}  (étape complète à chaque mise à jour)
  event: result → {query_id, confidence, best, alternatives, proofs, scores, timings, analysis}
  event: error  → {message}
`POST /api/search` renvoie le résultat final seul. `/api/preview/{id}` redirige vers l'extrait
officiel (aucun audio en cache local) ; `/api/cover/{id}` sert la pochette (petite image mise en
cache localement pour le mode hors ligne) ; `/api/feedback` journalise le verdict ;
`/api/replay/{scene}` rejoue une session enregistrée avec le même flux.

Le pipeline réutilise `finder.tools.Toolbox` (recherche hybride, audio_match, reranking). Les
étapes affichées suivent les titres de FRONT_PLAN.md § 3.2 ; SAM Audio est « skipped » tant
que la phase 9 n'est pas faite.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse

from core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

_TOOLBOX = None
RESULTS: dict[str, dict[str, Any]] = {}
MODEL_LABELS = {"clap": "CLAP", "peav": "PE-AV", "gemma": "EmbeddingGemma 2", "gemini": "Gemini", "coverhunter": "CoverHunter", "hum2song": "hum2song"}


def toolbox():
    global _TOOLBOX
    if _TOOLBOX is None:
        import os
        from finder.tools import Toolbox
        _TOOLBOX = Toolbox(subset=os.environ.get("SOUNDTRACE_SUBSET", "dev"))
    return _TOOLBOX


def set_toolbox(tb) -> None:
    global _TOOLBOX
    _TOOLBOX = tb


# --------------------------------------------------------------------------- utilitaires

def waveform(x: np.ndarray, n: int = 64) -> list[float]:
    if len(x) == 0:
        return [0.0] * n
    seg = np.array_split(np.abs(x), n)
    v = np.array([float(s.mean()) if len(s) else 0.0 for s in seg])
    m = v.max() or 1.0
    return [round(float(a / m), 3) for a in v]


def noise_detected(x: np.ndarray, sr: int) -> bool:
    """Heuristique simple : part d'énergie au-dessus de 4 kHz élevée ou dynamique faible."""
    if len(x) < sr:
        return False
    seg = x[: sr * 10].astype(np.float64)
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    hf = spec[freqs > 4000].sum() / (spec.sum() + 1e-9)
    frames = np.array_split(seg, 50)
    rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames if len(f)])
    dyn = (rms.max() - rms.min()) / (rms.max() + 1e-9)
    return bool(hf > 0.35 or dyn < 0.25)


def confidence_label(best: float, second: float | None, exact: bool) -> str:
    if exact:
        return "high"
    margin = best - (second or 0.0)
    if best >= 0.8 and margin >= 0.15:
        return "high"
    if best >= 0.5 or margin >= 0.1:
        return "medium"
    return "low"


@dataclass
class Step:
    id: str
    title: str
    model: str
    status: str = "pending"
    detail: str = ""
    started: float | None = None
    elapsed_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"id": self.id, "title": self.title, "model": self.model, "status": self.status, "detail": self.detail}
        if self.elapsed_ms is not None:
            d["elapsed_ms"] = self.elapsed_ms
        return d


def _track(tb, c) -> dict[str, Any]:
    album = None
    if c.track_id in tb.catalog.index:
        a = tb.catalog.at[c.track_id, "album_title"]
        album = a if isinstance(a, str) else None
    return {"id": c.track_id, "title": c.title, "artist": c.artist, "album": album, "year": c.year,
            "cover_url": f"/api/cover/{c.track_id}" if c.in_catalog else None,
            "preview_url": f"/api/preview/{c.track_id}" if c.in_catalog and c.preview_url else None,
            "source": c.source, "search_url": c.search_url, "in_catalog": c.in_catalog}


# --------------------------------------------------------------------------- pipeline

class Pipeline:
    def __init__(self, tb, text: str | None, audio: bytes | None, kind: str, note: str | None = None):
        self.tb = tb
        self.text = (text or "").strip() or None
        self.audio = audio
        self.kind = kind
        self.note = note
        self.query_id = uuid.uuid4().hex[:12]
        self.steps: list[Step] = []
        self.timings: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def emit(self, ev: str, data: dict[str, Any]) -> dict[str, Any]:
        d = {"event": ev, "data": data}
        self.events.append(d)
        return d

    def step(self, s: Step, status: str, detail: str | None = None):
        s.status = status
        if detail is not None:
            s.detail = detail
        if status == "running":
            s.started = time.time()
        if status in ("done", "skipped") and s.started:
            s.elapsed_ms = int((time.time() - s.started) * 1000)
            self.timings.append({"step": s.title, "ms": s.elapsed_ms})
        return self.emit("step", s.to_dict())

    def _find(self, sid: str) -> Step:
        return next(s for s in self.steps if s.id == sid)

    def run(self) -> Iterator[dict[str, Any]]:
        tb = self.tb
        humming = False
        if self.audio:
            from core.audio_io import load_audio
            try:
                x, sr = load_audio(self.audio, sr=32_000)
            except Exception as e:  # noqa: BLE001
                yield self.emit("error", {"message": f"Audio illisible : {e}"})
                return
            has_hum = "hum2song" in tb.active_embedders() and tb.vec("hum2song", "audio") is not None
            humming = bool(has_hum and self.kind in ("audio", "mixed") and tb.is_humming(self.audio))
            kind = "humming" if humming else self.kind
            noisy = (not humming) and noise_detected(x, sr)
            yield self.emit("input", {"kind": kind, "duration_s": round(len(x) / sr, 1), "noise_detected": noisy,
                                      "note": self.note, "waveform": waveform(x)})
        else:
            yield self.emit("input", {"kind": "text", "note": self.text, "waveform": []})

        general = [n for n in tb.active_embedders() if n not in ("coverhunter", "hum2song")]
        has_cover = "coverhunter" in tb.active_embedders()
        has_hum = "hum2song" in tb.active_embedders() and tb.vec("hum2song", "audio") is not None
        emb_label = " · ".join(MODEL_LABELS.get(n, n) for n in general)
        audio_steps = [Step("isolate", "Isolating the music", "SAM Audio"), Step("exact", "Exact match check", "Chromaprint"),
                       Step("audio_sim", "Audio similarity search", emb_label)] + ([Step("version", "Version matching", "CoverHunter")] if has_cover else []) \
                      + ([Step("melody", "Melody matching", "contour alignment + hum2song")] if has_hum else []) \
                      + [Step("lyrics", "Transcribing the lyrics", "Whisper · lyrics fingerprints")]
        text_steps = [Step("analyze", "Reading your description", "Llama 3.1"), Step("catalog", "Catalogue search", "BM25 · " + emb_label)]
        common = [Step("rank", "Ranking candidates", "Llama 3.1"), Step("verify", "Verification", "Web")]
        self.steps = (audio_steps if self.audio else []) + (text_steps if self.text else []) + common
        for s in self.steps:
            yield self.emit("step", s.to_dict())

        candidates: dict[str, Any] = {}
        lists: dict[str, list] = {}
        exact = None
        analysis: dict[str, Any] = {}

        def absorb(cands, key):
            for c in cands:
                prev = candidates.get(c.track_id)
                if prev is None or c.score > prev.score:
                    candidates[c.track_id] = c
            lists[key] = [(c.track_id, c.score) for c in cands]

        if self.audio:
            from core import isolate
            s_iso, s_exact, s_sim = self._find("isolate"), self._find("exact"), self._find("audio_sim")
            yield self.step(s_exact, "running", "Comparing the fingerprint with the catalogue…")
            exact = tb.fingerprint_match(self.audio)
            # SAM Audio (phase 9) : seulement si l'enregistrement est bruité, sans empreinte exacte, et si un moteur est installé
            if exact or humming or not noisy or isolate.engine() is None:
                why = ("Skipped: exact recording already found" if exact else "Skipped: a single voice" if humming
                       else "Skipped: clean recording" if not noisy else "Skipped: SAM Audio not installed")
                yield self.step(s_iso, "skipped", why)
            else:
                yield self.step(s_iso, "running", "Separating the music from the background noise…")
                iso = isolate.isolate_music(self.audio)
                if iso:
                    self.audio, info = iso[0], iso[1]
                    yield self.step(s_iso, "done", f"Music isolated with SAM Audio ({info.get('engine')}, {info.get('seconds', '?')} s).")
                    exact = tb.fingerprint_match(self.audio)   # seconde chance pour l'empreinte sur l'audio nettoyé
                else:
                    yield self.step(s_iso, "skipped", "SAM Audio failed — continuing with the raw recording.")
            yield self.step(s_exact, "done", "Same recording found." if exact else "No identical recording — not a Shazam case.")
            if exact and not self.text:
                # Empreinte forte = même enregistrement : réponse immédiate, les autres branches sont sautées (PLAN 6.1)
                for sid in ("audio_sim", "version", "melody"):
                    try:
                        yield self.step(self._find(sid), "skipped", "Skipped: exact recording already found")
                    except StopIteration:
                        pass
                candidates[exact.track_id] = exact
            else:
                yield self.step(s_sim, "running", "Comparing the sound of the music itself…")
                res = tb.audio_similarity(self.audio, exact, embedders=general, humming=humming)
                absorb(res["candidates"], "audio")
                yield self.step(s_sim, "done", ("Weak signal: a single voice." if humming else f"{len(res['candidates'])} candidates by audio similarity."))
            if has_cover and not (exact and not self.text):
                s_ver = self._find("version")
                if humming:
                    yield self.step(s_ver, "skipped", "Skipped: a single voice, melody matching instead")
                elif tb.vec("coverhunter", "audio") is not None:
                    yield self.step(s_ver, "running", "Looking for the same song in another version…")
                    res_v = tb.audio_similarity(self.audio, exact, embedders=["coverhunter"])
                    absorb(res_v["candidates"], "version")
                    lists["audio"] = lists.get("audio", []) + lists.get("version", [])
                    yield self.step(s_ver, "done", f"{len(res_v['candidates'])} candidates by melody and harmony.")
                else:
                    yield self.step(s_ver, "skipped", "Skipped")
            if has_hum and not (exact and not self.text):
                s_mel = self._find("melody")
                yield self.step(s_mel, "running", "Extracting the melodic contour…")
                res_m = tb.audio_similarity(self.audio, exact, embedders=["hum2song"], humming=humming)
                absorb(res_m["candidates"], "melody")
                lists["audio"] = lists.get("audio", []) + lists.get("melody", [])
                yield self.step(s_mel, "done", f"Contour aligned against the catalogue melodies ({len(res_m['candidates'])} candidates).")
            s_lyr = self._find("lyrics")
            if humming and not exact:
                yield self.step(s_lyr, "running", "Listening for sung words…")
                res_l = tb.sung_lyrics_search(self.audio)
                tr = res_l["transcript"]
                if tr.n_words >= 3:
                    absorb(res_l["candidates"], "lyrics")
                    lists["audio"] = lists.get("audio", []) + lists.get("lyrics", [])
                    analysis["sung_lyrics"] = {"text": tr.text, "language": tr.language, "branches": res_l["lists"]}
                    yield self.step(s_lyr, "done", f"Heard: “{tr.text[:70]}” → {len(res_l['candidates'])} candidates.")
                else:
                    yield self.step(s_lyr, "done", "No words recognised (humming only).")
            else:
                yield self.step(s_lyr, "skipped", "Skipped")
        text_hints: dict = {}
        if self.text:
            s_an, s_cat = self._find("analyze"), self._find("catalog")
            yield self.step(s_an, "running", "Extracting clues (era, language, instruments)…")
            rw = tb.rewrite_query(self.text)
            if rw:
                analysis = {"query": rw.query, "query_fr": rw.query_fr, "captions": rw.captions, "hints": {k: v for k, v in rw.hints.items() if v}}
                clues = [f"{k}: {v}" for k, v in (analysis.get("hints") or {}).items() if k in ("decade", "language", "medium", "genre", "vocals")]
                yield self.step(s_an, "done", ("Clues: " + ", ".join(clues)) if clues else "No strong clue found.")
            else:
                yield self.step(s_an, "done", "Used the description as is.")
            yield self.step(s_cat, "running", "Searching cards and audio…")
            ret = tb.retrieve(self.text, rw)
            text_hints = ret["hints"]
            absorb(tb.to_candidates(ret["fused"]), "text")
            yield self.step(s_cat, "done", f"{len(ret['fused'])} candidates ({', '.join(ret.get('lists', {}).keys())}).")

        s_rank = self._find("rank")
        ranked = sorted(candidates.values(), key=lambda c: -c.score)
        if exact:
            ranked = [exact] + [c for c in ranked if c.track_id != exact.track_id]
        yield self.step(s_rank, "running", "Weighing each candidate against the clues…" if tb.use_rerank else "Fusing the result lists…")
        if self.text and self.audio and ranked:
            from core.fusion import rrf
            order = {i: k for k, (i, _) in enumerate(rrf([lists.get("text", []), lists.get("audio", [])], k=60))}
            ranked.sort(key=lambda c: order.get(c.track_id, 1e9))
        if self.text and ranked and not exact:
            fused, reasons = tb.rerank_candidates(self.text, [(c.track_id, c.score) for c in ranked], text_hints)
            by_id = {c.track_id: c for c in ranked}
            ranked = []
            for i, sc in fused:
                c = by_id[i]; c.score = sc
                if reasons.get(i):
                    c.reason = reasons[i]
                ranked.append(c)
        yield self.step(s_rank, "done", f"Top candidate: {ranked[0].title} — {ranked[0].artist}" if ranked else "No candidate.")
        yield self.step(self._find("verify"), "skipped", "Skipped")

        if not ranked:
            yield self.emit("result", {"query_id": self.query_id, "confidence": "low", "best": None, "alternatives": [], "proofs": [],
                                       "scores": [], "timings": self.timings, "analysis": analysis})
            return
        best, others = ranked[0], ranked[1:4]
        proofs = []
        if exact:
            proofs.append({"source": "Fingerprint", "text": "The audio fingerprint matches this exact recording."})
        if best.reason:
            proofs.append({"source": "Context", "text": best.reason})
        if self.audio and not exact and best.source == "version":
            proofs.append({"source": "Version", "text": "Same song, different version (cover or live)."})
        if self.audio and not exact and best.source == "melody":
            proofs.append({"source": "Melody", "text": "Your hummed melody lines up with this song's melody."})
        if self.audio and not exact and best.source == "lyrics":
            proofs.append({"source": "Lyrics", "text": best.reason})
        if self.audio and not exact:
            proofs.append({"source": "Audio", "text": "The music itself sounds closest to this track across the audio encoders."})
        h = analysis.get("hints") or {}
        bits = [str(h[k]) for k in ("decade", "language", "genre") if h.get(k)]
        if bits:
            proofs.append({"source": "Clues", "text": "Matches the clues from your description: " + ", ".join(bits) + "."})
        scores = [{"track_id": c.track_id, "emb": {}, "final": round(float(c.score), 4), "source": c.source} for c in ranked[:10]]
        conf = confidence_label(float(best.score), float(others[0].score) if others else None, bool(exact))
        result = {"query_id": self.query_id, "confidence": conf, "best": _track(tb, best), "alternatives": [_track(tb, c) for c in others],
                  "proofs": proofs, "scores": scores, "timings": self.timings, "analysis": analysis}
        RESULTS[self.query_id] = {"result": result, "events": self.events, "t": time.time()}
        yield self.emit("result", result)


def _sse(gen: Iterator[dict[str, Any]]) -> StreamingResponse:
    def stream():
        for d in gen:
            yield f"event: {d['event']}\ndata: {json.dumps(d['data'], ensure_ascii=False)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _read_inputs(text: str | None, audio: UploadFile | None, video: UploadFile | None) -> tuple[str | None, bytes | None, str]:
    data, kind = None, "text"
    if video is not None:
        raw = await video.read()
        tmp = get_settings().data_dir / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        p = tmp / f"up_{uuid.uuid4().hex[:8]}{Path(video.filename or 'v.mp4').suffix or '.mp4'}"
        p.write_bytes(raw)
        from core.audio_io import load_audio, to_wav_bytes
        try:
            x, sr = load_audio(p, sr=32_000)
            data, kind = to_wav_bytes(x, sr), "video"
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"Vidéo illisible : {e}")
        finally:
            p.unlink(missing_ok=True)
    elif audio is not None:
        data, kind = await audio.read(), "audio"
    if data is not None and (text or "").strip():
        kind = "mixed"
    if data is None and not (text or "").strip():
        raise HTTPException(400, "Donnez un texte, un audio ou une vidéo.")
    return text, data, kind


@router.post("/search/stream")
async def search_stream(text: str | None = Form(None), audio: UploadFile | None = File(None), video: UploadFile | None = File(None),
                        options: str | None = Form(None)):
    t, data, kind = await _read_inputs(text, audio, video)
    return _sse(Pipeline(toolbox(), t, data, kind, note=t if data is not None else None).run())


@router.post("/search")
async def search(text: str | None = Form(None), audio: UploadFile | None = File(None), video: UploadFile | None = File(None),
                 options: str | None = Form(None)):
    t, data, kind = await _read_inputs(text, audio, video)
    result = None
    for d in Pipeline(toolbox(), t, data, kind, note=t if data is not None else None).run():
        if d["event"] == "result":
            result = d["data"]
        elif d["event"] == "error":
            raise HTTPException(500, d["data"]["message"])
    return result


@router.get("/preview/{track_id}")
def preview(track_id: str):
    tb = toolbox()
    if len(tb.previews) and track_id in tb.previews.index:
        from catalog.previews import fresh_preview_url
        url = tb.previews.at[track_id, "preview_url"]
        url = fresh_preview_url(track_id, url if isinstance(url, str) else None, deezer_id=tb.previews.at[track_id, "deezer_id"])
        if isinstance(url, str) and url:
            return RedirectResponse(url)
    raise HTTPException(404, "pas d'extrait")


def covers_dir() -> Path:
    d = get_settings().data_dir / "covers"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.get("/cover/{track_id}")
def cover(track_id: str):
    p = covers_dir() / f"{track_id}.jpg"
    if p.exists():
        return FileResponse(p, media_type="image/jpeg")
    tb = toolbox()
    if track_id not in tb.catalog.index:
        raise HTTPException(404)
    dz = tb.catalog.at[track_id, "deezer_id"]
    if dz is None or (isinstance(dz, float) and np.isnan(dz)):
        raise HTTPException(404)
    try:
        import httpx
        from catalog.deezer import DeezerClient
        t = DeezerClient().track(int(dz))
        url = ((t or {}).get("album") or {}).get("cover_medium")
        if not url:
            raise HTTPException(404)
        r = httpx.get(url, timeout=15, follow_redirects=True)
        r.raise_for_status()
        p.write_bytes(r.content)
        return FileResponse(p, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"pochette indisponible : {e}")


@router.post("/feedback")
def feedback(payload: dict):
    log_path = get_settings().data_dir / "feedback.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"t": time.time(), **{k: payload.get(k) for k in ("query_id", "track_id", "verdict")}}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}


def replays_dir() -> Path:
    d = get_settings().data_dir / "replays"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("/save/{query_id}/{scene}")
def save_scene(query_id: str, scene: str):
    if query_id not in RESULTS:
        raise HTTPException(404, "query inconnue")
    safe = "".join(c for c in scene if c.isalnum() or c in "-_")[:40] or "scene"
    (replays_dir() / f"api_{safe}.json").write_text(json.dumps(RESULTS[query_id]["events"], ensure_ascii=False, indent=1), encoding="utf-8")
    return {"scene": safe}


@router.get("/replay/{scene}")
def replay(scene: str, delay: float = 0.7):
    p = replays_dir() / f"api_{scene}.json"
    if not p.exists():
        raise HTTPException(404, "scène inconnue")
    events = json.loads(p.read_text(encoding="utf-8"))

    def gen():
        for d in events:
            yield d
            if d["event"] == "step" and d["data"].get("status") == "done":
                time.sleep(delay)
    return _sse(gen())


@router.get("/replays")
def list_scenes():
    return sorted(p.stem[4:] for p in replays_dir().glob("api_*.json"))
