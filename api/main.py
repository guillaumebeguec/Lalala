"""API SoundTrace (PLAN.md phase 8).

- POST /session                      → {session_id}
- POST /session/{id}/message         (texte ou réponse à une question) → SSE d'événements
- POST /session/{id}/audio           (enregistrement micro, multipart) → SSE
- POST /session/{id}/video           (fichier vidéo) → SSE (audio extrait, puis embedding vidéo si dispo)
- GET  /session/{id}/state           → état (indices, candidats, croyance)
- GET  /preview/{track_id}           → redirection vers l'URL d'extrait officielle (rien en cache local)
- POST /session/{id}/save/{name}     → enregistre la session (événements) pour le replay
- POST /replay/{name}                → rejoue une session enregistrée en SSE (plan B de démo)
- GET  /                             → web/index.html

Lancement : `uv run uvicorn api.main:app --reload`. Variables : SOUNDTRACE_SUBSET=dev|full,
SOUNDTRACE_POLICY=baseline|ig.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse

from core.config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

app = FastAPI(title="SoundTrace", version="0.1")

from api.demo import router as demo_router  # noqa: E402 — routes /api/* du front « Lalala » (FRONT_PLAN.md)
app.include_router(demo_router)


@app.on_event("startup")
def _warmup() -> None:
    """Préchauffe catalogue, index et embedders dans un thread (SOUNDTRACE_WARMUP=0 pour désactiver),
    pour que la première requête de la démo ne paie pas 30 s de chargement."""
    if os.environ.get("SOUNDTRACE_WARMUP", "1") == "0":
        return
    import threading

    def run():
        try:
            from api.demo import toolbox as demo_toolbox
            tb = demo_toolbox()
            _ = tb.catalog, tb.cards, tb.previews, tb.bm25(), tb.fingerprints(), tb.audfprint()
            for name in tb.active_embedders():
                if tb.vec(name, "audio") is not None or tb.vec(name, "text") is not None:
                    emb = tb.emb(name)
                    load = getattr(emb, "_load", None)
                    if load:
                        load()
            logger.info("préchauffage terminé")
        except Exception as e:  # noqa: BLE001
            logger.warning("préchauffage : %s", e)

    threading.Thread(target=run, name="warmup", daemon=True).start()

SESSIONS: dict[str, dict[str, Any]] = {}
_TOOLBOX = None


def toolbox():
    global _TOOLBOX
    if _TOOLBOX is None:
        from finder.tools import Toolbox
        _TOOLBOX = Toolbox(subset=os.environ.get("SOUNDTRACE_SUBSET", "dev"))
    return _TOOLBOX


def new_agent():
    from finder.agent import Agent
    from finder.belief import ig_policy
    policy = ig_policy() if os.environ.get("SOUNDTRACE_POLICY", "ig") == "ig" else None
    return Agent(toolbox(), question_policy=policy)


def replays_dir() -> Path:
    d = get_settings().data_dir / "replays"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sse(events: Iterator[Any], session: dict[str, Any] | None = None) -> StreamingResponse:
    def gen():
        for ev in events:
            d = {"type": ev.type, "data": ev.data, "t": ev.t}
            if session is not None:
                session["events"].append(d)
                yield f"event: {ev.type}\ndata: {json.dumps(d['data'], ensure_ascii=False)}\n\n"
            else:
                yield f"event: {d['type']}\ndata: {json.dumps(d['data'], ensure_ascii=False)}\n\n"
        yield "event: end\ndata: {}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _belief_snapshot(agent) -> dict[str, Any] | None:
    b = getattr(agent.state, "belief", None)
    if b is None:
        return None
    from finder.belief import entropy
    return {"entropy": round(entropy(b.probs), 4), "distribution": b.distribution(10), "history": b.history}


def _with_belief(agent, events: Iterator[Any]) -> Iterator[Any]:
    """Ajoute un événement `belief` après chaque question / proposition pour la courbe d'entropie."""
    from finder.agent import Event
    for ev in events:
        yield ev
        if ev.type in ("question", "proposal", "candidates"):
            snap = _belief_snapshot(agent)
            if snap:
                yield Event("belief", snap)


@app.post("/session")
def create_session():
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"agent": new_agent(), "events": [], "created": time.time()}
    return {"session_id": sid}


def _get(sid: str) -> dict[str, Any]:
    if sid not in SESSIONS:
        raise HTTPException(404, "session inconnue")
    return SESSIONS[sid]


@app.post("/session/{sid}/message")
async def message(sid: str, text: str = Form(...)):
    s = _get(sid)
    agent = s["agent"]
    if agent.state.done and agent.state.pending_question is None:
        agent.state.done = False  # « non » ou nouvel indice : on relance l'enquête
    return _sse(_with_belief(agent, agent.step(text)), s)


@app.post("/session/{sid}/audio")
async def audio(sid: str, file: UploadFile = File(...)):
    s = _get(sid)
    data = await file.read()
    agent = s["agent"]
    agent.state.done = False
    return _sse(_with_belief(agent, agent.step(None, audio=data)), s)


@app.post("/session/{sid}/video")
async def video(sid: str, file: UploadFile = File(...)):
    s = _get(sid)
    data = await file.read()
    tmp = get_settings().data_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / f"{sid}_{int(time.time())}{Path(file.filename or 'v.mp4').suffix or '.mp4'}"
    p.write_bytes(data)
    from core.audio_io import load_audio, to_wav_bytes
    try:
        x, sr = load_audio(p, sr=48_000)
        wav = to_wav_bytes(x, sr)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"vidéo illisible : {e}")
    agent = s["agent"]
    agent.state.done = False
    return _sse(_with_belief(agent, agent.step(None, audio=wav)), s)


@app.get("/session/{sid}/state")
def state(sid: str):
    s = _get(sid)
    st = s["agent"].state
    return {"clues": st.clues, "queries": st.queries, "questions": [{k: v for k, v in q.items() if k != "call_id"} for q in st.questions],
            "candidates": [c.to_dict() for c in sorted(st.candidates.values(), key=lambda c: -c.score)[:10]],
            "proposals": st.proposals, "tool_calls": st.tool_calls, "done": st.done, "belief": _belief_snapshot(s["agent"])}


@app.get("/preview/{track_id}")
def preview(track_id: str):
    tb = toolbox()
    if len(tb.previews) and track_id in tb.previews.index:
        from catalog.previews import fresh_preview_url
        url = tb.previews.at[track_id, "preview_url"]
        url = fresh_preview_url(track_id, url if isinstance(url, str) else None, deezer_id=tb.previews.at[track_id, "deezer_id"])
        if isinstance(url, str) and url:
            return RedirectResponse(url)
    raise HTTPException(404, "pas d'extrait pour ce morceau")


@app.post("/session/{sid}/save/{name}")
def save(sid: str, name: str):
    s = _get(sid)
    safe = "".join(c for c in name if c.isalnum() or c in "-_")[:40] or "session"
    p = replays_dir() / f"{safe}.json"
    p.write_text(json.dumps({"saved": time.time(), "events": s["events"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"name": safe, "events": len(s["events"])}


@app.post("/replay/{name}")
def replay(name: str, delay: float = 0.6):
    p = replays_dir() / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, "replay inconnu")
    events = json.loads(p.read_text(encoding="utf-8"))["events"]

    def gen():
        for d in events:
            yield f"event: {d['type']}\ndata: {json.dumps(d['data'], ensure_ascii=False)}\n\n"
            time.sleep(delay)
        yield "event: end\ndata: {}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/replays")
def list_replays():
    return sorted(p.stem for p in replays_dir().glob("*.json"))


_DIST = PROJECT_ROOT / "web" / "dist"
if _DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")


@app.get("/")
def index():
    """Build Vite du front (web/dist) en démo ; sinon l'ancien front vanilla (web_legacy)."""
    if (_DIST / "index.html").exists():
        return FileResponse(_DIST / "index.html")
    return FileResponse(PROJECT_ROOT / "web_legacy" / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "subset": os.environ.get("SOUNDTRACE_SUBSET", "dev")}
