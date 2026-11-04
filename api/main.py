"""Lalala API (PLAN.md phase 8).

- POST /session                      → {session_id}
- POST /session/{id}/message         (text or answer to a question) → SSE of events
- POST /session/{id}/audio           (microphone recording, multipart) → SSE
- POST /session/{id}/video           (video file) → SSE (audio extracted, then video embedding if available)
- GET  /session/{id}/state           → state (clues, candidates, belief)
- GET  /preview/{track_id}           → redirect to the official preview URL (nothing cached locally)
- POST /session/{id}/save/{name}     → saves the session (events) for replay
- POST /replay/{name}                → replays a recorded session over SSE (demo plan B)
- GET  /                             → web/index.html

Run: `uv run uvicorn api.main:app --reload`. Variables: LALALA_SUBSET=dev|full,
LALALA_POLICY=baseline|ig.
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

app = FastAPI(title="Lalala", version="0.1")

from api.demo import router as demo_router  # noqa: E402 — /api/* routes of the "Lalala" front (FRONT_PLAN.md)
app.include_router(demo_router)


@app.on_event("startup")
def _warmup() -> None:
    """Warms up the catalogue, indexes and embedders in a thread (LALALA_WARMUP=0 to disable),
    so that the demo's first request does not pay 30 s of loading."""
    if os.environ.get("LALALA_WARMUP", "1") == "0":
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
            logger.info("warm-up complete")
        except Exception as e:  # noqa: BLE001
            logger.warning("warm-up: %s", e)

    threading.Thread(target=run, name="warmup", daemon=True).start()

SESSIONS: dict[str, dict[str, Any]] = {}
_TOOLBOX = None


def toolbox():
    global _TOOLBOX
    if _TOOLBOX is None:
        from finder.tools import Toolbox
        _TOOLBOX = Toolbox(subset=os.environ.get("LALALA_SUBSET", "dev"))
    return _TOOLBOX


def new_agent():
    from finder.agent import Agent
    from finder.belief import ig_policy
    policy = ig_policy() if os.environ.get("LALALA_POLICY", "ig") == "ig" else None
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
    """Adds a `belief` event after each question / proposal, for the entropy curve."""
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
        raise HTTPException(404, "unknown session")
    return SESSIONS[sid]


@app.post("/session/{sid}/message")
async def message(sid: str, text: str = Form(...)):
    s = _get(sid)
    agent = s["agent"]
    if agent.state.done and agent.state.pending_question is None:
        agent.state.done = False  # "no" or a new clue: restart the investigation
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
        raise HTTPException(400, f"unreadable video: {e}")
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
    raise HTTPException(404, "no preview for this track")


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
        raise HTTPException(404, "unknown replay")
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
    """Vite build of the front (web/dist) in demo; otherwise the old vanilla front (web_legacy)."""
    if (_DIST / "index.html").exists():
        return FileResponse(_DIST / "index.html")
    return FileResponse(PROJECT_ROOT / "web_legacy" / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "subset": os.environ.get("LALALA_SUBSET", "dev")}
