"""Lalala agent loop (PLAN.md phase 6.2) — baseline: Llama freely chooses
search / ask / propose, through OpenAI tool calls.

Explicit state (`SessionState`): input, clues, candidates, queries made,
questions asked. Guard rails: `max_questions` (3), `max_tool_calls` (8), no
repeated query, no title without a source. Events emitted for the front:
`clue`, `search`, `candidates`, `question`, `proposal`, `message`, `error`.

The agent is **synchronous per turn**: `step(user_message)` consumes one message
from the user (description, answer to a question, or "no") and advances until
the next question or proposal. Phase 7 replaces the choice of questions with
information gain (`finder/belief.py`) via `question_policy`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from core import llm
from finder.prompts import AGENT_SYSTEM
from finder.tools import Candidate, Toolbox

logger = logging.getLogger(__name__)

TOOLS = [
    {"type": "function", "function": {"name": "catalog_search", "description": "Hybrid search in the song catalogue (text cards + audio).",
                                      "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Web search for usages (ads, series, films), quoted lyrics, facts.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_user", "description": "Ask ONE closed question with 2-4 options.",
                                      "parameters": {"type": "object", "properties": {"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4}},
                                                     "required": ["question", "options"]}}},
    {"type": "function", "function": {"name": "propose", "description": "Propose 1 to 3 candidates (track_id from search results) with confidence 0-1 and a reason.",
                                      "parameters": {"type": "object", "properties": {"candidates": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                                          "type": "object", "properties": {"track_id": {"type": "string"}, "confidence": {"type": "number"}, "reason": {"type": "string"}},
                                          "required": ["track_id", "confidence", "reason"]}}}, "required": ["candidates"]}}},
]

DONT_KNOW = "je ne sais pas"


@dataclass
class Event:
    type: str
    data: dict[str, Any]
    t: float = field(default_factory=time.time)


@dataclass
class SessionState:
    language: str = "français"
    clues: list[str] = field(default_factory=list)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    rejected: set[str] = field(default_factory=set)
    queries: list[str] = field(default_factory=list)
    questions: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    proposals: list[list[dict]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)   # LLM history
    pending_question: dict[str, Any] | None = None
    done: bool = False


class Agent:
    def __init__(self, tools: Toolbox, max_questions: int = 3, max_tool_calls: int = 8, model: str | None = None,
                 question_policy: Callable[["Agent", SessionState], dict | None] | None = None, language: str = "français"):
        self.tools = tools
        self.max_questions = max_questions
        self.max_tool_calls = max_tool_calls
        self.model = model
        self.question_policy = question_policy   # phase 7: information gain
        self.state = SessionState(language=language)
        try:
            n_catalog = len(self.tools.catalog)
        except Exception:  # noqa: BLE001 — fake tools or missing catalogue
            n_catalog = 0
        self.state.messages.append({"role": "system", "content": AGENT_SYSTEM.format(
            n_catalog=n_catalog, max_questions=max_questions, language=language)})

    # ---- user input ----
    def step(self, user_message: str | None = None, audio: bytes | None = None) -> Iterator[Event]:
        st = self.state
        if st.pending_question is not None and user_message is not None:
            q = st.pending_question
            st.pending_question = None
            st.questions[-1]["answer"] = user_message
            if user_message.strip().lower() != DONT_KNOW:
                st.clues.append(f"{q['question']} → {user_message}")
                yield Event("clue", {"text": st.clues[-1]})
            st.messages.append({"role": "tool", "tool_call_id": q["call_id"], "name": "ask_user", "content": json.dumps({"answer": user_message}, ensure_ascii=False)})
        elif user_message is not None:
            low = user_message.strip().lower()
            if st.proposals and low in {"non", "no", "ce n'est pas ça", "pas celui-là"}:
                for c in st.proposals[-1]:
                    st.rejected.add(c["track_id"])
                    st.candidates.pop(c["track_id"], None)
                st.messages.append({"role": "user", "content": "No, none of these. Keep investigating with the remaining clues."})
                yield Event("message", {"text": "All right, I am dropping those candidates and continuing the investigation."})
            else:
                st.clues.append(user_message.strip())
                st.messages.append({"role": "user", "content": user_message.strip()})
                yield Event("clue", {"text": user_message.strip()})
        if audio is not None:
            yield from self._audio(audio)
        yield from self._run()

    def _audio(self, audio: bytes) -> Iterator[Event]:
        st = self.state
        st.tool_calls += 1
        res = self.tools.audio_match(audio)
        self._absorb(res["candidates"])
        yield Event("search", {"tool": "audio_match", "n": len(res["candidates"]), "exact": bool(res["exact"])})
        yield Event("candidates", {"candidates": [c.to_dict() for c in self._top(10)]})
        if res["exact"]:
            yield from self._propose([{"track_id": res["exact"].track_id, "confidence": 0.98, "reason": "Identical audio fingerprint: this is that very recording."}])
            return
        summary = [{"track_id": c.track_id, "title": c.title, "artist": c.artist, "year": c.year, "score": round(c.score, 3)} for c in res["candidates"][:10]]
        st.messages.append({"role": "user", "content": "The user sent an audio recording. audio_match results (best first): " + json.dumps(summary, ensure_ascii=False)})

    # ---- loop ----
    def _run(self) -> Iterator[Event]:
        st = self.state
        while not st.done and st.pending_question is None:
            if st.tool_calls >= self.max_tool_calls:
                yield from self._propose(self._auto_proposal())
                return
            # phase 7: the external question policy (information gain) takes precedence
            if self.question_policy and st.candidates and len(st.questions) < self.max_questions:
                q = self.question_policy(self, st)
                if q:
                    yield from self._ask(q["question"], q["options"], call_id=f"policy_{len(st.questions)}", extra=q)
                    return
            try:
                r = llm.chat(st.messages, tools=TOOLS, temperature=0.0, model=self.model)
            except llm.LLMError as e:
                yield Event("error", {"text": str(e)})
                yield from self._propose(self._auto_proposal())
                return
            st.messages.append(r.message)
            if not r.tool_calls:
                # No tool call: force a proposal if we have candidates, otherwise a search.
                if st.candidates:
                    yield from self._propose(self._auto_proposal())
                    return
                st.messages.append({"role": "user", "content": "Use a tool: call catalog_search with the description."})
                st.tool_calls += 1
                continue
            tc = r.tool_calls[0]
            for extra in r.tool_calls[1:]:   # one tool per turn; the others get an empty reply
                st.messages.append({"role": "tool", "tool_call_id": extra.id, "name": extra.name, "content": "ignored: one tool per turn"})
            st.tool_calls += 1
            yield from self._dispatch(tc)

    def _dispatch(self, tc) -> Iterator[Event]:
        st = self.state
        name, args = tc.name, tc.arguments
        if name == "catalog_search":
            text = str(args.get("text") or " ".join(st.clues))
            if text in st.queries:
                st.messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": "Already searched this exact text. Use different words, ask the user, or propose."})
                return
            st.queries.append(text)
            res = self.tools.catalog_search(text)
            self._absorb(res["candidates"])
            rw = res.get("rewrite")
            if rw and (rw.decade or rw.language):
                for k in ("decade", "language"):
                    v = getattr(rw, k)
                    if v:
                        st.clues.append(f"{k}: {v}")
                        yield Event("clue", {"text": st.clues[-1], "inferred": True})
            yield Event("search", {"tool": name, "query": text, "n": len(res["candidates"]), "lists": res.get("lists", {})})
            yield Event("candidates", {"candidates": [c.to_dict() for c in self._top(10)]})
            payload = [{"track_id": c.track_id, "title": c.title, "artist": c.artist, "year": c.year, "score": round(c.score, 3), "why": c.reason} for c in res["candidates"][:10]]
            st.messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": json.dumps(payload, ensure_ascii=False) or "[]"})
        elif name == "web_search":
            q = str(args.get("query") or " ".join(st.clues))
            if "web:" + q in st.queries:
                st.messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": "Already searched."})
                return
            st.queries.append("web:" + q)
            res = self.tools.web_search(q)
            yield Event("search", {"tool": name, "query": q, "n": len(res)})
            content = json.dumps([{"title": r["title"], "snippet": r["snippet"][:300]} for r in res], ensure_ascii=False) if res else \
                "Web search is unavailable. Rely on the catalogue or ask the user."
            st.messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": content + (
                "\nIf you identified a song title + artist in these results, call catalog_search with 'title artist' to retrieve it." if res else "")})
        elif name == "ask_user":
            if len(st.questions) >= self.max_questions:
                st.messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": "Question budget exhausted: propose now."})
                return
            opts = [str(o) for o in (args.get("options") or [])][:4]
            if len(opts) < 2:
                opts = opts + ["autre"]
            yield from self._ask(str(args.get("question") or "?"), opts, call_id=tc.id)
        elif name == "propose":
            cands = [c for c in (args.get("candidates") or []) if isinstance(c, dict)]
            valid = [c for c in cands if str(c.get("track_id")) in st.candidates]
            if not valid:
                st.messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": "Rejected: track_id must come from search results. Search first."})
                return
            st.messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": "ok"})
            yield from self._propose(valid)
        else:
            st.messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": f"unknown tool {name}"})

    # ---- actions ----
    def _ask(self, question: str, options: list[str], call_id: str, extra: dict | None = None) -> Iterator[Event]:
        st = self.state
        if DONT_KNOW not in [o.lower() for o in options]:
            options = options + [DONT_KNOW]
        st.questions.append({"question": question, "options": options, "call_id": call_id, **({k: v for k, v in (extra or {}).items() if k not in ("question", "options")})})
        st.pending_question = st.questions[-1]
        yield Event("question", {"question": question, "options": options, **{k: v for k, v in (extra or {}).items() if k in ("expected_gain", "entropy")}})

    def _propose(self, cands: list[dict]) -> Iterator[Event]:
        st = self.state
        out = []
        for c in cands[:3]:
            cand = st.candidates.get(str(c.get("track_id")))
            if cand is None:
                continue
            d = cand.to_dict()
            d["confiance"] = float(max(0.0, min(1.0, float(c.get("confidence", cand.score)))))
            d["raison"] = str(c.get("reason") or cand.reason)
            out.append(d)
        if not out:
            yield Event("message", {"text": "I could not find a convincing candidate. Can you give me another detail?"})
            st.done = True
            return
        st.proposals.append(out)
        st.done = True
        yield Event("proposal", {"candidates": out})

    def _auto_proposal(self) -> list[dict]:
        return [{"track_id": c.track_id, "confidence": c.score, "reason": c.reason or "best search score"} for c in self._top(3)]

    def _absorb(self, cands: list[Candidate]) -> None:
        st = self.state
        for c in cands:
            if c.track_id in st.rejected:
                continue
            prev = st.candidates.get(c.track_id)
            if prev is None or c.score > prev.score:
                st.candidates[c.track_id] = c

    def _top(self, n: int) -> list[Candidate]:
        return sorted(self.state.candidates.values(), key=lambda c: -c.score)[:n]
