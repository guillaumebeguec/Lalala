"""Simulated user to evaluate the agent without a human (DEMO.md § 3.3, PLAN.md phase 2.6).

A Llama distinct from the agent plays the person looking for the track. It knows
**only the original text of the request** (title + post body for ToT_Music, the
description for T2). It answers the agent's closed questions from that text
alone, and "I don't know" otherwise.

"Informed" variant (`informed=True`): it additionally knows the year and language
of the true track (upper bound). Nothing else: never the title nor the artist.

Anti-leak guarantee: `SimulatedUser.prompt_messages()` is the only way to build
the prompt, and `tests/test_simulated_user.py` checks that the expected answer
never appears in it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from core import llm

logger = logging.getLogger(__name__)

DONT_KNOW = "je ne sais pas"

_SYSTEM = (
    "Tu joues une personne qui cherche le titre d'une chanson qu'elle a en tête. "
    "Tu ne disposes QUE de ta description d'origine ci-dessous : c'est tout ce dont tu te souviens. "
    "Un assistant te pose des questions fermées avec des options. Réponds en choisissant UNE option "
    "uniquement si ta description le justifie clairement ou le laisse fortement supposer. "
    f"Sinon réponds exactement « {DONT_KNOW} ». N'invente rien, ne déduis pas de connaissances extérieures. "
    "Réponds en JSON : {\"evidence\": \"<citation EXACTE du passage de ta description qui justifie ta réponse, "
    "ou chaîne vide>\", \"choice\": \"<option exacte ou je ne sais pas>\"}. "
    f"Si tu ne peux rien citer, choice doit être « {DONT_KNOW} ».\n\n"
    "Exemples. Description : « un morceau disco des années 70 avec une voix de femme ».\n"
    "Question : Dans quelle décennie ? Options : 1970s | 1980s | je ne sais pas → "
    "{\"evidence\": \"des années 70\", \"choice\": \"1970s\"}\n"
    "Question : En quelle langue ? Options : français | anglais | je ne sais pas → "
    "{\"evidence\": \"\", \"choice\": \"je ne sais pas\"} (la description ne dit rien sur la langue)."
)

_ANSWER_SCHEMA = {
    "title": "simulated_answer",
    "type": "object",
    "required": ["evidence", "choice"],
    "properties": {"evidence": {"type": "string"}, "choice": {"type": "string"}},
}


@dataclass
class SimulatedUser:
    description: str                      # original text (sole knowledge)
    informed: bool = False
    known_year: int | None = None         # only if informed
    known_language: str | None = None     # only if informed
    model: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.informed:
            # An uninformed user carries no external information, not even by mistake.
            self.known_year = None
            self.known_language = None

    # ---- prompt construction: single entry point ----
    def prompt_messages(self, question: str, options: list[str]) -> list[dict[str, str]]:
        known = ""
        if self.informed:
            extra = []
            if self.known_year:
                extra.append(f"le morceau est sorti vers {self.known_year}")
            if self.known_language:
                extra.append(f"les paroles sont en {self.known_language}")
            if extra:
                known = "\nTu es aussi sûr·e de ceci : " + " ; ".join(extra) + "."
        opts = options + ([DONT_KNOW] if DONT_KNOW not in [o.lower() for o in options] else [])
        user = (
            f"Ta description d'origine :\n\"\"\"\n{self.description}\n\"\"\"{known}\n\n"
            + (("Échanges précédents :\n" + "\n".join(f"- Q : {h['q']} → R : {h['a']}" for h in self.history) + "\n\n")
               if self.history else "")
            + f"Question de l'assistant : {question}\nOptions : " + " | ".join(opts)
        )
        return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]

    def _grounded(self, evidence: str) -> bool:
        from catalog.normalize import fold
        ev = fold(evidence)
        if len(ev) < 3:
            return False
        known = self.description
        if self.informed:
            known += f" {self.known_year or ''} {self.known_language or ''}"
        return ev in fold(known)

    def opening_message(self) -> str:
        """The first message sent to the agent: the description as is."""
        return self.description

    def answer(self, question: str, options: list[str]) -> str:
        msgs = self.prompt_messages(question, options)
        try:
            r = llm.chat(msgs, json_schema=_ANSWER_SCHEMA, temperature=0.0, max_tokens=160, model=self.model)
            choice = str((r.json or {}).get("choice", DONT_KNOW)).strip()
            evidence = str((r.json or {}).get("evidence") or "").strip()
        except llm.LLMError as e:
            logger.warning("simulated user: %s; answering \"%s\"", e, DONT_KNOW)
            choice, evidence = DONT_KNOW, ""
        choice = _snap(choice, options)
        # Guard rail: an answer must be backed by a real passage of what the user knows.
        # The 8B often forgets to quote: ask it once more for the exact quote.
        if choice != DONT_KNOW and not self._grounded(evidence):
            follow = msgs + [
                {"role": "assistant", "content": json.dumps({"evidence": evidence, "choice": choice}, ensure_ascii=False)},
                {"role": "user", "content": (f"Quel passage EXACT de ta description justifie « {choice} » ? Recopie-le mot pour mot "
                                             f"dans evidence. S'il n'y en a pas, choice doit être « {DONT_KNOW} ». JSON complet.")},
            ]
            try:
                r2 = llm.chat(follow, json_schema=_ANSWER_SCHEMA, temperature=0.0, max_tokens=160, model=self.model)
                choice2 = _snap(str((r2.json or {}).get("choice", DONT_KNOW)), options)
                evidence = str((r2.json or {}).get("evidence") or "").strip()
                choice = choice2 if choice2 == DONT_KNOW or self._grounded(evidence) else DONT_KNOW
            except llm.LLMError:
                choice = DONT_KNOW
            if choice == DONT_KNOW:
                logger.debug("unsupported answer (quote: %r) → %s", evidence, DONT_KNOW)
        self.history.append({"q": question, "a": choice})
        return choice


def _snap(choice: str, options: list[str]) -> str:
    """Snaps the answer onto an exact option (or `DONT_KNOW`)."""
    c = choice.strip().lower()
    if not c or DONT_KNOW in c or c in {"i don't know", "unknown", "?"}:
        return DONT_KNOW
    for o in options:
        if c == o.strip().lower():
            return o
    import re
    for o in options:
        ol = o.strip().lower()
        if re.search(rf"(?<!\w){re.escape(ol)}(?!\w)", c) or (len(c) >= 3 and c in ol):
            return o
    return DONT_KNOW


def from_tot(query: Any, informed: bool = False, year: int | None = None, language: str | None = None) -> SimulatedUser:
    """Builds the user from a ToT_Music request without ever reading its `answer_*` fields."""
    return SimulatedUser(description=query.visible_text, informed=informed,
                         known_year=year if informed else None, known_language=language if informed else None)
