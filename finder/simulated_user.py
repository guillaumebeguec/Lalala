"""Utilisateur simulé pour évaluer l'agent sans humain (DEMO.md § 3.3, PLAN.md phase 2.6).

Un Llama distinct de l'agent joue la personne qui cherche le morceau. Il ne
connaît **que le texte d'origine de la demande** (titre + corps du post pour
ToT_Music, la description pour T2). Il répond aux questions fermées de l'agent
à partir de ce texte seulement, et « je ne sais pas » sinon.

Variante « informé » (`informed=True`) : il connaît en plus l'année et la langue
du vrai morceau (borne haute). Rien d'autre : jamais le titre ni l'artiste.

Garantie anti-fuite : `SimulatedUser.prompt_messages()` est la seule façon de
construire le prompt, et `tests/test_simulated_user.py` vérifie que la réponse
attendue n'y apparaît jamais.
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
    description: str                      # texte d'origine (seule connaissance)
    informed: bool = False
    known_year: int | None = None         # uniquement si informed
    known_language: str | None = None     # uniquement si informed
    model: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.informed:
            # Un utilisateur non informé ne porte aucune information externe, même par erreur.
            self.known_year = None
            self.known_language = None

    # ---- construction du prompt : point unique ----
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
        """Le premier message envoyé à l'agent : la description telle quelle."""
        return self.description

    def answer(self, question: str, options: list[str]) -> str:
        msgs = self.prompt_messages(question, options)
        try:
            r = llm.chat(msgs, json_schema=_ANSWER_SCHEMA, temperature=0.0, max_tokens=160, model=self.model)
            choice = str((r.json or {}).get("choice", DONT_KNOW)).strip()
            evidence = str((r.json or {}).get("evidence") or "").strip()
        except llm.LLMError as e:
            logger.warning("utilisateur simulé : %s ; réponse « %s »", e, DONT_KNOW)
            choice, evidence = DONT_KNOW, ""
        choice = _snap(choice, options)
        # Garde-fou : une réponse doit être étayée par un passage réel de ce que l'utilisateur sait.
        # Le 8B oublie souvent de citer : on lui redemande une fois la citation exacte.
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
                logger.debug("réponse non étayée (citation : %r) → %s", evidence, DONT_KNOW)
        self.history.append({"q": question, "a": choice})
        return choice


def _snap(choice: str, options: list[str]) -> str:
    """Ramène la réponse sur une option exacte (ou « je ne sais pas »)."""
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
    """Construit l'utilisateur depuis une demande ToT_Music sans jamais lire ses champs `answer_*`."""
    return SimulatedUser(description=query.visible_text, informed=informed,
                         known_year=year if informed else None, known_language=language if informed else None)
