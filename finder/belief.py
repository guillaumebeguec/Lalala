"""🧪 Questions par gain d'information (PLAN.md phase 7).

Croyance P(h) sur les candidats (top-N après reranking) + hypothèse « autre ».
Prior = softmax(score / température). Répertoire fixe de questions dont la réponse
se lit dans les métadonnées (décennie, langue, voix, genre, tempo, support), avec
un modèle d'observation P(r | h) exact depuis la fiche, lissé par ε (faux
souvenirs). « je ne sais pas » est peu informatif (probabilité fixe, indépendante
de h). Choix : argmax IG(q) = H(P) − Σ_r P(r) H(P | r). Mise à jour bayésienne.
Arrêt : max P(h) > `stop_prob` ou budget de questions.

Tout est exposé à chaque tour (entropie, distribution, question, gain attendu)
pour le panneau « enquête » du front.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

OTHER = "__other__"
DONT_KNOW = "je ne sais pas"

# --------------------------------------------------------------------------- répertoire de questions

GENRE_FAMILIES = {
    "pop": ["pop", "dance", "synth", "disco", "electro", "house", "techno"],
    "rock": ["rock", "metal", "punk", "grunge", "indie", "alternative"],
    "rap / r&b": ["hip hop", "rap", "r&b", "soul", "funk"],
    "chanson / variété": ["chanson", "variété", "folk", "country", "singer"],
    "jazz / classique / autre": ["jazz", "classical", "classique", "blues", "reggae", "latin", "world", "soundtrack"],
}


def _decade(meta: Mapping) -> str | None:
    y = meta.get("year")
    return f"{int(y) // 10 * 10}s" if y else None


def _language(meta: Mapping) -> str | None:
    l = (meta.get("language") or "").lower()
    if not l:
        return None
    if l in ("french", "fra", "fre", "fr"):
        return "français"
    if l in ("english", "eng", "en"):
        return "anglais"
    if l in ("zxx", "instrumental"):
        return "instrumental"
    return "autre langue"


def _genre(meta: Mapping) -> str | None:
    g = (meta.get("genre") or "").lower()
    if not g:
        return None
    for fam, keys in GENRE_FAMILIES.items():
        if any(k in g for k in keys):
            return fam
    return "jazz / classique / autre"


def _vocals(meta: Mapping) -> str | None:
    v = (meta.get("vocals") or "").lower()
    return {"male": "masculine", "female": "féminine", "mixed": "mixte", "instrumental": "instrumental"}.get(v)


def _tempo(meta: Mapping) -> str | None:
    t = (meta.get("tempo") or "").lower()
    if t in ("slow", "medium", "fast"):
        return {"slow": "lent", "medium": "moyen", "fast": "rapide"}[t]
    bpm = meta.get("bpm")
    if bpm:
        return "lent" if bpm < 95 else ("rapide" if bpm > 130 else "moyen")
    return None


def _medium(meta: Mapping) -> str | None:
    u = (meta.get("usage") or "").lower()
    if not u:
        return None
    if "pub" in u or " ad" in f" {u}":
        return "une publicité"
    if "série" in u or "serie" in u or "générique" in u:
        return "une série ou un générique"
    if "film" in u:
        return "un film"
    return "autre"


@dataclass
class Question:
    key: str
    text: str
    extractor: Callable[[Mapping], str | None]
    options_fn: Callable[[set[str]], list[str]]


def _decade_options(vals: set[str]) -> list[str]:
    return sorted(vals)


def _fixed(order: list[str]) -> Callable[[set[str]], list[str]]:
    return lambda vals: [o for o in order if o in vals] or sorted(vals)


REPERTOIRE: list[Question] = [
    Question("decade", "De quelle décennie date le morceau, à votre avis ?", _decade, _decade_options),
    Question("language", "Dans quelle langue sont les paroles ?", _language, _fixed(["français", "anglais", "autre langue", "instrumental"])),
    Question("vocals", "La voix principale est-elle masculine ou féminine ?", _vocals, _fixed(["masculine", "féminine", "mixte", "instrumental"])),
    Question("genre", "Quel style s'en rapproche le plus ?", _genre, _fixed(list(GENRE_FAMILIES))),
    Question("tempo", "Le tempo est plutôt… ?", _tempo, _fixed(["lent", "moyen", "rapide"])),
    Question("medium", "Où l'avez-vous entendu ?", _medium, _fixed(["une publicité", "une série ou un générique", "un film", "autre"])),
]


# --------------------------------------------------------------------------- croyance

def entropy(p: Sequence[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


def softmax(scores: Sequence[float], temperature: float) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    ex = [math.exp((s - m) / max(temperature, 1e-6)) for s in scores]
    z = sum(ex)
    return [e / z for e in ex]


@dataclass
class Belief:
    ids: list[str]                     # candidats + OTHER en dernier
    probs: list[float]
    meta: dict[str, Mapping]
    epsilon: float = 0.15              # erreur de mémoire : masse hors de la valeur de la fiche
    p_dont_know: float = 0.2           # probabilité de « je ne sais pas », indépendante de h
    history: list[dict[str, Any]] = field(default_factory=list)
    asked: set[str] = field(default_factory=set)

    @classmethod
    def from_candidates(cls, scored: Sequence[tuple[str, float]], meta: Mapping[str, Mapping], temperature: float = 0.15,
                        p_other: float = 0.15, **kw) -> "Belief":
        ids = [i for i, _ in scored]
        p = softmax([s for _, s in scored], temperature)
        probs = [x * (1 - p_other) for x in p] + [p_other]
        return cls(ids=ids + [OTHER], probs=probs, meta={i: dict(meta.get(i, {})) for i in ids}, **kw)

    # ---- modèle d'observation ----
    def likelihood(self, q: Question, response: str, options: list[str], h: str) -> float:
        """P(réponse | h). Pour OTHER ou fiche sans valeur : uniforme sur les options."""
        if response == DONT_KNOW:
            return self.p_dont_know
        rest = 1.0 - self.p_dont_know
        n = len(options)
        if h == OTHER:
            return rest / n
        v = q.extractor(self.meta[h])
        if v is None or v not in options:
            return rest / n
        if response == v:
            return rest * (1 - self.epsilon)
        return rest * self.epsilon / max(n - 1, 1)

    def _options(self, q: Question) -> list[str]:
        vals = {v for i in self.ids if i != OTHER for v in [q.extractor(self.meta[i])] if v}
        return q.options_fn(vals)

    def posterior(self, q: Question, response: str, options: list[str]) -> list[float]:
        joint = [p * self.likelihood(q, response, options, h) for h, p in zip(self.ids, self.probs)]
        z = sum(joint)
        return [x / z for x in joint] if z > 0 else list(self.probs)

    def expected_gain(self, q: Question) -> tuple[float, list[str]]:
        options = self._options(q)
        if len(options) < 2:
            return 0.0, options
        h0 = entropy(self.probs)
        exp_h = 0.0
        for r in options + [DONT_KNOW]:
            pr = sum(p * self.likelihood(q, r, options, h) for h, p in zip(self.ids, self.probs))
            if pr <= 0:
                continue
            exp_h += pr * entropy(self.posterior(q, r, options))
        return h0 - exp_h, options

    def best_question(self, min_gain: float = 0.05) -> dict[str, Any] | None:
        best = None
        for q in REPERTOIRE:
            if q.key in self.asked:
                continue
            gain, options = self.expected_gain(q)
            if gain > (best["expected_gain"] if best else min_gain):
                best = {"key": q.key, "question": q.text, "options": options[:4], "expected_gain": round(gain, 4),
                        "entropy": round(entropy(self.probs), 4)}
        return best

    def update(self, key: str, response: str, options: list[str]) -> dict[str, Any]:
        q = next(x for x in REPERTOIRE if x.key == key)
        before = entropy(self.probs)
        self.probs = self.posterior(q, response, options)
        self.asked.add(key)
        rec = {"key": key, "response": response, "entropy_before": round(before, 4), "entropy_after": round(entropy(self.probs), 4)}
        self.history.append(rec)
        return rec

    # ---- lecture ----
    @property
    def top(self) -> tuple[str, float]:
        i = max(range(len(self.probs)), key=lambda j: self.probs[j])
        return self.ids[i], self.probs[i]

    @property
    def p_other(self) -> float:
        return self.probs[self.ids.index(OTHER)]

    def distribution(self, n: int = 10) -> list[dict[str, Any]]:
        order = sorted(range(len(self.ids)), key=lambda j: -self.probs[j])
        return [{"id": self.ids[j], "p": round(self.probs[j], 4)} for j in order[:n]]

    def merge_candidates(self, scored: Sequence[tuple[str, float]], meta: Mapping[str, Mapping], temperature: float = 0.15) -> None:
        """Réinjecte de nouveaux candidats (après une recherche relancée) : la masse de OTHER
        est redistribuée sur les nouveaux selon leurs scores ; les anciens gardent leur probabilité."""
        new = [(i, s) for i, s in scored if i not in self.ids]
        if not new:
            return
        p_other = self.p_other
        share = softmax([s for _, s in new], temperature)
        keep = p_other * 0.5
        idx = self.ids.index(OTHER)
        self.ids = self.ids[:idx] + [i for i, _ in new] + [OTHER]
        self.probs = self.probs[:idx] + [(p_other - keep) * x for x in share] + [keep]
        for i, _ in new:
            self.meta[i] = dict(meta.get(i, {}))


def ig_policy(stop_prob: float = 0.7, temperature: float = 0.15, epsilon: float = 0.15, p_other: float = 0.15):
    """Fabrique une `question_policy` pour `finder.agent.Agent` : maintient une croyance sur
    `agent.state.candidates`, choisit la question au meilleur gain, s'arrête quand la croyance
    est concentrée. La croyance est stockée sur l'état (`state.belief`)."""

    def policy(agent, state):
        from finder.agent import SessionState  # noqa: F401 — typage souple
        meta = {c.track_id: agent.tools.meta_for_belief(c.track_id) for c in state.candidates.values()} if hasattr(agent.tools, "meta_for_belief") else \
            {c.track_id: {"year": c.year} for c in state.candidates.values()}
        scored = sorted(((c.track_id, c.score) for c in state.candidates.values()), key=lambda x: -x[1])[:20]
        b: Belief | None = getattr(state, "belief", None)
        if b is None:
            b = Belief.from_candidates(scored, meta, temperature=temperature, p_other=p_other, epsilon=epsilon)
            state.belief = b
        else:
            b.merge_candidates(scored, meta, temperature)
        # réponse à la question précédente
        if state.questions and state.questions[-1].get("key") and "answer" in state.questions[-1] and state.questions[-1].get("key") not in b.asked:
            q = state.questions[-1]
            b.update(q["key"], q["answer"], [o for o in q["options"] if o != DONT_KNOW])
        top_id, top_p = b.top
        if top_p >= stop_prob and top_id != OTHER:
            return None
        q = b.best_question()
        if q is None:
            return None
        return q

    return policy
