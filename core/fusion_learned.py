"""🧪 Learned fusion (recommendation no. 4): logistic regression over simple
per-(query, candidate) features — score and rank per branch, presence in the branch, input
type, agreement with the decade and language hints. Trained on the train/val splits
of ToT_Music (never on the evaluation sample), evaluated on T1 test. More defensible
than a fixed-weight RRF, and RRF remains the baseline.

Model serialised in data/index/fusion_lr.pkl; `LearnedFusion.rank(lists, hints, meta)`
returns [(track_id, probability)] sorted, like `rrf`.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from core.config import get_settings

logger = logging.getLogger(__name__)

Hit = tuple[str, float]


def _norm(hits: Sequence[Hit]) -> dict[str, tuple[float, int]]:
    """{id: (min-max normalised score, rank)}"""
    if not hits:
        return {}
    vals = [s for _, s in hits]
    lo, hi = min(vals), max(vals)
    out = {}
    for r, (i, s) in enumerate(hits, 1):
        if i not in out:
            out[i] = ((s - lo) / (hi - lo) if hi > lo else 1.0, r)
    return out


@dataclass
class LearnedFusion:
    branches: list[str]
    model: object = None                      # sklearn LogisticRegression
    input_types: list[str] = field(default_factory=lambda: ["text", "audio", "humming"])

    # ---- features ----
    def feature_names(self) -> list[str]:
        names = []
        for b in self.branches:
            names += [f"{b}:present", f"{b}:score", f"{b}:inv_rank"]
        names += ["n_branches", "decade_match", "decade_adjacent", "language_match"] + [f"input:{t}" for t in self.input_types]
        return names

    def features(self, lists: Mapping[str, Sequence[Hit]], hints: Mapping | None, meta: Mapping[str, Mapping],
                 input_type: str = "text") -> tuple[list[str], np.ndarray]:
        per = {b: _norm(lists.get(b, [])) for b in self.branches}
        ids = sorted({i for d in per.values() for i in d})
        hints = hints or {}
        dec = hints.get("decade")
        dec_val = int(str(dec)[:4]) if dec and str(dec)[:4].isdigit() else None
        lang = str(hints.get("language") or "").lower() or None
        X = np.zeros((len(ids), len(self.feature_names())), np.float32)
        for row, i in enumerate(ids):
            col = 0
            n = 0
            for b in self.branches:
                if i in per[b]:
                    s, r = per[b][i]
                    X[row, col:col + 3] = [1.0, s, 1.0 / r]
                    n += 1
                col += 3
            m = meta.get(i, {})
            y = m.get("year")
            d = int(y) // 10 * 10 if y else None
            X[row, col] = n / max(len(self.branches), 1); col += 1
            X[row, col] = 1.0 if (dec_val is not None and d == dec_val) else 0.0; col += 1
            X[row, col] = 1.0 if (dec_val is not None and d is not None and abs(d - dec_val) == 10) else 0.0; col += 1
            X[row, col] = 1.0 if (lang and str(m.get("language") or "").lower() == lang) else 0.0; col += 1
            for t in self.input_types:
                X[row, col] = 1.0 if input_type == t else 0.0; col += 1
        return ids, X

    # ---- training / inference ----
    def fit(self, samples: list[tuple[Mapping[str, Sequence[Hit]], Mapping | None, Mapping[str, Mapping], str, set[str]]]) -> dict:
        """`samples`: (lists, hints, meta, input_type, answer_ids). Pointwise: 1 if candidate = answer."""
        from sklearn.linear_model import LogisticRegression
        Xs, ys = [], []
        for lists, hints, meta, itype, answers in samples:
            ids, X = self.features(lists, hints, meta, itype)
            if not ids:
                continue
            Xs.append(X); ys.append(np.array([1.0 if i in answers else 0.0 for i in ids], np.float32))
        X = np.concatenate(Xs); y = np.concatenate(ys)
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X, y)
        coef = dict(zip(self.feature_names(), self.model.coef_[0].round(3).tolist()))
        return {"n_pairs": int(len(y)), "n_pos": int(y.sum()), "coef": coef}

    def rank(self, lists: Mapping[str, Sequence[Hit]], hints: Mapping | None, meta: Mapping[str, Mapping],
             input_type: str = "text", top: int | None = None) -> list[Hit]:
        ids, X = self.features(lists, hints, meta, input_type)
        if not ids:
            return []
        p = self.model.predict_proba(X)[:, 1]
        order = np.argsort(-p)
        out = [(ids[j], float(p[j])) for j in order]
        return out[:top] if top else out

    # ---- persistence ----
    @classmethod
    def path(cls) -> Path:
        return get_settings().index_dir / "fusion_lr.pkl"

    def save(self) -> Path:
        p = self.path(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(self))
        return p

    @classmethod
    def load(cls) -> "LearnedFusion":
        return pickle.loads(cls.path().read_bytes())
