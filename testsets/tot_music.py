"""T1 — ToT_Music (Bhargav, Schuth, Hauff, SIGIR 2023), CC0-1.0 licence.

Source: https://github.com/spotify-research/tot (file `data.jsonl.zip`, one
request per line). 2,274 resolved r/TipOfMyTongue requests, each linked to a
track of the WASABI 2.0 corpus (`wasabi_2.0_id`, `wasabi_isrc`) and to its
Spotify resolution (title, artists).

This module only **fetches and parses**: building the evaluation set (catalog
filtering, stratified sample) is done in phase 2.

Anti-leak rule: `ToTQuery.text` and `ToTQuery.title` are the only fields an
agent or a simulated user is allowed to see. The answer (`answer_*`) must
never enter a prompt.
"""

from __future__ import annotations

import io
import json
import logging
import re
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from core.config import get_settings

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/spotify-research/tot"
DATA_ZIP_URL = "https://github.com/spotify-research/tot/raw/main/data.jsonl.zip"

# Tags that post authors add to the title: [TOMT][SONG][ROCK][80s]
_TAG_RE = re.compile(r"\[([^\]]+)\]")
_DECADE_RE = re.compile(r"(?<!\d)(?:19|20)?(\d0)'?s\b", re.IGNORECASE)


@dataclass
class ToTQuery:
    id: str
    title: str                      # cleaned post title (visible to the agent)
    text: str                       # cleaned post body (visible to the agent)
    raw_title: str                  # raw title with [TOMT][SONG]... tags (visible)
    query_types: list[str]
    split: str | None               # train / val / test / None (split from the paper)
    # ---- answer: NEVER in a prompt ----
    answer_wasabi_id: str
    answer_isrc: str
    answer_title: str
    answer_artists: list[str]
    answer_spotify_urls: list[str] = field(default_factory=list)
    answer_isrcs: list[str] = field(default_factory=list)      # C1: all acceptable recordings
    reformulations: dict | None = None                          # C1: LLM reformulations from the paper (baseline)
    annotation_labels: list[str] = field(default_factory=list)  # C1: per-sentence labels (genre, era…)
    # "informed user" hints derived from the title tags (decade)
    tag_decade: str | None = None
    tag_genre: str | None = None

    @property
    def visible_text(self) -> str:
        """What the agent sees: post title + text, nothing else."""
        return f"{self.title}\n\n{self.text}".strip()


def raw_dir() -> Path:
    return get_settings().data_dir / "raw" / "tot"


def jsonl_path() -> Path:
    return raw_dir() / "data.jsonl"


def download(force: bool = False) -> Path:
    """Downloads `data.jsonl.zip` from the repository and unpacks it into data/raw/tot/."""
    out = jsonl_path()
    if out.exists() and not force:
        return out
    raw_dir().mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ToT_Music from %s", DATA_ZIP_URL)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        r = client.get(DATA_ZIP_URL)
        r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".jsonl")]
        if not names:
            raise RuntimeError("data.jsonl not found in the ToT_Music archive")
        out.write_bytes(zf.read(names[0]))
    logger.info("ToT_Music written to %s", out)
    return out


def _parse_tags(raw_title: str) -> tuple[str | None, str | None]:
    decade = genre = None
    for tag in _TAG_RE.findall(raw_title or ""):
        t = tag.strip()
        up = t.upper()
        if up in {"TOMT", "SONG", "MUSIC", "SOLVED"}:
            continue
        m = _DECADE_RE.search(t)
        if m and decade is None:
            d = m.group(1)
            decade = ("20" if d in {"00", "10", "20"} else "19") + d + "s"
            continue
        if genre is None and len(t) <= 20 and not any(c.isdigit() for c in t):
            genre = t
    return decade, genre


def _parse_row(row: dict) -> ToTQuery:
    sp = row.get("spotify_resolved") or {}
    items = sp.get("items") or []
    artists: list[str] = []
    for it in items:
        for a in it.get("artists") or []:
            if a.get("name") and a["name"] not in artists:
                artists.append(a["name"])
    if not artists:
        for a in sp.get("artists") or []:
            if isinstance(a, dict) and a.get("name"):
                artists.append(a["name"])
    decade, genre = _parse_tags(row.get("raw_title") or "")
    isrcs: list[str] = []
    for it in items:
        for code in it.get("isrc") or []:
            if code and code not in isrcs:
                isrcs.append(code)
    if row.get("wasabi_isrc") and row["wasabi_isrc"] not in isrcs:
        isrcs.insert(0, row["wasabi_isrc"])
    labels: list[str] = []
    for a in row.get("annotations") or []:
        for lab in (a.get("labels") or []):
            lab = str(lab)
            if lab not in labels:
                labels.append(lab)
    return ToTQuery(
        id=row["id"],
        title=(row.get("title") or "").strip(),
        text=(row.get("text") or "").strip(),
        raw_title=(row.get("raw_title") or "").strip(),
        query_types=list(row.get("query_types") or []),
        split=row.get("split"),
        answer_wasabi_id=str(row["wasabi_2.0_id"]),
        answer_isrc=str(row.get("wasabi_isrc") or ""),
        answer_title=sp.get("title") or "",
        answer_artists=artists,
        answer_spotify_urls=[it["url"] for it in items if it.get("url")],
        answer_isrcs=isrcs,
        reformulations=row.get("reformulations") or None,
        annotation_labels=labels,
        tag_decade=decade,
        tag_genre=genre,
    )


def load(download_if_missing: bool = True) -> list[ToTQuery]:
    p = jsonl_path()
    if not p.exists():
        if not download_if_missing:
            raise FileNotFoundError(p)
        download()
    out: list[ToTQuery] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(_parse_row(json.loads(line)))
    logger.info("ToT_Music: %d requests loaded", len(out))
    return out


def answer_ids(queries: list[ToTQuery] | None = None) -> dict[str, dict]:
    """Answer tracks, indexed by WASABI id, for phase 1
    (`catalog/build.py` must include all of them as a priority)."""
    queries = queries if queries is not None else load()
    out: dict[str, dict] = {}
    for q in queries:
        out.setdefault(q.answer_wasabi_id, {
            "wasabi_id": q.answer_wasabi_id,
            "isrc": q.answer_isrc,
            "title": q.answer_title,
            "artists": q.answer_artists,
            "n_queries": 0,
        })["n_queries"] += 1
    return out


if __name__ == "__main__":
    import collections
    logging.basicConfig(level=logging.INFO)
    qs = load()
    print(f"{len(qs)} requests, {len(answer_ids(qs))} distinct answer tracks")
    print("splits:", collections.Counter(q.split for q in qs))
    print("decades (tags):", collections.Counter(q.tag_decade for q in qs).most_common(8))


# --------------------------------------------------------------------------- Phase 2: frozen T1 set

def build_t1(catalog: "pd.DataFrame", previews: "pd.DataFrame | None" = None, n_eval: int = 200,
             seed: int = 20261028) -> tuple[list[dict], list[dict]]:
    """T1: requests whose answer is in the catalog **with a preview**.
    C1: the evaluation sample is drawn from the paper's **test** `split` (stratified by the
    answer track's decade); `train`/`val` and requests without a split go to the reserve
    (hyperparameter tuning on `train`/`val` only). A result is correct if it matches one of
    the acceptable recordings (`reponse_ids`, all resolved ISRCs).
    Fields: id, texte, reponse_id (primary), reponse_ids, decennie, split_article, query_types,
    labels, reformulations (never shown to the agent)."""
    import pandas as pd  # noqa: F811
    import random

    cat = catalog.set_index("wasabi_id", drop=False)
    by_isrc = catalog[catalog.isrc.notna()].drop_duplicates("isrc").set_index("isrc")
    with_preview = set(previews[previews.status == "ok"].track_id) if previews is not None else None
    rows = []
    for q in load():
        if q.answer_wasabi_id not in cat.index:
            continue
        c = cat.loc[q.answer_wasabi_id]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[0]
        ids = [c.track_id] + [by_isrc.at[i, "track_id"] for i in q.answer_isrcs if i in by_isrc.index and by_isrc.at[i, "track_id"] != c.track_id]
        if with_preview is not None:
            ids = [i for i in ids if i in with_preview]
        if not ids:
            continue
        year = int(c.year) if pd.notna(c.year) else None
        rows.append({"id": q.id, "texte": q.visible_text, "reponse_id": ids[0], "reponse_ids": ids,
                     "decennie": f"{year // 10 * 10}s" if year else "inconnue", "split_article": q.split,
                     "query_types": q.query_types, "labels": q.annotation_labels, "reformulations": q.reformulations})
    rng = random.Random(seed)
    rng.shuffle(rows)
    test_rows = [r for r in rows if r["split_article"] == "test"]
    pool = test_rows if len(test_rows) >= n_eval else rows
    by_dec: dict[str, list[dict]] = {}
    for r in pool:
        by_dec.setdefault(r["decennie"], []).append(r)
    total = len(pool)
    eval_rows: list[dict] = []
    for dec, lst in sorted(by_dec.items(), key=lambda kv: -len(kv[1])):
        k = max(1, round(n_eval * len(lst) / total))
        eval_rows += lst[:k]
    eval_rows = eval_rows[:n_eval]
    chosen = {r["id"] for r in eval_rows}
    reserve = [r for r in rows if r["id"] not in chosen]
    return eval_rows, reserve
