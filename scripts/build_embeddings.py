"""Computes the catalogue embeddings for an embedder: audio of the previews (read in
memory, never written to disk) and text of the cards.

    uv run python scripts/build_embeddings.py --embedder clap|peav|gemma|gemini --subset dev|full [--modality audio,text] [--limit N]

Resumable: every vector is cached (data/cache/emb_<embedder>/, key = content hash).
Output: data/index/<embedder>_<modality>/ (FAISS) via core.index.VectorIndex, and a
time/coverage summary in data/index/<embedder>_<modality>/stats.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.previews import fetch_audio, load_previews  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.embedders.base import get_embedder  # noqa: E402
from core.index import VectorIndex, index_dir  # noqa: E402

log = logging.getLogger("build_embeddings")


def audio_vectors(embs: dict, cat: pd.DataFrame, prev: pd.DataFrame, batch: int, limit: int | None) -> tuple[list[str], dict[str, np.ndarray], dict]:
    """A single pass over the audio (read in memory) for all requested embedders."""
    ok = prev[prev.status == "ok"].set_index("track_id")
    ids = [t for t in cat.track_id if t in ok.index]
    if limit:
        ids = ids[:limit]
    vecs: dict[str, list[np.ndarray]] = {k: [] for k in embs}
    times: dict[str, float] = {k: 0.0 for k in embs}
    done_ids, n_fail = [], 0
    t0 = time.time()
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        items, keep = [], []
        for tid in chunk:
            try:
                data = fetch_audio(tid, ok.at[tid, "preview_url"], deezer_id=ok.at[tid, "deezer_id"])
            except Exception as e:  # noqa: BLE001
                log.warning("preview %s: %s", tid, e)
                data = None
            if data:
                items.append(data)
                keep.append(tid)
            else:
                n_fail += 1
        if items:
            # decode first: an unreadable preview is dropped without failing the batch
            from core.audio_io import load_audio
            good_items, good_ids = [], []
            for data, tid in zip(items, keep):
                try:
                    load_audio(data, sr=16_000)
                    good_items.append(data); good_ids.append(tid)
                except Exception as e:  # noqa: BLE001
                    n_fail += 1
                    log.warning("undecodable preview %s: %s", tid, str(e)[:120])
            items, keep = good_items, good_ids
        if items:
            for name, emb in embs.items():
                t1 = time.time()
                vecs[name].append(emb.embed_audio(items, batch_size=batch))
                times[name] += time.time() - t1
            done_ids += keep
        if (start // batch) % 20 == 0:
            el = time.time() - t0
            log.info("audio %d / %d (%.2f s/preview, %d failures); %s", len(done_ids), len(ids), el / max(len(done_ids), 1), n_fail,
                     ", ".join(f"{k} {v / max(len(done_ids), 1):.2f} s" for k, v in times.items()))
    V = {k: (np.concatenate(v) if v else np.zeros((0, embs[k].dim), np.float32)) for k, v in vecs.items()}
    return done_ids, V, {"n": len(done_ids), "n_fail": n_fail, "seconds": round(time.time() - t0, 1),
                         "seconds_per_embedder": {k: round(v, 1) for k, v in times.items()}}


def text_vectors(emb, cards: pd.DataFrame, batch: int, limit: int | None) -> tuple[list[str], np.ndarray, dict]:
    df = cards.head(limit) if limit else cards
    t0 = time.time()
    V = emb.embed_text(df.card_text.tolist(), task="document", batch_size=batch)
    return df.track_id.tolist(), V, {"n": len(df), "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", required=True, help="one or more of clap,peav,gemma,gemini,coverhunter (comma-separated)")
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--modality", default="audio,text")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--extra-ids", default=None, help="jsonl files (comma-separated) whose `reponse_id` are added to the indexed pool (e.g. T6a)")
    ap.add_argument("--shard", default=None, help="i/n: only computes (and caches) the previews whose index ≡ i mod n, without writing an index; "
                                                   "run n processes then a final pass without --shard to assemble from the cache")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    names = [n.strip() for n in args.embedder.split(",") if n.strip()]
    embs = {n: get_embedder(n) for n in names}
    cat = cb.load(args.subset)
    if args.extra_ids:
        from testsets.common import read_jsonl
        extra = set()
        for f in args.extra_ids.split(","):
            extra |= {c["reponse_id"] for c in read_jsonl(Path(f.strip()))}
        full = cb.load("full")
        add = full[full.track_id.isin(extra) & ~full.track_id.isin(set(cat.track_id))]
        log.info("pool: %d tracks from the subset + %d extra targets", len(cat), len(add))
        cat = pd.concat([cat, add], ignore_index=True)
    mods = {m.strip() for m in args.modality.split(",")}
    if "audio" in mods:
        prev = load_previews()
        if args.shard:
            i, n = (int(x) for x in args.shard.split("/"))
            cat = cat.iloc[i::n]
            log.info("shard %d/%d: %d previews (cache only)", i, n, len(cat))
            _, _, stats = audio_vectors(embs, cat, prev, args.batch, args.limit)
            log.info("shard %d/%d done: %s", i, n, stats)
            return 0
        ids, V, stats = audio_vectors(embs, cat, prev, args.batch, args.limit)
        for n, vec in V.items():
            VectorIndex.build(n, "audio", ids, vec).save()
            st = dict(stats, subset=args.subset, dim=int(vec.shape[1]) if len(vec) else embs[n].dim, embedder=n)
            (index_dir(n, "audio") / "stats.json").write_text(json.dumps(st, indent=1))
        log.info("audio : %s", stats)
    if "text" in mods:
        cards_path = get_settings().data_dir / "catalog" / f"cards_{args.subset}.parquet"
        if not cards_path.exists():
            log.error("cards missing: %s (scripts/build_cards.py)", cards_path)
            return 1
        cards = pd.read_parquet(cards_path)
        cards = cards[cards.track_id.isin(set(cat.track_id))]
        for n, emb in embs.items():
            if n == "coverhunter":
                log.info("coverhunter: no text index (audio only)")
                continue
            ids, V, stats = text_vectors(emb, cards, max(args.batch, 16), args.limit)
            VectorIndex.build(n, "text", ids, V).save()
            stats.update(subset=args.subset, dim=int(V.shape[1]) if len(V) else emb.dim)
            (index_dir(n, "text") / "stats.json").write_text(json.dumps(stats, indent=1))
            log.info("text %s: %s", n, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
