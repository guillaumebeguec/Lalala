"""Contours mélodiques (RMVPE) des extraits d'un sous-ensemble → `data/index/contours_<subset>.npz`.

    uv run python scripts/build_contours.py --subset dev [--workers 3] [--extra-ids data/testsets/t6a_chad_hummings.jsonl] [--limit N]

Même pool que les index audio (sous-ensemble + réponses des jeux `--extra-ids`). Les contours sont
mis en cache par hash du contenu audio (`data/cache/contours/`) : relance gratuite. RMVPE tourne sur
le CPU (≈ 3–4 s par extrait de 30 s) ; `--workers` lance des processus indépendants.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.previews import fetch_audio, load_previews  # noqa: E402
from core.cache import DiskCache  # noqa: E402
from core.melody import ContourStore, contour_from_track  # noqa: E402
from testsets.common import read_jsonl  # noqa: E402

log = logging.getLogger("build_contours")
_EMB = None


def _worker(job: tuple[str, str, int | None]) -> tuple[str, np.ndarray | None]:
    """Un processus = un RMVPE ; renvoie (track_id, contour) ou (track_id, None)."""
    global _EMB
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tid, url, dz = job
    try:
        data = fetch_audio(tid, url, deezer_id=dz)
        if not data:
            return tid, None
        cache = DiskCache("contours")
        key = cache.key(data, v=1)
        c = cache.get_array(key)
        if c is None:
            if _EMB is None:
                from core.embedders.hum2song import Hum2SongEmbedder
                _EMB = Hum2SongEmbedder(role="song")
            c = contour_from_track(_EMB.pitch_track(data))
            cache.put_array(key, c)
        return tid, c
    except Exception as e:  # noqa: BLE001
        log.warning("%s : %s", tid, e)
        return tid, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--extra-ids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--provider", choices=["catalog", "itunes"], default="catalog",
                    help="itunes : seconde fenêtre de 30 s par morceau via l'extrait iTunes (recherche titre/artiste, ~20 req/min) → contours_<subset>_itunes.npz")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cat = cb.load(args.subset)
    ids = list(cat.track_id)
    if args.extra_ids:
        extra = set()
        for f in args.extra_ids.split(","):
            extra |= {r["reponse_id"] for r in read_jsonl(Path(f))}
        ids += sorted(extra - set(ids))
    prev = load_previews()
    prev = prev[prev.status == "ok"].set_index("track_id")
    ids = [t for t in ids if t in prev.index]
    if args.limit:
        ids = ids[: args.limit]
    if args.provider == "itunes":
        # seconde fenêtre : extrait iTunes (souvent un autre passage du morceau que l'extrait Deezer)
        from catalog.previews import ITunesClient
        it = ITunesClient()
        meta = cb.load("full").set_index("track_id")
        jobs = []
        t0 = time.time()
        for i, t in enumerate(ids, 1):
            if prev.at[t, "provider"] == "itunes" or t not in meta.index:
                continue      # déjà la fenêtre iTunes dans le magasin principal
            try:
                m = it.best_match(str(meta.at[t, "artist"]), str(meta.at[t, "title"]))
            except Exception as e:  # noqa: BLE001
                log.warning("itunes %s : %s", t, e)
                m = None
            if m and m.get("previewUrl"):
                jobs.append((t, m["previewUrl"], None))
            if i % 200 == 0:
                log.info("recherche iTunes : %d / %d (%d extraits trouvés, %.0f s)", i, len(ids), len(jobs), time.time() - t0)
        log.info("iTunes : %d extraits sur %d morceaux", len(jobs), len(ids))
    else:
        jobs = [(t, prev.at[t, "preview_url"], prev.at[t, "deezer_id"]) for t in ids]
    log.info("%d extraits, %d processus", len(jobs), args.workers)

    contours: dict[str, np.ndarray] = {}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, (tid, c) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=4), 1):
            if c is not None and len(c):
                contours[tid] = c
            if i % 100 == 0 or i == len(jobs):
                log.info("contours : %d / %d (%.1f s/extrait, %d ok)", i, len(jobs), (time.time() - t0) / i, len(contours))
    store = ContourStore.build(contours)
    p = store.path(args.subset, provider=None if args.provider == "catalog" else args.provider)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, ids=np.asarray(store.ids), data=store.data)
    voiced = float(np.mean(~np.isnan(store.data))) if len(store) else 0.0
    log.info("contours : {'n': %d, 'trames': %d, 'voisé': %.2f, 'fichier': '%s'}", len(store), store.data.shape[1] if len(store) else 0, voiced, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
