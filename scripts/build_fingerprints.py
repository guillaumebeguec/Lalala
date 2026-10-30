"""Empreintes Chromaprint des extraits du catalogue (baseline type Shazam), audio lu en mémoire.

    uv run python scripts/build_fingerprints.py --subset dev|full [--limit N]

Reprise : les empreintes déjà présentes dans data/index/fingerprints.npz sont gardées.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.previews import fetch_audio, load_previews  # noqa: E402
from core.fingerprint import FingerprintIndex, fpcalc_raw  # noqa: E402

log = logging.getLogger("build_fingerprints")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cat = cb.load(args.subset)
    prev = load_previews().set_index("track_id")
    fps = FingerprintIndex.load().fps if FingerprintIndex.path().exists() else {}
    ids = [t for t in cat.track_id if t in prev.index and prev.at[t, "status"] == "ok" and t not in fps]
    if args.limit:
        ids = ids[: args.limit]
    t0 = time.time()
    n_fail = 0
    for i, tid in enumerate(ids, 1):
        try:
            data = fetch_audio(tid, prev.at[tid, "preview_url"], deezer_id=prev.at[tid, "deezer_id"])
            if data:
                fps[tid] = fpcalc_raw(data)
            else:
                n_fail += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            log.warning("%s : %s", tid, e)
        if i % 200 == 0:
            FingerprintIndex(fps).save()
            log.info("%d / %d (%.2f s/extrait, échecs %d)", i, len(ids), (time.time() - t0) / i, n_fail)
    FingerprintIndex(fps).save()
    print(f"{len(fps)} empreintes ({n_fail} échecs) en {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
