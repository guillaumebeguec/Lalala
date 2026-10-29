"""Extrait les métadonnées WASABI (song/album/artist) du dump JSON vers Parquet.

    uv run python scripts/extract_wasabi.py            # data/raw/wasabi/json → data/catalog/wasabi
    uv run python scripts/extract_wasabi.py --src DIR --dst DIR [--force] [--tolerate-truncated]

Prérequis : `wasabi-2-0.tar` téléchargé depuis Zenodo et `json/json.zip` extrait
(song.json, album.json, artist-*.json). Rejouable : les Parquet existants sont gardés.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import wasabi  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None, help="dossier contenant song.json, album.json, artist-*.json")
    ap.add_argument("--dst", type=Path, default=None, help="dossier de sortie des Parquet")
    ap.add_argument("--force", action="store_true", help="recalculer même si les Parquet existent")
    ap.add_argument("--tolerate-truncated", action="store_true", help="accepter un JSON tronqué (échantillons)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    counts = wasabi.extract(args.src, args.dst, tolerate_truncated=args.tolerate_truncated, force=args.force)
    print(f"\nExtraction terminée en {time.time() - t0:.0f} s : {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
