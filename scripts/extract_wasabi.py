"""Extracts the WASABI metadata (song/album/artist) from the JSON dump to Parquet.

    uv run python scripts/extract_wasabi.py            # data/raw/wasabi/json → data/catalog/wasabi
    uv run python scripts/extract_wasabi.py --src DIR --dst DIR [--force] [--tolerate-truncated]

Prerequisites: `wasabi-2-0.tar` downloaded from Zenodo and `json/json.zip` extracted
(song.json, album.json, artist-*.json). Replayable: existing Parquet files are kept.
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
    ap.add_argument("--src", type=Path, default=None, help="folder containing song.json, album.json, artist-*.json")
    ap.add_argument("--dst", type=Path, default=None, help="output folder for the Parquet files")
    ap.add_argument("--force", action="store_true", help="recompute even if the Parquet files exist")
    ap.add_argument("--tolerate-truncated", action="store_true", help="accept a truncated JSON (samples)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    t0 = time.time()
    counts = wasabi.extract(args.src, args.dst, tolerate_truncated=args.tolerate_truncated, force=args.force)
    print(f"\nExtraction finished in {time.time() - t0:.0f} s: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
