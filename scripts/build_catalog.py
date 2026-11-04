"""Builds the mainstream catalogue (metadata only, no audio).

    uv run python scripts/build_catalog.py [--target 25000] [--no-distractors] [--no-resolve-tot]

Prerequisites: `scripts/extract_wasabi.py` has been run; ToT_Music downloaded (automatic).
Replayable: all Deezer responses are cached in data/cache/deezer_api.
Produces data/catalog/catalog.parquet and catalog_dev.parquet, then eval/catalog.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.report import write_catalog_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=cb.TARGET_SIZE)
    ap.add_argument("--no-distractors", action="store_true")
    ap.add_argument("--no-resolve-tot", action="store_true", help="do not resolve the ToT answers on Deezer (ISRC)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    t0 = time.time()
    df = cb.build(target=args.target, resolve_tot=not args.no_resolve_tot, with_distractors=not args.no_distractors)
    path = write_catalog_report(df)
    print(f"\n{len(df)} tracks in {time.time() - t0:.0f} s; report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
