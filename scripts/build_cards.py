"""Génère les fiches morceaux (Wikipédia FR/EN + description audio Llama) pour un sous-ensemble.

    uv run python scripts/build_cards.py --subset dev [--no-wiki] [--no-llm] [--limit N]

Rejouable : Wikipédia et Llama sont en cache disque (data/cache/wikipedia, llama_audio_desc).
Sortie : data/catalog/cards_<subset>.parquet.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.cards import build_cards  # noqa: E402
from core.config import get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--no-wiki", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cat = cb.load(args.subset)
    if args.limit:
        cat = cat.head(args.limit)
    out = get_settings().data_dir / "catalog" / f"cards_{args.subset}.parquet"
    t0 = time.time()
    df = build_cards(cat, with_wiki=not args.no_wiki, with_llm=not args.no_llm, out=out, log_every=50)
    dt = time.time() - t0
    print(f"\n{len(df)} fiches en {dt:.0f} s ({dt / max(len(df), 1):.2f} s/fiche) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
