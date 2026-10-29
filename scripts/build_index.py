"""Construit l'index BM25 sur les fiches d'un sous-ensemble (les index FAISS sont écrits
par build_embeddings.py).

    uv run python scripts/build_index.py --subset dev|full
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings  # noqa: E402
from core.index import BM25Index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    cards = pd.read_parquet(get_settings().data_dir / "catalog" / f"cards_{args.subset}.parquet")
    idx = BM25Index(cards.track_id.tolist(), cards.card_text.tolist())
    p = idx.save(get_settings().index_dir / f"bm25_cards_{args.subset}.pkl")
    print(f"BM25 : {len(cards)} fiches → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
