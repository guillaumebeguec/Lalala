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
    # index « contexte » : la fiche sans sa description audio générée (titre, artiste, résumé, thème des paroles)
    texts = cards.context_text.tolist() if "context_text" in cards.columns else cards.card_text.tolist()
    idx = BM25Index(cards.track_id.tolist(), texts)
    p = idx.save(get_settings().index_dir / f"bm25_cards_{args.subset}.pkl")
    print(f"BM25 contexte : {len(cards)} fiches → {p}")
    # index « son » : description audio générée seule, pour les mots sonores (instruments, voix, tempo)
    if "sound_text" in cards.columns:
        snd = cards[cards.sound_text.notna() & (cards.sound_text.astype(str).str.len() > 0)]
        if len(snd):
            ps = BM25Index(snd.track_id.tolist(), snd.sound_text.tolist()).save(get_settings().index_dir / f"bm25_sound_{args.subset}.pkl")
            print(f"BM25 son : {len(snd)} fiches avec description audio → {ps}")
        else:
            print("BM25 son : aucune description audio (lancer build_cards avec Llama)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
