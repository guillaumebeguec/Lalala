"""Résout les métadonnées d'extrait (URL Deezer/iTunes courante, contrôle d'appariement)
pour tout le catalogue. Métadonnées seulement : aucun audio n'est téléchargé ici.

    uv run python scripts/resolve_previews.py [--subset dev|full] [--no-itunes] [--workers 4]

Rejouable (reprise ligne à ligne). Met ensuite à jour eval/catalog.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.previews import resolve_all  # noqa: E402
from catalog.report import write_catalog_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["dev", "full"], default="full")
    ap.add_argument("--no-itunes", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cat = cb.load(args.subset)
    df = resolve_all(cat, use_itunes=not args.no_itunes, workers=args.workers)
    full = cb.load("full")
    write_catalog_report(full, previews=df)
    print(f"\n{len(df)} lignes ; ok {100 * df.status.eq('ok').mean():.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
