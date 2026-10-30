"""Contrôles manuels demandés par PLAN.md (phase 1.4 : 30 appariements extrait ↔ métadonnées ;
phase 2.4 : 20 paires reprise → original) → eval/manual_checks.md à annoter à la main.

    uv run python scripts/review_pairs.py [--n-previews 30] [--n-covers 20] [--seed 7]

Pour chaque ligne : ce que dit le catalogue, ce que Deezer/iTunes a renvoyé, le score
d'appariement, et un lien d'écoute (`/api/preview/<id>` quand l'API tourne, sinon l'URL
d'extrait rafraîchie). Colonne « OK ? » à remplir (oui / non / douteux).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.previews import fresh_preview_url, load_previews  # noqa: E402
from core.config import PROJECT_ROOT  # noqa: E402
from testsets.common import read_jsonl, testsets_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-previews", type=int, default=30)
    ap.add_argument("--n-covers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cat = cb.load("full").set_index("track_id", drop=False)
    prev = load_previews().set_index("track_id")
    ok = prev[prev.status == "ok"]
    sample = ok.sample(min(args.n_previews, len(ok)), random_state=args.seed)
    lines = ["# Contrôles manuels", "", f"Échantillons tirés avec la graine {args.seed}. Remplir la colonne « OK ? ».", "",
             f"## {len(sample)} appariements extrait ↔ métadonnées (phase 1.4)", "",
             "| # | Catalogue (titre — artiste, année) | Résolu (titre — artiste) | Mode | Score | Écoute | OK ? |", "|---|---|---|---|---|---|---|"]
    for i, (tid, r) in enumerate(sample.iterrows(), 1):
        c = cat.loc[tid]
        url = fresh_preview_url(tid, r.preview_url, deezer_id=r.deezer_id) or r.preview_url
        year = int(c.year) if pd.notna(c.year) else "?"
        lines.append(f"| {i} | {c.title} — {c.artist}, {year} | {r.resolved_title} — {r.resolved_artist} | {r.get('resolved_by', r.provider)} | {r.match_score} | [écouter]({url}) | |")
    t4 = testsets_dir() / "t4_covers.jsonl"
    if t4.exists():
        rows = read_jsonl(t4)
        import random
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.n_covers]
        lines += ["", f"## {len(rows)} paires reprise → original (phase 2.4)", "",
                  "| # | Reprise (titre — artiste, type) | Original (titre — artiste, année) | Écoute reprise | Écoute original | OK ? |", "|---|---|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            o = cat.loc[r["reponse_id"]]
            ou = fresh_preview_url(r["reponse_id"], prev.at[r["reponse_id"], "preview_url"] if r["reponse_id"] in prev.index else None,
                                   deezer_id=prev.at[r["reponse_id"], "deezer_id"] if r["reponse_id"] in prev.index else None)
            cu = fresh_preview_url(f"cover{r['cover_deezer_id']}", r["cover_preview_url"], deezer_id=r["cover_deezer_id"]) or r["cover_preview_url"]
            year = int(o.year) if pd.notna(o.year) else "?"
            lines.append(f"| {i} | {r['cover_title']} — {r['cover_artist']}, {r['kind']} | {o.title} — {o.artist}, {year} | [écouter]({cu}) | [écouter]({ou}) | |")
    out = PROJECT_ROOT / "eval" / "manual_checks.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Écrit : {out} (les liens d'écoute Deezer expirent en moins d'une heure ; relancer le script pour les rafraîchir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
