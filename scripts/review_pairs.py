"""Manual checks requested by PLAN.md (phase 1.4: 30 preview ↔ metadata matches;
phase 2.4: 20 cover → original pairs) → eval/manual_checks.md to annotate by hand.

    uv run python scripts/review_pairs.py [--n-previews 30] [--n-covers 20] [--seed 7]

For each row: what the catalogue says, what Deezer/iTunes returned, the matching
score, and a listening link (`/api/preview/<id>` when the API is running, otherwise the
refreshed preview URL). "OK?" column to fill in (yes / no / doubtful).
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
    lines = ["# Manual checks", "", f"Samples drawn with seed {args.seed}. Fill in the “OK?” column.", "",
             f"## {len(sample)} preview ↔ metadata matches (phase 1.4)", "",
             "| # | Catalogue (title — artist, year) | Resolved (title — artist) | Mode | Score | Listen | OK? |", "|---|---|---|---|---|---|---|"]
    for i, (tid, r) in enumerate(sample.iterrows(), 1):
        c = cat.loc[tid]
        url = fresh_preview_url(tid, r.preview_url, deezer_id=r.deezer_id) or r.preview_url
        year = int(c.year) if pd.notna(c.year) else "?"
        lines.append(f"| {i} | {c.title} — {c.artist}, {year} | {r.resolved_title} — {r.resolved_artist} | {r.get('resolved_by', r.provider)} | {r.match_score} | [listen]({url}) | |")
    t4 = testsets_dir() / "t4_covers.jsonl"
    if t4.exists():
        rows = read_jsonl(t4)
        import random
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.n_covers]
        lines += ["", f"## {len(rows)} cover → original pairs (phase 2.4)", "",
                  "| # | Cover (title — artist, type) | Original (title — artist, year) | Listen cover | Listen original | OK? |", "|---|---|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            o = cat.loc[r["reponse_id"]]
            ou = fresh_preview_url(r["reponse_id"], prev.at[r["reponse_id"], "preview_url"] if r["reponse_id"] in prev.index else None,
                                   deezer_id=prev.at[r["reponse_id"], "deezer_id"] if r["reponse_id"] in prev.index else None)
            cu = fresh_preview_url(f"cover{r['cover_deezer_id']}", r["cover_preview_url"], deezer_id=r["cover_deezer_id"]) or r["cover_preview_url"]
            year = int(o.year) if pd.notna(o.year) else "?"
            lines.append(f"| {i} | {r['cover_title']} — {r['cover_artist']}, {r['kind']} | {o.title} — {o.artist}, {year} | [listen]({cu}) | [listen]({ou}) | |")
    out = PROJECT_ROOT / "eval" / "manual_checks.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written: {out} (Deezer listening links expire in under an hour; rerun the script to refresh them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
