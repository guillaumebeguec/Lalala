"""Fige les jeux de test T1–T4 dans data/testsets/*.jsonl et écrit eval/testsets.md.

    uv run python scripts/build_testsets.py [--sets t1,t2,t3,t4] [--t4-hits 300]

Prérequis : catalogue construit et extraits résolus (scripts/resolve_previews.py).
T5 (vidéo) est fourni à la main dans data/manual/video/<track_id>.mp4.
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.previews import load_previews  # noqa: E402
from core.config import PROJECT_ROOT, get_settings  # noqa: E402
from testsets import covers, fr_memories, noisy, tot_music  # noqa: E402
from testsets.common import read_jsonl, testsets_dir, write_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="t1,t2,t3,t4")
    ap.add_argument("--t4-hits", type=int, default=300)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sets = {s.strip() for s in args.sets.split(",")}
    cat = cb.load("full")
    prev = load_previews()
    d = testsets_dir()
    report = ["# Jeux de test — rapport de phase 2", "", f"Catalogue : {len(cat)} morceaux, {int(prev.status.eq('ok').sum())} avec extrait.", ""]

    if "t1" in sets:
        ev, res = tot_music.build_t1(cat, prev)
        write_jsonl(d / "t1_tot_eval.jsonl", ev)
        write_jsonl(d / "t1_tot_reserve.jsonl", res)
        dec = collections.Counter(r["decennie"] for r in ev)
        qt = collections.Counter(t for r in ev for t in r["query_types"])
        report += ["## T1 — ToT_Music", "", f"{len(ev)} demandes d'évaluation, {len(res)} en réserve (réponse dans le catalogue avec extrait).", "",
                   "| Décennie | Éval |", "|---|---|", *[f"| {k} | {v} |" for k, v in sorted(dec.items())], "",
                   "Types de demande (éval) : " + ", ".join(f"{k} {v}" for k, v in qt.most_common()), ""]
    if "t2" in sets:
        out = fr_memories.write_t2(cat)
        rows = read_jsonl(out)
        report += ["## T2 — Souvenirs FR", "", f"{len(rows)} descriptions appariées au catalogue (fichier `data/manual/fr_memories.csv`).", ""]
    if "t3" in sets:
        out = noisy.write_t3(cat, prev)
        rows = read_jsonl(out)
        kinds = collections.Counter((r["kind"], r["snr_db"]) for r in rows)
        report += ["## T3 — Bruit", "", f"{len(rows)} cas (recette rejouable, audio mixé à la volée).", "",
                   "| Type | SNR | Cas |", "|---|---|---|", *[f"| {k[0]} | {k[1]} | {v} |" for k, v in sorted(kinds.items(), key=str)], ""]
    if "t4" in sets:
        out = covers.write_t4(cat, prev, n_hits=args.t4_hits)
        rows = read_jsonl(out)
        kinds = collections.Counter(r["kind"] for r in rows)
        report += ["## T4 — Reprises", "", f"{len(rows)} paires reprise → original ({len({r['reponse_id'] for r in rows})} originaux).", "",
                   "| Type | Paires |", "|---|---|", *[f"| {k} | {v} |" for k, v in kinds.most_common()], "",
                   "Contrôle manuel de 20 paires : à faire (voir `scripts/review_pairs.py`).", ""]
    vids = sorted((get_settings().manual_dir / "video").glob("*.mp4"))
    report += ["## T5 — Vidéo", "", f"{len(vids)} extraits dans `data/manual/video/`.", ""]
    path = PROJECT_ROOT / "eval" / "testsets.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nRapport : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
