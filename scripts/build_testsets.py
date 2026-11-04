"""Freezes the T1–T4 test sets into data/testsets/*.jsonl and writes eval/testsets.md.

    uv run python scripts/build_testsets.py [--sets t1,t2,t3,t4,t6a,t6b] [--t4-hits 300]

Prerequisites: catalogue built and previews resolved (scripts/resolve_previews.py).
T5 (video) is provided by hand in data/manual/video/<track_id>.mp4.
"""

from __future__ import annotations

import argparse
import collections
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.previews import load_previews  # noqa: E402
from core.config import PROJECT_ROOT, get_settings  # noqa: E402
from testsets import covers, fr_memories, noisy, tot_music  # noqa: E402
from testsets.common import read_jsonl, testsets_dir, write_jsonl  # noqa: E402


def merge_report(path: Path, new: str) -> str:
    """Merges the report: regenerated sections (`--sets`) replace the old ones, the others are kept
    (a partial run no longer wipes the report of the other sets)."""
    order = ["T1", "T2", "T3", "T4", "T6a", "T6b", "T5"]

    def split(text: str) -> tuple[str, dict[str, str]]:
        parts = re.split(r"(?m)^(?=## )", text)
        head, secs = parts[0], {}
        for p in parts[1:]:
            secs[p.split(" ")[1]] = p.rstrip() + "\n"
        return head, secs

    head, secs = split(new)
    if path.exists():
        _, old = split(path.read_text(encoding="utf-8"))
        for k, v in old.items():
            secs.setdefault(k, v)   # sections not regenerated: keep the old one
    keys = [k for k in order if k in secs] + sorted(k for k in secs if k not in order)
    return head + "\n".join(secs[k] for k in keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="t1,t2,t3,t4,t6a,t6b")
    ap.add_argument("--t4-hits", type=int, default=300)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sets = {s.strip() for s in args.sets.split(",")}
    cat = cb.load("full")
    prev = load_previews()
    d = testsets_dir()
    report = ["# Test sets — phase 2 report", "", f"Catalogue: {len(cat)} tracks, {int(prev.status.eq('ok').sum())} with a preview.", ""]

    if "t1" in sets:
        ev, res = tot_music.build_t1(cat, prev)
        write_jsonl(d / "t1_tot_eval.jsonl", ev)
        write_jsonl(d / "t1_tot_reserve.jsonl", res)
        dec = collections.Counter(r["decennie"] for r in ev)
        qt = collections.Counter(t for r in ev for t in r["query_types"])
        splits = collections.Counter(r["split_article"] for r in ev)
        multi = sum(1 for r in ev if len(r.get("reponse_ids", [])) > 1)
        # C1: `user_created` = queries that come with a user recording (the paper's
        # "Recording" label: vocaroo, etc.). We count those whose text still contains a link.
        from testsets import tot_music as tm
        all_q = tm.load(download_if_missing=False)
        uc = [q for q in all_q if "user_created" in q.query_types]
        uc_links = sum(1 for q in uc if "http" in (q.text or "").lower())
        report += ["## T1 — ToT_Music", "", f"{len(ev)} evaluation queries, {len(res)} in reserve (answer in the catalogue with a preview).",
                   f"Paper split within the eval sample: {dict(splits)} (C1: drawn from the `test` split). "
                   f"{multi} queries have several acceptable recordings (multiple ISRCs).", "",
                   "| Decade | Eval |", "|---|---|", *[f"| {k} | {v} |" for k, v in sorted(dec.items())], "",
                   "Query types (eval): " + ", ".join(f"{k} {v}" for k, v in qt.most_common()), "",
                   f"`user_created` (C1): {len(uc)} queries across all of ToT_Music; {uc_links} still contain a link to a user media file "
                   "(vocaroo…), whose accessibility was not checked (no download).", ""]
    if "t2" in sets:
        out = fr_memories.write_t2(cat)
        rows = read_jsonl(out)
        report += ["## T2 — FR memories", "", f"{len(rows)} descriptions matched to the catalogue (file `data/manual/fr_memories.csv`).", ""]
    if "t3" in sets:
        out = noisy.write_t3(cat, prev)
        rows = read_jsonl(out)
        kinds = collections.Counter((r["kind"], r["snr_db"]) for r in rows)
        report += ["## T3 — Noise", "", f"{len(rows)} cases (replayable recipe, audio mixed on the fly).", "",
                   "| Type | SNR | Cases |", "|---|---|---|", *[f"| {k[0]} | {k[1]} | {v} |" for k, v in sorted(kinds.items(), key=str)], ""]
    if "t4" in sets:
        out = covers.write_t4(cat, prev, n_hits=args.t4_hits)
        rows = read_jsonl(out)
        kinds = collections.Counter(r["kind"] for r in rows)
        report += ["## T4 — Covers", "", f"{len(rows)} cover → original pairs ({len({r['reponse_id'] for r in rows})} originals).", "",
                   "| Type | Pairs |", "|---|---|", *[f"| {k} | {v} |" for k, v in kinds.most_common()], "",
                   "Manual check of 20 pairs: to do (see `scripts/review_pairs.py`).", ""]
    if "t6a" in sets:
        from testsets import chad
        out, stats = chad.write_t6a(cat, prev)
        report += ["## T6a — CHAD hums", "", f"{stats['cases']} hums → original in the catalogue; CHAD groups {stats['groups']}, "
                   f"YouTube title read {stats['with_title']}, title/artist parsed {stats['parsed']}, in the catalogue {stats['in_catalog']}, "
                   f"with a preview {stats['with_preview']}. Limitation: position of the fragment in the track ≠ position of the Deezer preview (30 s).", ""]
    if "t6b" in sets:
        from testsets import hums
        out = hums.write_t6b(cat)
        rows = read_jsonl(out)
        kinds = collections.Counter(r["kind"] for r in rows)
        report += ["## T6b — Home hums", "", f"{len(rows)} hums (`data/manual/hums/` real, `data/manual/hums_synth/` synthetic): "
                   + ", ".join(f"{k} {v}" for k, v in kinds.most_common()) + ".", ""]
    vids = sorted((get_settings().manual_dir / "video").glob("*.mp4"))
    report += ["## T5 — Video", "", f"{len(vids)} clips in `data/manual/video/`.", ""]
    path = PROJECT_ROOT / "eval" / "testsets.md"
    path.write_text(merge_report(path, "\n".join(report)), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
