"""Retrieval evaluation without the agent (PLAN.md phase 4.1) → eval/retrieval.md + csv.

    uv run python scripts/eval_retrieval.py --subset dev --sets t1,t3,t4 [--embedders clap,peav,gemma] [--limit N]

- T1, T2 (text → track): BM25 over the cards; text → cards and text → audio for each embedder.
  French queries (T2) are encoded as is; CLAP (English) will see them in phase 5
  through translation/rewriting.
- T3, T4 (audio → track): Chromaprint; audio → audio for each embedder. The query audio
  (Deezer covers, MUSAN mixes) is read in memory.
- RRF fusion (k ∈ {10, 30, 60}) of all the lists of a given query.
An embedder's scores are restricted to the tracks present in its index.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.previews import fetch_audio  # noqa: E402
from core.config import PROJECT_ROOT, get_settings  # noqa: E402
from core.embedders.base import get_embedder  # noqa: E402
from core.fingerprint import FingerprintIndex, fpcalc_raw  # noqa: E402
from core.fusion import rrf  # noqa: E402
from core.index import BM25Index, VectorIndex  # noqa: E402
from core.metrics import rank_of, summarize  # noqa: E402
from testsets.common import read_jsonl, testsets_dir  # noqa: E402

log = logging.getLogger("eval_retrieval")
K = 50
# English labels of the test sets in the report; the French names remain the keys of the sections
# (eval/retrieval_sections_*.json), of the CSV `set` column and of scripts/make_results_slide.py.
LABELS = {"T1 ToT_Music": "T1 ToT_Music", "T2 Souvenirs FR": "T2 FR memories", "T3 Bruit": "T3 Noise", "T4 Reprises": "T4 Covers",
          "T6a Fredonnements CHAD": "T6a CHAD hums", "T6b Fredonnements maison": "T6b Home hums"}


def _restrict(cases: list[dict], ids: set[str]) -> list[dict]:
    return [c for c in cases if c["reponse_id"] in ids]


def _rank(hits, case: dict) -> int | None:
    """C1: rank of the first acceptable recording (`reponse_ids`, else `reponse_id`)."""
    ok = set(case.get("reponse_ids") or [case["reponse_id"]])
    for i, (id_, _) in enumerate(hits, 1):
        if id_ in ok:
            return i
    return None


def eval_text_set(name: str, cases: list[dict], embedders: list[str], subset: str, bm25: BM25Index | None) -> dict[str, list]:
    runs: dict[str, list] = defaultdict(list)  # system → list of hits per case
    if bm25:
        for c in cases:
            runs["bm25"].append(bm25.search(c["texte"], k=K))
        # C1 / C6: LLM reformulations from the ToT_Music paper as a direct baseline (BM25 on `reform`)
        if any(c.get("reformulations") for c in cases):
            for c in cases:
                rf = (c.get("reformulations") or {}).get("reform") or (c.get("reformulations") or {}).get("reform_50")
                runs["bm25:reformulation-article"].append(bm25.search(rf, k=K) if rf else [])
    for name_e in embedders:
        emb = get_embedder(name_e)
        if getattr(emb, "languages", "") == "none":   # audio only (CoverHunter, hum2song)
            continue
        q = emb.embed_text([c["texte"] for c in cases], task="query")
        for mod in ("text", "audio"):
            if not VectorIndex.exists(name_e, mod):
                continue
            vi = VectorIndex.load(name_e, mod)
            for hits in vi.search(q, k=K):
                runs[f"{name_e}:text→{mod}"].append(hits)
    return runs


def _query_audio(case: dict, catalog_prev: pd.DataFrame | None) -> bytes | tuple[np.ndarray, int] | None:
    if case.get("file"):   # real recordings, CHAD hums (T6a)
        return Path(case["file"]).read_bytes()
    if "cover_preview_url" in case:
        return fetch_audio(f"cover{case['cover_deezer_id']}", case["cover_preview_url"], deezer_id=case["cover_deezer_id"])
    if "noise_parts" in case:
        from testsets.noisy import render, SR
        clean = fetch_audio(case["reponse_id"], catalog_prev.at[case["reponse_id"], "preview_url"], deezer_id=catalog_prev.at[case["reponse_id"], "deezer_id"])
        if not clean:
            return None
        mixed, _ = render(case, clean)
        return (mixed, SR)
    return None


def eval_audio_set(name: str, cases: list[dict], embedders: list[str], prev: pd.DataFrame, fp: FingerprintIndex | None, afp=None, subset: str = "dev") -> dict[str, list]:
    runs: dict[str, list] = defaultdict(list)
    audios = []
    kept = []
    for c in cases:
        try:
            a = _query_audio(c, prev)
        except Exception as e:  # noqa: BLE001
            log.warning("%s : %s", c["id"], e)
            a = None
        if a is not None:
            audios.append(a)
            kept.append(c)
    cases[:] = kept
    if fp:
        for a in audios:
            try:
                runs["chromaprint"].append(fp.search(fpcalc_raw(a), k=K))
            except Exception as e:  # noqa: BLE001
                log.warning("chromaprint : %s", e)
                runs["chromaprint"].append([])
    if afp is not None:
        for a in audios:
            try:
                runs["audfprint"].append([(tid, sc) for tid, sc, _ in afp.search(a, k=10)])
            except Exception as e:  # noqa: BLE001
                log.warning("audfprint : %s", e)
                runs["audfprint"].append([])
    from core.melody import ContourStore, contour_from_track, match
    if ContourStore.exists(subset) and ("hum2song" in embedders or "melody" in embedders):
        store = ContourStore.load(subset)
        from core.embedders.hum2song import Hum2SongEmbedder
        h = Hum2SongEmbedder(role="song")
        t0 = time.time()
        for a in audios:
            try:
                runs["melody"].append([(t, sc) for t, sc, _ in match(contour_from_track(h.pitch_track(a)), store, top=K)])
            except Exception as e:  # noqa: BLE001
                log.warning("melody: %s", e)
                runs["melody"].append([])
        log.info("melody: %d queries in %.0f s (%d contours)", len(audios), time.time() - t0, len(store))
    for name_e in embedders:
        if not VectorIndex.exists(name_e, "audio"):
            continue
        emb = get_embedder(name_e)
        t0 = time.time()
        q = emb.embed_audio(audios, batch_size=4)
        log.info("%s: %d audio queries in %.0f s", name_e, len(audios), time.time() - t0)
        vi = VectorIndex.load(name_e, "audio")
        for hits in vi.search(q, k=K):
            runs[f"{name_e}:audio→audio"].append(hits)
    return runs


def add_fusions(runs: dict[str, list], n: int) -> None:
    systems = [s for s in runs if ":" in s or s in ("bm25", "chromaprint", "audfprint")]
    if len(systems) < 2:
        return
    for k in (10, 30, 60):
        runs[f"rrf(k={k})"] = [rrf([runs[s][i] for s in systems], k=k, top=K) for i in range(n)]
    emb_only = [s for s in systems if ":" in s]
    if len(emb_only) >= 2:
        runs["rrf-embeddings(k=60)"] = [rrf([runs[s][i] for s in emb_only], k=60, top=K) for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--sets", default="t1,t2,t3,t4,t6a,t6b")
    ap.add_argument("--embedders", default=None, help="default: the active embedders from the config")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-fingerprint", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    s = get_settings()
    embedders = [e.strip() for e in (args.embedders or ",".join(s.active_embedders)).split(",") if e.strip()]
    cat = cb.load(args.subset)
    ids = set(cat.track_id)
    from catalog.previews import load_previews
    prev = load_previews().set_index("track_id")
    d = testsets_dir()
    sets = {x.strip() for x in args.sets.split(",")}
    bm25_path = s.index_dir / f"bm25_cards_{args.subset}.pkl"
    bm25 = BM25Index.load(bm25_path) if bm25_path.exists() else None
    fp = FingerprintIndex.load() if (FingerprintIndex.path().exists() and not args.no_fingerprint) else None
    from core.fingerprint_audfprint import AudfprintIndex
    afp = AudfprintIndex.load() if (AudfprintIndex.path().exists() and not args.no_fingerprint) else None
    header = [f"# Retrieval without the agent — {args.subset} subset", "",
              f"Catalogue: {len(cat)} tracks; embedders (last run): {', '.join(embedders)}; BM25: {'yes' if bm25 else 'no'}; Chromaprint: {'yes' if fp else 'no'}; audfprint: {'yes' if afp else 'no'}.", ""]
    sections_path = PROJECT_ROOT / "eval" / f"retrieval_sections_{args.subset}.json"
    sections: dict = json.loads(sections_path.read_text()) if sections_path.exists() else {}
    report: list[str] = []
    rows_csv = []
    examples = []
    # evaluated pool = subset + targets added to the indexes (--extra-ids of build_embeddings)
    for name_e in embedders:
        if VectorIndex.exists(name_e, "audio"):
            ids |= set(VectorIndex.load(name_e, "audio").ids)
    for name, fname, kind in (("T1 ToT_Music", "t1_tot_eval.jsonl", "text"), ("T2 Souvenirs FR", "t2_fr_memories.jsonl", "text"),
                              ("T3 Bruit", "t3_noisy.jsonl", "audio"), ("T4 Reprises", "t4_covers.jsonl", "audio"),
                              ("T6a Fredonnements CHAD", "t6a_chad_hummings.jsonl", "audio"),
                              ("T6b Fredonnements maison", "t6b_manual_hums.jsonl", "audio")):
        key = name.split()[0].lower()  # t1, t2, t3, t4, t6a, t6b
        if key not in sets or not (d / fname).exists():
            continue
        cases = _restrict(read_jsonl(d / fname), ids)
        if args.limit:
            cases = cases[: args.limit]
        if not cases:
            sections[name] = {"md": "\n".join([f"## {LABELS.get(name, name)}", "", "no case whose answer is in this subset.", ""]), "csv": []}
            continue
        t0 = time.time()
        report = []
        runs = eval_text_set(name, cases, embedders, args.subset, bm25) if kind == "text" else eval_audio_set(name, cases, embedders, prev, fp, afp, subset=args.subset)
        add_fusions(runs, len(cases))
        report += [f"## {LABELS.get(name, name)}", "", f"{len(cases)} cases; {time.time() - t0:.0f} s; systems: {', '.join(runs)}.", "", "| System | top-1 | top-5 | top-10 | MRR |", "|---|---|---|---|---|"]
        rows_set: list[dict] = []
        for sysname, lists in runs.items():
            ranks = [_rank(h, c) for h, c in zip(lists, cases)]
            m = summarize(ranks)
            report.append(f"| {sysname} | {m['top1']:.3f} | {m['top5']:.3f} | {m['top10']:.3f} | {m['mrr']:.3f} |")
            rows_set.append({"set": name, "system": sysname, **{k: round(v, 4) for k, v in m.items()}})
            if sysname.startswith("rrf(k=60)"):
                for h, c, r in list(zip(lists, cases, ranks))[:5]:
                    examples.append((name, c, r, h[:3]))
        # C1: breakdown by query type (T1)
        if kind == "text" and runs and any(c.get("query_types") for c in cases):
            best = max(runs, key=lambda k: summarize([_rank(h, c) for h, c in zip(runs[k], cases)])["top10"])
            by = defaultdict(list)
            for h, c in zip(runs[best], cases):
                for t in c.get("query_types") or ["?"]:
                    by[t].append(_rank(h, c))
            report += ["", f"Breakdown of `{best}` by query type (`query_types`):", "", "| Type | n | top-1 | top-5 | top-10 |", "|---|---|---|---|---|"]
            for g, rk in sorted(by.items()):
                m = summarize(rk)
                report.append(f"| {g} | {m['n']} | {m['top1']:.3f} | {m['top5']:.3f} | {m['top10']:.3f} |")
            if "bm25" in runs:
                report += ["", "Comparability (C6): the `bm25` row reproduces the paper's baseline (full query) on the ToT_Music test split, "
                           f"but on our catalogue of {len(ids)} tracks instead of the full WASABI corpus: the task is easier there, so the figures are not directly comparable."]
        # qualitative examples by SNR / type
        if kind == "audio" and runs:
            best = max(runs, key=lambda k: summarize([_rank(h, c) for h, c in zip(runs[k], cases)])["top5"])
            by = defaultdict(list)
            for h, c in zip(runs[best], cases):
                g = c.get("snr_db", c.get("kind"))
                if c.get("original_interval"):
                    a = c["original_interval"][0]
                    g = f"fragment {'at the start (< 60 s)' if a < 60 else 'in the middle (60–150 s)' if a < 150 else 'at the end (> 150 s)'} of the track"
                by[str(g)].append(_rank(h, c))
            report += ["", f"Breakdown of `{best}` by subgroup (SNR dB or type):", "", "| Group | n | top-1 | top-5 |", "|---|---|---|---|"]
            for g, rk in sorted(by.items(), key=str):
                m = summarize(rk)
                report.append(f"| {g} | {m['n']} | {m['top1']:.3f} | {m['top5']:.3f} |")
        report.append("")
        sections[name] = {"md": "\n".join(report), "csv": rows_set}
    sections_path.write_text(json.dumps(sections, ensure_ascii=False, indent=1), encoding="utf-8")
    order = ["T1 ToT_Music", "T2 Souvenirs FR", "T3 Bruit", "T4 Reprises", "T6a Fredonnements CHAD", "T6b Fredonnements maison"]
    report = header + [sections[n]["md"] for n in order if n in sections]
    rows_csv = [r for n in order if n in sections for r in sections[n]["csv"]]
    if examples:
        meta = cb.load("full").set_index("track_id")   # the indexed pool may exceed the subset (T6a targets)
        report += ["## Qualitative examples (RRF fusion k=60)", ""]
        for name, c, r, top in examples[:10]:
            ans = meta.loc[c["reponse_id"]]
            q = (c.get("texte") or c.get("cover_title") or c["id"])[:160].replace("\n", " ")
            tops = "; ".join(f"{meta.loc[t].title} — {meta.loc[t].artist}" if t in meta.index else t for t, _ in top)
            report += [f"- **{LABELS.get(name, name)}** “{q}” → expected *{ans.title} — {ans.artist}* (rank {r}); top-3: {tops}"]
        report.append("")
    out_md = PROJECT_ROOT / "eval" / "retrieval.md"
    out_md.write_text("\n".join(report), encoding="utf-8")
    with (PROJECT_ROOT / "eval" / "retrieval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["set", "system", "top1", "top5", "top10", "mrr", "n"])
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\nReport: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
