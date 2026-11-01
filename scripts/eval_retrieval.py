"""Évaluation de la recherche sans agent (PLAN.md phase 4.1) → eval/retrieval.md + csv.

    uv run python scripts/eval_retrieval.py --subset dev --sets t1,t3,t4 [--embedders clap,peav,gemma] [--limit N]

- T1, T2 (texte → morceau) : BM25 fiches ; texte → fiches et texte → audio pour chaque embedder.
  Les requêtes françaises (T2) sont encodées telles quelles ; CLAP (anglais) les verra en phase 5
  via la traduction/réécriture.
- T3, T4 (audio → morceau) : Chromaprint ; audio → audio pour chaque embedder. L'audio des
  requêtes (reprises Deezer, mixages MUSAN) est lu en mémoire.
- Fusion RRF (k ∈ {10, 30, 60}) de toutes les listes d'une même requête.
Les scores d'un embedder sont restreints aux morceaux présents dans son index.
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


def _restrict(cases: list[dict], ids: set[str]) -> list[dict]:
    return [c for c in cases if c["reponse_id"] in ids]


def _rank(hits, case: dict) -> int | None:
    """C1 : rang du premier enregistrement acceptable (`reponse_ids`, sinon `reponse_id`)."""
    ok = set(case.get("reponse_ids") or [case["reponse_id"]])
    for i, (id_, _) in enumerate(hits, 1):
        if id_ in ok:
            return i
    return None


def eval_text_set(name: str, cases: list[dict], embedders: list[str], subset: str, bm25: BM25Index | None) -> dict[str, list]:
    runs: dict[str, list] = defaultdict(list)  # système → liste de hits par cas
    if bm25:
        for c in cases:
            runs["bm25"].append(bm25.search(c["texte"], k=K))
        # C1 / C6 : reformulations LLM de l'article ToT_Music comme baseline directe (BM25 sur `reform`)
        if any(c.get("reformulations") for c in cases):
            for c in cases:
                rf = (c.get("reformulations") or {}).get("reform") or (c.get("reformulations") or {}).get("reform_50")
                runs["bm25:reformulation-article"].append(bm25.search(rf, k=K) if rf else [])
    for name_e in embedders:
        emb = get_embedder(name_e)
        if getattr(emb, "languages", "") == "none":   # audio seulement (CoverHunter, hum2song)
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
    if case.get("file"):   # enregistrements réels, fredonnements CHAD (T6a)
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
                log.warning("mélodie : %s", e)
                runs["melody"].append([])
        log.info("melody : %d requêtes en %.0f s (%d contours)", len(audios), time.time() - t0, len(store))
    for name_e in embedders:
        if not VectorIndex.exists(name_e, "audio"):
            continue
        emb = get_embedder(name_e)
        t0 = time.time()
        q = emb.embed_audio(audios, batch_size=4)
        log.info("%s : %d requêtes audio en %.0f s", name_e, len(audios), time.time() - t0)
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
    ap.add_argument("--embedders", default=None, help="défaut : embedders actifs de la config")
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
    header = [f"# Recherche sans agent — sous-ensemble {args.subset}", "",
              f"Catalogue : {len(cat)} morceaux ; embedders (dernier run) : {', '.join(embedders)} ; BM25 : {'oui' if bm25 else 'non'} ; Chromaprint : {'oui' if fp else 'non'} ; audfprint : {'oui' if afp else 'non'}.", ""]
    sections_path = PROJECT_ROOT / "eval" / f"retrieval_sections_{args.subset}.json"
    sections: dict = json.loads(sections_path.read_text()) if sections_path.exists() else {}
    report: list[str] = []
    rows_csv = []
    examples = []
    # le pool évalué = sous-ensemble + cibles ajoutées aux index (--extra-ids de build_embeddings)
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
            sections[name] = {"md": "\n".join([f"## {name}", "", "aucun cas dont la réponse est dans ce sous-ensemble.", ""]), "csv": []}
            continue
        t0 = time.time()
        report = []
        runs = eval_text_set(name, cases, embedders, args.subset, bm25) if kind == "text" else eval_audio_set(name, cases, embedders, prev, fp, afp, subset=args.subset)
        add_fusions(runs, len(cases))
        report += [f"## {name}", "", f"{len(cases)} cas ; {time.time() - t0:.0f} s ; systèmes : {', '.join(runs)}.", "", "| Système | top-1 | top-5 | top-10 | MRR |", "|---|---|---|---|---|"]
        rows_set: list[dict] = []
        for sysname, lists in runs.items():
            ranks = [_rank(h, c) for h, c in zip(lists, cases)]
            m = summarize(ranks)
            report.append(f"| {sysname} | {m['top1']:.3f} | {m['top5']:.3f} | {m['top10']:.3f} | {m['mrr']:.3f} |")
            rows_set.append({"set": name, "system": sysname, **{k: round(v, 4) for k, v in m.items()}})
            if sysname.startswith("rrf(k=60)"):
                for h, c, r in list(zip(lists, cases, ranks))[:5]:
                    examples.append((name, c, r, h[:3]))
        # C1 : découpage par type de demande (T1)
        if kind == "text" and runs and any(c.get("query_types") for c in cases):
            best = max(runs, key=lambda k: summarize([_rank(h, c) for h, c in zip(runs[k], cases)])["top10"])
            by = defaultdict(list)
            for h, c in zip(runs[best], cases):
                for t in c.get("query_types") or ["?"]:
                    by[t].append(_rank(h, c))
            report += ["", f"Détail de `{best}` par type de demande (`query_types`) :", "", "| Type | n | top-1 | top-5 | top-10 |", "|---|---|---|---|---|"]
            for g, rk in sorted(by.items()):
                m = summarize(rk)
                report.append(f"| {g} | {m['n']} | {m['top1']:.3f} | {m['top5']:.3f} | {m['top10']:.3f} |")
            if "bm25" in runs:
                report += ["", "Comparabilité (C6) : la ligne `bm25` reproduit la baseline de l'article (requête complète) sur le split test de ToT_Music, "
                           f"mais sur notre catalogue de {len(ids)} morceaux au lieu du corpus WASABI complet : la tâche y est plus facile, les chiffres ne sont pas directement comparables."]
        # exemples qualitatifs par SNR / type
        if kind == "audio" and runs:
            best = max(runs, key=lambda k: summarize([_rank(h, c) for h, c in zip(runs[k], cases)])["top5"])
            by = defaultdict(list)
            for h, c in zip(runs[best], cases):
                g = c.get("snr_db", c.get("kind"))
                if c.get("original_interval"):
                    a = c["original_interval"][0]
                    g = f"fragment en {'début (< 60 s)' if a < 60 else 'milieu (60–150 s)' if a < 150 else 'fin (> 150 s)'} du morceau"
                by[str(g)].append(_rank(h, c))
            report += ["", f"Détail de `{best}` par sous-groupe (SNR dB ou type) :", "", "| Groupe | n | top-1 | top-5 |", "|---|---|---|---|"]
            for g, rk in sorted(by.items(), key=str):
                m = summarize(rk)
                report.append(f"| {g} | {m['n']} | {m['top1']:.3f} | {m['top5']:.3f} |")
        report.append("")
        sections[name] = {"md": "\n".join(report), "csv": rows_set}
    sections_path.write_text(json.dumps(sections, ensure_ascii=False, indent=1), encoding="utf-8")
    order = ["T1 ToT_Music", "T2 Souvenirs FR", "T3 Bruit", "T4 Reprises", "T6a Fredonnements CHAD"]
    report = header + [sections[n]["md"] for n in order if n in sections]
    rows_csv = [r for n in order if n in sections for r in sections[n]["csv"]]
    if examples:
        meta = cb.load("full").set_index("track_id")   # le pool indexé peut dépasser le sous-ensemble (cibles T6a)
        report += ["## Exemples qualitatifs (fusion RRF k=60)", ""]
        for name, c, r, top in examples[:10]:
            ans = meta.loc[c["reponse_id"]]
            q = (c.get("texte") or c.get("cover_title") or c["id"])[:160].replace("\n", " ")
            tops = "; ".join(f"{meta.loc[t].title} — {meta.loc[t].artist}" if t in meta.index else t for t, _ in top)
            report += [f"- **{name}** « {q} » → attendu *{ans.title} — {ans.artist}* (rang {r}) ; top-3 : {tops}"]
        report.append("")
    out_md = PROJECT_ROOT / "eval" / "retrieval.md"
    out_md.write_text("\n".join(report), encoding="utf-8")
    with (PROJECT_ROOT / "eval" / "retrieval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["set", "system", "top1", "top5", "top10", "mrr", "n"])
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\nRapport : {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
