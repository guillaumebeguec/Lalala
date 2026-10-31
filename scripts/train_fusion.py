"""Entraîne la fusion apprise (recommandation 4) sur des demandes ToT_Music des splits train/val
(réserve T1), évalue contre RRF sur l'échantillon d'évaluation T1 (split test), écrit eval/fusion.md.

    uv run python scripts/train_fusion.py --subset dev [--n-train 300] [--no-rewrite] [--llm]

Les listes par branche viennent de `Toolbox.retrieve` ; par défaut sans Llama-retriever (60 s par
demande sur un 8B) et avec la réécriture (cache). Rien de l'échantillon d'évaluation n'est vu à
l'entraînement.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from core.config import PROJECT_ROOT  # noqa: E402
from core.fusion import rrf  # noqa: E402
from core.fusion_learned import LearnedFusion  # noqa: E402
from core.metrics import bootstrap_ci, summarize  # noqa: E402
from finder.tools import Toolbox  # noqa: E402
from testsets.common import read_jsonl, testsets_dir  # noqa: E402

log = logging.getLogger("train_fusion")


def branch_lists(tb: Toolbox, text: str, use_rewrite: bool) -> tuple[dict, dict]:
    """Rejoue `retrieve` en conservant les listes par branche (sans seuils ni fusion)."""
    rw = tb.rewrite_query(text) if use_rewrite else None
    lists: dict[str, list] = {}
    bm = tb.bm25()
    if bm is not None:
        lists["bm25:brut"] = bm.search(text, k=50)
        if rw and rw.query:
            lists["bm25:réécrit"] = bm.search(rw.query + (" " + rw.query_fr if rw.query_fr else ""), k=50)
    if tb.use_llm_retriever:
        hits, _ = tb.llm_retrieve(text)
        if hits:
            lists["llama:connaissance"] = hits
    for name in tb.active_embedders():
        vt, va = tb.vec(name, "text"), tb.vec(name, "audio")
        if vt is None and va is None:
            continue
        emb = tb.emb(name)
        if getattr(emb, "languages", "") == "none":
            continue
        english_only = getattr(emb, "languages", "") == "en"
        q_text = rw.query if (english_only and rw and rw.query) else text
        if vt is not None:
            lists[f"{name}:texte→fiches"] = vt.search(emb.embed_text([q_text], task="query"), k=50)[0]
        if va is not None and rw is not None and rw.describes_sound:
            import numpy as np
            caps = (rw.captions or [rw.query]) + ([] if english_only else [text])
            V = emb.embed_text(caps, task="query")
            lists[f"{name}:légendes→audio"] = va.search(np.mean(V, axis=0, keepdims=True), k=50)[0]
    hints = {"decade": rw.decade, "language": rw.language} if rw else {}
    return lists, hints


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--llm", action="store_true", help="inclure la branche Llama-retriever (lent sur un 8B)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for n in ("httpx", "httpcore", "openai"):
        logging.getLogger(n).setLevel(logging.WARNING)
    tb = Toolbox(subset=args.subset, use_rerank=False)
    tb.use_llm_retriever = args.llm
    cat = cb.load(args.subset).set_index("track_id", drop=False)
    ids = set(cat.track_id)
    meta = {i: {"year": int(r.year) if r.year == r.year and r.year is not None else None, "language": r.language} for i, r in cat.iterrows()}
    d = testsets_dir()
    reserve = [c for c in read_jsonl(d / "t1_tot_reserve.jsonl") if c["reponse_id"] in ids and c.get("split_article") in ("train", "val")]
    test = [c for c in read_jsonl(d / "t1_tot_eval.jsonl") if c["reponse_id"] in ids]
    rng = random.Random(args.seed)
    rng.shuffle(reserve)
    train = reserve[: args.n_train]
    log.info("entraînement : %d demandes (train/val de l'article) ; test : %d", len(train), len(test))
    t0 = time.time()
    samples, branches = [], set()
    for i, c in enumerate(train, 1):
        lists, hints = branch_lists(tb, c["texte"], not args.no_rewrite)
        branches |= set(lists)
        samples.append((lists, hints, meta, "text", set(c.get("reponse_ids") or [c["reponse_id"]])))
        if i % 25 == 0:
            log.info("caractéristiques : %d / %d (%.1f s/demande)", i, len(train), (time.time() - t0) / i)
    lf = LearnedFusion(branches=sorted(branches))
    info = lf.fit(samples)
    lf.save()
    log.info("modèle : %s", info)
    # évaluation : RRF contre fusion apprise sur T1 test
    ranks_rrf, ranks_lf, ranks_best = [], [], []
    best_single: dict[str, list] = {b: [] for b in branches}
    for c in test:
        lists, hints = branch_lists(tb, c["texte"], not args.no_rewrite)
        answers = set(c.get("reponse_ids") or [c["reponse_id"]])
        def rank_of(hits):
            for k, (i, _) in enumerate(hits, 1):
                if i in answers:
                    return k
            return None
        ranks_rrf.append(rank_of(rrf(list(lists.values()), k=60)))
        ranks_lf.append(rank_of(lf.rank(lists, hints, meta)))
        for b in branches:
            best_single[b].append(rank_of(lists.get(b, [])))
    lines = [f"# Fusion apprise contre RRF — T1 test ({len(test)} demandes, sous-ensemble {args.subset})", "",
             f"Entraînement : {len(train)} demandes des splits train/val ; {info['n_pairs']} paires, {info['n_pos']} positives. Branches : {', '.join(sorted(branches))}.", "",
             "| Système | top-1 [IC 95 %] | top-5 | top-10 | MRR |", "|---|---|---|---|---|"]
    for name, rk in [("RRF k=60", ranks_rrf), ("fusion apprise (LR)", ranks_lf)] + [(f"seule : {b}", r) for b, r in best_single.items()]:
        m = summarize(rk); lo, hi = bootstrap_ci(rk, 1)
        lines.append(f"| {name} | {m['top1']:.3f} [{lo:.2f}, {hi:.2f}] | {m['top5']:.3f} | {m['top10']:.3f} | {m['mrr']:.3f} |")
    lines += ["", "Coefficients (régression logistique, caractéristiques standardisées par construction) :", "",
              "| Caractéristique | Coefficient |", "|---|---|", *[f"| {k} | {v} |" for k, v in sorted(info["coef"].items(), key=lambda kv: -abs(kv[1]))[:20]], ""]
    out = PROJECT_ROOT / "eval" / "fusion.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
