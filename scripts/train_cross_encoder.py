"""🧪 Reranker cross-encoder maison (recommandation 7, option 2).

Entraîne un cross-encoder (requête ToT ↔ fiche morceau) sur les splits train/val de ToT_Music
(réserve T1), avec des **négatifs difficiles** tirés du top BM25 + Gemma de la fiche, perte
listwise (softmax sur chaque groupe requête : 1 positif + k négatifs). Évalue sur T1 test contre
la fusion seule et le reranking Llama. Modèle de départ multilingue léger (les souvenirs peuvent
être en français) : `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache 2.0).

    uv run python scripts/train_cross_encoder.py --subset dev [--epochs 2] [--negatives 7] [--base <modèle HF>]

Sortie : data/index/cross_encoder/ (poids) et eval/cross_encoder.md. Rien de l'échantillon
d'évaluation n'est vu à l'entraînement.
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
from core.config import PROJECT_ROOT, get_settings  # noqa: E402
from core.device import get_device  # noqa: E402
from core.fusion import rrf  # noqa: E402
from core.metrics import bootstrap_ci, summarize  # noqa: E402
from finder.tools import Toolbox  # noqa: E402
from testsets.common import read_jsonl, testsets_dir  # noqa: E402

log = logging.getLogger("train_ce")
DEFAULT_BASE = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def candidates_for(tb: Toolbox, text: str, k: int = 30) -> list[str]:
    """Candidats par BM25 + Gemma texte→fiches (sans Llama) : source des négatifs difficiles."""
    lists = []
    bm = tb.bm25()
    if bm is not None:
        lists.append(bm.search(text, k=k))
    vt = tb.vec("gemma", "text")
    if vt is not None:
        lists.append(vt.search(tb.emb("gemma").embed_text([text], task="query"), k=k)[0])
    return [i for i, _ in rrf(lists, k=60, top=k)] if lists else []


def short_card(cards: dict, tid: str, max_chars: int = 500) -> str:
    c = cards.get(tid, "")
    return c if len(c) <= max_chars else c[:max_chars].rsplit(" ", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--negatives", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for n in ("httpx", "httpcore", "openai"):
        logging.getLogger(n).setLevel(logging.WARNING)
    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    tb = Toolbox(subset=args.subset, use_rerank=False)
    tb.use_llm_retriever = False
    cards = tb.cards
    ids = set(cb.load(args.subset).track_id)
    d = testsets_dir()
    reserve = [c for c in read_jsonl(d / "t1_tot_reserve.jsonl") if c["reponse_id"] in ids and c.get("split_article") in ("train", "val")]
    test = [c for c in read_jsonl(d / "t1_tot_eval.jsonl") if c["reponse_id"] in ids]
    rng = random.Random(args.seed)
    rng.shuffle(reserve)
    train = reserve[: args.n_train]
    log.info("entraînement : %d demandes ; test : %d ; base %s", len(train), len(test), args.base)

    # ---- groupes listwise : (requête, [positif] + négatifs difficiles)
    groups = []
    t0 = time.time()
    for i, c in enumerate(train, 1):
        answers = set(c.get("reponse_ids") or [c["reponse_id"]])
        pos = next((a for a in answers if a in cards), None)
        if pos is None:
            continue
        negs = [t for t in candidates_for(tb, c["texte"]) if t not in answers and t in cards][: args.negatives]
        if len(negs) < 2:
            continue
        groups.append((c["texte"], pos, negs))
        if i % 50 == 0:
            log.info("groupes : %d / %d (%.1f s/demande)", len(groups), i, (time.time() - t0) / i)
    log.info("%d groupes d'entraînement", len(groups))

    # ---- entraînement (perte listwise : CrossEntropy sur les logits du groupe)
    device = str(get_device())  # SOUNDTRACE_DEVICE=cpu permet de laisser le GPU à un autre job
    model = CrossEncoder(args.base, num_labels=1, max_length=384, device=device)
    tok = model.tokenizer
    opt = torch.optim.AdamW(model.model.parameters(), lr=2e-5)
    model.model.train()
    for epoch in range(args.epochs):
        rng.shuffle(groups)
        tot, n = 0.0, 0
        for gi, (q, pos, negs) in enumerate(groups, 1):
            docs = [short_card(cards, pos)] + [short_card(cards, t) for t in negs]
            enc = tok([q] * len(docs), docs, padding=True, truncation=True, max_length=384, return_tensors="pt").to(device)
            logits = model.model(**enc).logits.view(-1)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))
            loss.backward()
            if gi % args.batch == 0:
                opt.step(); opt.zero_grad()
            tot += float(loss); n += 1
            if gi % 100 == 0:
                log.info("époque %d : %d / %d groupes, perte %.3f", epoch + 1, gi, len(groups), tot / n)
        opt.step(); opt.zero_grad()
        log.info("époque %d terminée : perte moyenne %.3f", epoch + 1, tot / max(n, 1))
    out_dir = get_settings().index_dir / "cross_encoder"
    model.save(str(out_dir))
    model.model.eval()

    # ---- évaluation sur T1 test : fusion seule contre cross-encoder sur le top-20
    ranks_fused, ranks_ce = [], []
    with torch.inference_mode():
        for c in test:
            answers = set(c.get("reponse_ids") or [c["reponse_id"]])
            cands = candidates_for(tb, c["texte"], k=20)
            ranks_fused.append(next((i for i, t in enumerate(cands, 1) if t in answers), None))
            if not cands:
                ranks_ce.append(None); continue
            docs = [short_card(cards, t) for t in cands]
            enc = tok([c["texte"]] * len(docs), docs, padding=True, truncation=True, max_length=384, return_tensors="pt").to(device)
            scores = model.model(**enc).logits.view(-1).cpu().numpy()
            order = [cands[j] for j in scores.argsort()[::-1]]
            ranks_ce.append(next((i for i, t in enumerate(order, 1) if t in answers), None))
    lines = [f"# Cross-encoder maison contre fusion — T1 test ({len(test)} demandes, sous-ensemble {args.subset})", "",
             f"Base : `{args.base}` ; {len(groups)} groupes (1 positif + ≤ {args.negatives} négatifs BM25/Gemma), {args.epochs} époques, perte listwise. Candidats rerankés : top-20 de BM25 + Gemma.", "",
             "| Système | top-1 [IC 95 %] | top-3 | top-5 | top-10 | MRR |", "|---|---|---|---|---|---|"]
    for name, rk in (("fusion BM25 + Gemma (top-20)", ranks_fused), ("+ cross-encoder maison", ranks_ce)):
        m = summarize(rk, ks=(1, 3, 5, 10)); lo, hi = bootstrap_ci(rk, 1)
        lines.append(f"| {name} | {m['top1']:.3f} [{lo:.2f}, {hi:.2f}] | {m['top3']:.3f} | {m['top5']:.3f} | {m['top10']:.3f} | {m['mrr']:.3f} |")
    out = PROJECT_ROOT / "eval" / "cross_encoder.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nRapport : {out} ; modèle : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
