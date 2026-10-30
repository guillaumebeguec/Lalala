"""Diapo résultats (PLAN.md phase 10.3) : tableau SVG généré depuis eval/retrieval.csv
(Lalala = meilleure fusion, contre BM25, embedding brut, Chromaprint) → eval/results_slide.svg.

    uv run python scripts/make_results_slide.py [--metric top5] [--out eval/results_slide.svg]

Sans dépendance graphique : SVG écrit à la main, lisible au projecteur (fond sombre, tokens du front).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import PROJECT_ROOT  # noqa: E402

COLORS = {"bg": "#15122B", "text": "#F7F3FF", "muted": "#B9B2D6", "accent": "#FF6FB5", "yellow": "#FFE066", "border": "#342D5C"}


def pick(rows: list[dict], metric: str) -> dict[str, dict[str, float]]:
    """{jeu: {colonne: valeur}} avec colonnes BM25 / meilleur embedding seul / Chromaprint / Lalala (meilleure fusion)."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    by_set: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_set[r["set"]].append(r)
    for s, rs in by_set.items():
        val = {r["system"]: float(r[metric]) for r in rs}
        if "bm25" in val:
            out[s]["BM25"] = val["bm25"]
        emb = {k: v for k, v in val.items() if ":" in k and not k.startswith("rrf")}
        if emb:
            k = max(emb, key=emb.get)
            out[s][f"Embedding seul ({k.split(':')[0]})"] = emb[k]
        if "chromaprint" in val:
            out[s]["Empreinte (Chromaprint)"] = val["chromaprint"]
        fus = {k: v for k, v in val.items() if k.startswith("rrf")}
        if fus:
            k = max(fus, key=fus.get)
            out[s]["Lalala (fusion)"] = fus[k]
    return out


def svg(table: dict[str, dict[str, float]], metric: str) -> str:
    cols = ["BM25", "Embedding seul", "Empreinte (Chromaprint)", "Lalala (fusion)"]
    def find(d, prefix):
        for k, v in d.items():
            if k.startswith(prefix):
                return v
        return None
    W, rowh, x0, y0 = 1200, 64, 60, 150
    colw = (W - x0 * 2 - 300) / len(cols)
    H = y0 + rowh * (len(table) + 1) + 80
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="DM Sans, Helvetica, Arial, sans-serif">',
             f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
             f'<text x="{x0}" y="70" font-size="40" font-weight="800" fill="{COLORS["text"]}">Lalala contre les baselines</text>',
             f'<text x="{x0}" y="105" font-size="18" fill="{COLORS["muted"]}">{metric} par jeu de test (sans agent) — fusion RRF des index texte et audio</text>']
    for j, c in enumerate(cols):
        parts.append(f'<text x="{x0 + 300 + j * colw + colw / 2}" y="{y0}" font-size="18" text-anchor="middle" fill="{COLORS["yellow" if c.startswith("Lalala") else "muted"]}">{c}</text>')
    parts.append(f'<line x1="{x0}" y1="{y0 + 14}" x2="{W - x0}" y2="{y0 + 14}" stroke="{COLORS["border"]}"/>')
    for i, (s, d) in enumerate(table.items()):
        y = y0 + rowh * (i + 1)
        parts.append(f'<text x="{x0}" y="{y}" font-size="22" font-weight="600" fill="{COLORS["text"]}">{s}</text>')
        vals = [find(d, c.split(" (")[0]) for c in cols]
        best = max((v for v in vals if v is not None), default=None)
        for j, v in enumerate(vals):
            if v is None:
                txt, color = "–", COLORS["muted"]
            else:
                txt, color = f"{100 * v:.0f} %", (COLORS["accent"] if v == best else COLORS["text"])
            parts.append(f'<text x="{x0 + 300 + j * colw + colw / 2}" y="{y}" font-size="26" font-weight="{700 if v == best else 400}" text-anchor="middle" fill="{color}">{txt}</text>')
        parts.append(f'<line x1="{x0}" y1="{y + 18}" x2="{W - x0}" y2="{y + 18}" stroke="{COLORS["border"]}" stroke-opacity=".5"/>')
    parts.append(f'<text x="{x0}" y="{H - 30}" font-size="14" fill="{COLORS["muted"]}">Jeux : ToT_Music (Bhargav et al., SIGIR 2023), MUSAN, reprises Deezer, CHAD. Catalogue de 24 k morceaux : chiffres non comparables au corpus WASABI complet.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="top5", choices=["top1", "top5", "top10", "mrr"])
    ap.add_argument("--csv", default=str(PROJECT_ROOT / "eval" / "retrieval.csv"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "eval" / "results_slide.svg"))
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    table = pick(rows, args.metric)
    Path(args.out).write_text(svg(table, args.metric), encoding="utf-8")
    print(f"Écrit : {args.out} ({len(table)} jeux)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
