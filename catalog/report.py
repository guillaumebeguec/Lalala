"""Rapport `eval/catalog.md` : taille, couverture, répartitions (PLAN.md phase 1, validation)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import PROJECT_ROOT


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def write_catalog_report(df: pd.DataFrame, path: Path | None = None, previews: pd.DataFrame | None = None) -> Path:
    path = path or (PROJECT_ROOT / "eval" / "catalog.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(df)
    src = df.sources.explode().value_counts()
    dec = (df.year // 10 * 10).value_counts().sort_index()
    lang = df.language.fillna("inconnue").value_counts()
    tot = df[df.is_tot_answer]
    lines = [
        "# Catalogue — rapport de phase 1",
        "",
        f"Généré depuis `data/catalog/catalog.parquet` ({n} morceaux, {int(df.in_dev.sum())} dans le sous-ensemble dev).",
        "",
        "## Taille et sources",
        "",
        "| Source | Morceaux |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in src.items()],
        "",
        "## Couverture des métadonnées",
        "",
        "| Champ | Couverture |", "|---|---|",
        f"| identifiant Deezer | {_pct(df.deezer_id.notna().mean())} |",
        f"| ISRC | {_pct(df.isrc.notna().mean())} |",
        f"| URL d'extrait (métadonnée Deezer) | {_pct(df.preview_url.notna().mean())} |",
        f"| année (métadonnée) | {_pct((df.year_source == 'meta').mean())} |",
        f"| année (métadonnée ou playlist) | {_pct(df.year.notna().mean())} |",
        f"| genre | {_pct(df.genre.notna().mean())} |",
        f"| langue | {_pct(df.language.notna().mean())} |",
        f"| MusicBrainz | {_pct(df.musicbrainz_id.notna().mean())} |",
        "",
    ]
    if previews is not None and len(previews):
        ok = previews.status.eq("ok").mean()
        lines += ["## Extraits audio", "", f"Extraits obtenus : {_pct(ok)} ({int(previews.status.eq('ok').sum())} / {len(previews)}).", ""]
        by = previews.groupby("provider").status.apply(lambda s: s.eq("ok").mean())
        lines += ["| Fournisseur | Taux |", "|---|---|", *[f"| {k} | {_pct(v)} |" for k, v in by.items()], ""]
    lines += [
        "## Réponses ToT_Music",
        "",
        f"{len(tot)} morceaux réponses distincts dans le catalogue, {_pct(tot.deezer_id.notna().mean())} avec identifiant Deezer courant.",
        "",
        "## Répartition par décennie",
        "",
        "| Décennie | Morceaux | dont ToT |", "|---|---|---|",
        *[f"| {int(d)}s | {int(c)} | {int(((tot.year // 10 * 10) == d).sum())} |" for d, c in dec.items()],
        f"| inconnue | {int(df.year.isna().sum())} | {int(tot.year.isna().sum())} |",
        "",
        "## Répartition par langue (métadonnée WASABI)",
        "",
        "| Langue | Morceaux |", "|---|---|",
        *[f"| {k} | {int(v)} |" for k, v in lang.head(12).items()],
        "",
        f"Morceaux issus des playlists France (`country_hint = FR`) : {int(df.country_hint.eq('FR').sum())}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
