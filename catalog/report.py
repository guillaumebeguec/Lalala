"""`eval/catalog.md` report: size, coverage, distributions (PLAN.md phase 1, validation)."""

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
    lang = df.language.fillna("unknown").value_counts()
    tot = df[df.is_tot_answer]
    lines = [
        "# Catalogue — phase 1 report",
        "",
        f"Generated from `data/catalog/catalog.parquet` ({n} tracks, {int(df.in_dev.sum())} in the dev subset).",
        "",
        "## Size and sources",
        "",
        "| Source | Tracks |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in src.items()],
        "",
        "## Metadata coverage",
        "",
        "| Field | Coverage |", "|---|---|",
        f"| Deezer identifier | {_pct(df.deezer_id.notna().mean())} |",
        f"| ISRC | {_pct(df.isrc.notna().mean())} |",
        f"| preview URL (Deezer metadata) | {_pct(df.preview_url.notna().mean())} |",
        f"| year (metadata) | {_pct((df.year_source == 'meta').mean())} |",
        f"| year (metadata or playlist) | {_pct(df.year.notna().mean())} |",
        f"| genre | {_pct(df.genre.notna().mean())} |",
        f"| language | {_pct(df.language.notna().mean())} |",
        f"| MusicBrainz | {_pct(df.musicbrainz_id.notna().mean())} |",
        "",
    ]
    if previews is not None and len(previews):
        ok = previews.status.eq("ok").mean()
        lines += ["## Audio previews", "", f"Previews obtained: {_pct(ok)} ({int(previews.status.eq('ok').sum())} / {len(previews)}).", ""]
        by = previews.groupby("provider").status.apply(lambda s: s.eq("ok").mean())
        lines += ["| Provider | Rate |", "|---|---|", *[f"| {k} | {_pct(v)} |" for k, v in by.items()], ""]
        if "resolved_by" in previews.columns:
            okp = previews[previews.status == "ok"]
            tot_ids = set(df[df.is_tot_answer].track_id)
            lines += ["Preview resolution mode (C1):", "", "| Mode | All | ToT answers |", "|---|---|---|"]
            for mode in ("id", "isrc", "search", "itunes"):
                n_all = int((okp.resolved_by == mode).sum())
                n_tot = int((okp[okp.track_id.isin(tot_ids)].resolved_by == mode).sum())
                lines.append(f"| {mode} | {n_all} | {n_tot} |")
            n_none = int((previews.status != "ok").sum()); n_none_tot = int((previews[previews.track_id.isin(tot_ids)].status != "ok").sum())
            lines += [f"| not found | {n_none} | {n_none_tot} |", ""]
    lines += [
        "## ToT_Music answers",
        "",
        f"{len(tot)} distinct answer tracks in the catalogue, {_pct(tot.deezer_id.notna().mean())} with a current Deezer identifier.",
        "",
        "## Breakdown by decade",
        "",
        "| Decade | Tracks | of which ToT |", "|---|---|---|",
        *[f"| {int(d)}s | {int(c)} | {int(((tot.year // 10 * 10) == d).sum())} |" for d, c in dec.items()],
        f"| unknown | {int(df.year.isna().sum())} | {int(tot.year.isna().sum())} |",
        "",
        "## Breakdown by language (WASABI metadata)",
        "",
        "| Language | Tracks |", "|---|---|",
        *[f"| {k} | {int(v)} |" for k, v in lang.head(12).items()],
        "",
        f"Tracks from the France playlists (`country_hint = FR`): {int(df.country_hint.eq('FR').sum())}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
