"""T6b — Home-made hums: real recordings and synthetic hums.

- `data/manual/hums/<track_id>[_n].<m4a|wav|mp3|webm|ogg>`: real hums (microphone), produced
  by `scripts/record_queries.py --mode hum` or dropped in by hand;
- `data/manual/hums_synth/<track_id>[_n].wav`: synthetic hums rendered from the dominant
  melody of the preview (`scripts/synth_hums.py`). They exercise the melody pipeline without a
  human, but do not replace real hums (CHAD, T6a, remains the reference).

The file name carries the expected answer; the agent never sees these identifiers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import get_settings
from testsets.common import testsets_dir, write_jsonl

AUDIO_EXT = (".m4a", ".wav", ".mp3", ".webm", ".ogg", ".flac")


def manual_hums(catalog_ids: set[str] | None = None) -> list[dict]:
    """T6b rows from both folders; `catalog_ids` filters on known answers."""
    rows: list[dict] = []
    for sub, kind in (("hums", "real"), ("hums_synth", "synth")):
        d = get_settings().manual_dir / sub
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in AUDIO_EXT or p.name.startswith("."):
                continue
            tid = p.stem.split("_")[0]
            if catalog_ids is not None and tid not in catalog_ids:
                continue
            rows.append({"id": f"t6b_{kind}_{p.stem}", "reponse_id": tid, "kind": kind, "file": str(p)})
    return rows


def write_t6b(catalog: pd.DataFrame) -> Path:
    rows = manual_hums(set(catalog["track_id"]))
    out = testsets_dir() / "t6b_manual_hums.jsonl"
    write_jsonl(out, rows)
    return out
