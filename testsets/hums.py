"""T6b — Fredonnements « maison » : enregistrements réels et fredonnements synthétiques.

- `data/manual/hums/<track_id>[_n].<m4a|wav|mp3|webm|ogg>` : vrais fredonnements (micro), produits
  par `scripts/record_queries.py --mode hum` ou déposés à la main ;
- `data/manual/hums_synth/<track_id>[_n].wav` : fredonnements synthétiques rendus depuis la mélodie
  dominante de l'extrait (`scripts/synth_hums.py`). Ils testent la chaîne mélodique sans humain,
  mais ne remplacent pas des fredonnements réels (CHAD, T6a, reste la référence).

Le nom de fichier porte la réponse attendue ; l'agent ne voit jamais ces identifiants.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import get_settings
from testsets.common import testsets_dir, write_jsonl

AUDIO_EXT = (".m4a", ".wav", ".mp3", ".webm", ".ogg", ".flac")


def manual_hums(catalog_ids: set[str] | None = None) -> list[dict]:
    """Lignes T6b depuis les deux dossiers ; `catalog_ids` filtre sur les réponses connues."""
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
