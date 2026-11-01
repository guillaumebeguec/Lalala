"""🎙️ Fabrique des requêtes audio réelles avec le micro du Mac (jamais commitées).

Deux modes :

- `rerecord` : joue un extrait de catalogue dans les haut-parleurs et enregistre en même temps avec
  le micro → `data/manual/real_recordings/<track_id>.m4a` (T3 « réel » : vraie chaîne acoustique
  enceinte → pièce → micro, bien plus réaliste que du bruit ajouté numériquement). Pièce calme,
  volume moyen, ne rien faire d'autre pendant la prise.

      uv run python scripts/record_queries.py --mode rerecord --n 20 --seconds 10 [--offset 8] [--subset dev]

- `hum` : affiche titre + artiste, puis enregistre votre fredonnement → `data/manual/hums/<track_id>_<k>.m4a`
  (T6b). Entrée = enregistrer, `s` = passer, `r` = refaire la dernière prise, `q` = quitter.

      uv run python scripts/record_queries.py --mode hum --n 20 --seconds 12 [--ids dz123,dz456] [--known]

  `--known` ne propose que des morceaux à gros succès (rang Deezer), plus faciles à fredonner.

Prérequis : ffmpeg avec AVFoundation (brew), autorisation micro accordée au terminal (macOS la
demande à la première prise). Ensuite : `scripts/build_testsets.py --sets t3,t6b` puis
`scripts/eval_retrieval.py --sets t3,t6b`.
"""

from __future__ import annotations

import argparse
import logging
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.previews import fetch_audio, load_previews  # noqa: E402
from core.audio_io import crop, load_audio, to_wav_bytes  # noqa: E402
from core.config import get_settings  # noqa: E402

log = logging.getLogger("record")


def ffmpeg_record(out: Path, seconds: float, device: int) -> subprocess.Popen:
    """Lance l'enregistrement micro (AVFoundation) en arrière-plan ; AAC mono 44,1 kHz."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "avfoundation", "-i", f":{device}",
           "-t", f"{seconds:.1f}", "-ac", "1", "-ar", "44100", "-c:a", "aac", "-b:a", "96k", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL)


def countdown(n: int = 3) -> None:
    for i in range(n, 0, -1):
        print(f"  {i}…", end=" ", flush=True)
        time.sleep(1)
    print("🔴 enregistrement", flush=True)


def pick_tracks(subset: str, n: int, seed: int, ids: list[str] | None, known: bool) -> list[dict]:
    cat = cb.load(subset)
    prev = load_previews()
    prev = prev[prev.status == "ok"].set_index("track_id")
    cat = cat[cat.track_id.isin(prev.index)]
    if ids:
        cat = cat[cat.track_id.isin(ids)]
    elif known and "rank" in prev.columns:
        ranked = prev.loc[cat.track_id, "rank"].fillna(0).astype(float)
        cat = cat.assign(_rank=ranked.values).sort_values("_rank", ascending=False).head(max(n * 5, 50))
    rows = cat.to_dict("records")
    random.Random(seed).shuffle(rows)
    rows = rows[:n] if not ids else rows
    for r in rows:
        r["preview_url"], r["deezer_id"] = prev.at[r["track_id"], "preview_url"], prev.at[r["track_id"], "deezer_id"]
    return rows


def mode_rerecord(args: argparse.Namespace) -> int:
    out_dir = get_settings().manual_dir / "real_recordings"
    rows = pick_tracks(args.subset, args.n, args.seed, args.ids, args.known)
    print(f"{len(rows)} extraits à rejouer ({args.seconds} s chacun, décalage {args.offset} s). Pièce calme, volume moyen.")
    input("Entrée pour commencer (Ctrl-C pour annuler) ")
    done = 0
    for i, r in enumerate(rows, 1):
        target = out_dir / f"{r['track_id']}.m4a"
        if target.exists() and not args.overwrite:
            print(f"[{i}/{len(rows)}] {r['title']} — {r['artist']} : déjà enregistré, passé")
            continue
        data = fetch_audio(r["track_id"], r["preview_url"], deezer_id=r["deezer_id"])
        if not data:
            print(f"[{i}/{len(rows)}] {r['title']} : extrait indisponible, passé")
            continue
        x, sr = load_audio(data, sr=44_100)
        seg = crop(x, sr, args.offset, args.seconds)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(to_wav_bytes(seg, sr))
            tmp_path = Path(tmp.name)
        print(f"[{i}/{len(rows)}] {r['title']} — {r['artist']}")
        rec = ffmpeg_record(target, args.seconds + 1.0, args.device)
        time.sleep(0.6)                                    # laisse le micro s'ouvrir avant la lecture
        subprocess.run(["afplay", str(tmp_path)], check=False)
        rec.wait()
        tmp_path.unlink(missing_ok=True)
        if rec.returncode != 0 or not target.exists():
            print("  ⚠️ échec ffmpeg (autorisation micro ? périphérique ?) — voir --device")
            return 1
        done += 1
        time.sleep(0.5)
    print(f"{done} enregistrements dans {out_dir}")
    return 0


def mode_hum(args: argparse.Namespace) -> int:
    out_dir = get_settings().manual_dir / "hums"
    rows = pick_tracks(args.subset, args.n, args.seed, args.ids, args.known)
    print(f"{len(rows)} morceaux à fredonner ({args.seconds} s par prise). Entrée = enregistrer, s = passer, r = refaire, q = quitter.")
    i, done = 0, 0
    while i < len(rows):
        r = rows[i]
        existing = sorted(out_dir.glob(f"{r['track_id']}_*.m4a"))
        print(f"\n[{i + 1}/{len(rows)}] 🎵 {r['title']} — {r['artist']} ({r.get('year') or '?'})" + (f"  [{len(existing)} prise(s) déjà]" if existing else ""))
        ans = input("  > ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            i += 1
            continue
        if ans == "r" and existing:
            existing[-1].unlink()
            existing = existing[:-1]
        target = out_dir / f"{r['track_id']}_{len(existing) + 1}.m4a"
        countdown(3)
        rec = ffmpeg_record(target, args.seconds, args.device)
        rec.wait()
        if rec.returncode != 0 or not target.exists():
            print("  ⚠️ échec ffmpeg (autorisation micro ? périphérique ?) — voir --device")
            return 1
        print(f"  ✅ {target.name}")
        done += 1
        i += 1
    print(f"{done} prise(s) dans {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["rerecord", "hum"], required=True)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=None, help="durée d'une prise (défaut : 10 s rerecord, 12 s hum)")
    ap.add_argument("--offset", type=float, default=8.0, help="rerecord : début de l'extrait rejoué, en s")
    ap.add_argument("--ids", default=None, help="identifiants de morceaux séparés par des virgules")
    ap.add_argument("--known", action="store_true", help="privilégier les morceaux les plus populaires")
    ap.add_argument("--device", type=int, default=0, help="index AVFoundation du micro (ffmpeg -f avfoundation -list_devices true -i '')")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    args.ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
    if args.seconds is None:
        args.seconds = 10.0 if args.mode == "rerecord" else 12.0
    return mode_rerecord(args) if args.mode == "rerecord" else mode_hum(args)


if __name__ == "__main__":
    raise SystemExit(main())
