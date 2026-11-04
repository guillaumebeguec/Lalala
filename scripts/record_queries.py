"""🎙️ Makes real audio queries with the Mac's microphone (never committed).

Two modes:

- `rerecord`: plays a catalogue preview through the speakers and records at the same time with
  the microphone → `data/manual/real_recordings/<track_id>.m4a` ("real" T3: a true acoustic chain
  speaker → room → microphone, far more realistic than digitally added noise). Quiet room,
  medium volume, do nothing else during the take.

      uv run python scripts/record_queries.py --mode rerecord --n 20 --seconds 10 [--offset 8] [--subset dev]

- `hum`: shows title + artist, then records your humming → `data/manual/hums/<track_id>_<k>.m4a`
  (T6b). Enter = record, `s` = skip, `r` = redo the last take, `q` = quit.

      uv run python scripts/record_queries.py --mode hum --n 20 --seconds 12 [--ids dz123,dz456] [--known]

  `--known` only offers big-hit tracks (Deezer rank), easier to hum.

Prerequisites: ffmpeg with AVFoundation (brew), microphone permission granted to the terminal (macOS
asks on the first take). Then: `scripts/build_testsets.py --sets t3,t6b` then
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
    """Starts the microphone recording (AVFoundation) in the background; AAC mono 44.1 kHz."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "avfoundation", "-i", f":{device}",
           "-t", f"{seconds:.1f}", "-ac", "1", "-ar", "44100", "-c:a", "aac", "-b:a", "96k", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL)


def countdown(n: int = 3) -> None:
    for i in range(n, 0, -1):
        print(f"  {i}…", end=" ", flush=True)
        time.sleep(1)
    print("🔴 recording", flush=True)


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
    print(f"{len(rows)} previews to replay ({args.seconds} s each, offset {args.offset} s). Quiet room, medium volume.")
    input("Enter to start (Ctrl-C to cancel) ")
    done = 0
    for i, r in enumerate(rows, 1):
        target = out_dir / f"{r['track_id']}.m4a"
        if target.exists() and not args.overwrite:
            print(f"[{i}/{len(rows)}] {r['title']} — {r['artist']}: already recorded, skipped")
            continue
        data = fetch_audio(r["track_id"], r["preview_url"], deezer_id=r["deezer_id"])
        if not data:
            print(f"[{i}/{len(rows)}] {r['title']}: preview unavailable, skipped")
            continue
        x, sr = load_audio(data, sr=44_100)
        seg = crop(x, sr, args.offset, args.seconds)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(to_wav_bytes(seg, sr))
            tmp_path = Path(tmp.name)
        print(f"[{i}/{len(rows)}] {r['title']} — {r['artist']}")
        rec = ffmpeg_record(target, args.seconds + 1.0, args.device)
        time.sleep(0.6)                                    # let the microphone open before playback
        subprocess.run(["afplay", str(tmp_path)], check=False)
        rec.wait()
        tmp_path.unlink(missing_ok=True)
        if rec.returncode != 0 or not target.exists():
            print("  ⚠️ ffmpeg failed (microphone permission? device?) — see --device")
            return 1
        done += 1
        time.sleep(0.5)
    print(f"{done} recordings in {out_dir}")
    return 0


def mode_hum(args: argparse.Namespace) -> int:
    out_dir = get_settings().manual_dir / "hums"
    rows = pick_tracks(args.subset, args.n, args.seed, args.ids, args.known)
    print(f"{len(rows)} tracks to hum ({args.seconds} s per take). Enter = record, s = skip, r = redo, q = quit.")
    i, done = 0, 0
    while i < len(rows):
        r = rows[i]
        existing = sorted(out_dir.glob(f"{r['track_id']}_*.m4a"))
        print(f"\n[{i + 1}/{len(rows)}] 🎵 {r['title']} — {r['artist']} ({r.get('year') or '?'})" + (f"  [{len(existing)} take(s) already]" if existing else ""))
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
            print("  ⚠️ ffmpeg failed (microphone permission? device?) — see --device")
            return 1
        print(f"  ✅ {target.name}")
        done += 1
        i += 1
    print(f"{done} take(s) in {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["rerecord", "hum"], required=True)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=None, help="duration of a take (default: 10 s rerecord, 12 s hum)")
    ap.add_argument("--offset", type=float, default=8.0, help="rerecord: start of the replayed preview, in s")
    ap.add_argument("--ids", default=None, help="comma-separated track identifiers")
    ap.add_argument("--known", action="store_true", help="favour the most popular tracks")
    ap.add_argument("--device", type=int, default=0, help="AVFoundation index of the microphone (ffmpeg -f avfoundation -list_devices true -i '')")
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
