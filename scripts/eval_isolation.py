"""📏 Contribution of SAM Audio (phase 9) on the T3 noisy recordings: fingerprint and similarity, with and without isolation.

    uv run python scripts/eval_isolation.py --subset dev [--max-snr 5] [--limit N] [--engine mlx]

For each synthetic T3 case whose answer is in the indexed pool: render the mixture (MUSAN recipe),
then (a) audfprint/Chromaprint fingerprint and Gemma rank on the mixture, (b) same on the music isolated by
SAM Audio (`core.isolate`, results cached). Writes `eval/sam_audio.md`. SI-SDR values are computed against
the clean preview (10 s, 16 kHz).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog.previews import fetch_audio, load_previews  # noqa: E402
from core.audio_io import load_audio, to_wav_bytes  # noqa: E402
from core.config import PROJECT_ROOT  # noqa: E402
from core.index import VectorIndex  # noqa: E402
from testsets.common import read_jsonl, testsets_dir  # noqa: E402

log = logging.getLogger("eval_isolation")


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est, ref = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    ref, est = ref - ref.mean(), est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-9)
    proj = a * ref
    return float(10 * np.log10(np.sum(proj ** 2) / (np.sum((est - proj) ** 2) + 1e-9)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--max-snr", type=float, default=5.0, help="keep only the cases with SNR ≤ this value (dB)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--engine", default=None, help="mlx | torch (default: SAM_AUDIO_ENGINE / auto)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.engine:
        os.environ["SAM_AUDIO_ENGINE"] = args.engine

    from core import isolate
    from finder.tools import Toolbox
    from testsets.noisy import SR, render

    eng = isolate.engine()
    if eng is None:
        log.error("no SAM Audio engine available (SAM_AUDIO_ENGINE, venvs)")
        return 1
    tb = Toolbox(subset=args.subset)
    pool = set(VectorIndex.load("gemma", "audio").ids) if VectorIndex.exists("gemma", "audio") else set(tb.catalog().index)
    cases = [c for c in read_jsonl(testsets_dir() / "t3_noisy.jsonl")
             if "noise_parts" in c and c.get("snr_db") is not None and c["snr_db"] <= args.max_snr and c["reponse_id"] in pool]
    if args.limit:
        cases = cases[: args.limit]
    prev = load_previews().set_index("track_id")
    log.info("%d T3 cases (SNR ≤ %.0f dB); engine %s", len(cases), args.max_snr, eng)

    rows = []
    t_iso = []
    for i, c in enumerate(cases, 1):
        tid = c["reponse_id"]
        clean = fetch_audio(tid, prev.at[tid, "preview_url"], deezer_id=prev.at[tid, "deezer_id"])
        if not clean:
            continue
        mixed, clean_x = render(c, clean)
        wav_mixed = to_wav_bytes(mixed, SR)
        t0 = time.time()
        iso = isolate.isolate_music(wav_mixed)
        if iso is None:
            log.warning("%s: isolation failed", c["id"])
            continue
        t_iso.append(iso[1].get("seconds") or (time.time() - t0))
        y, _ = load_audio(iso[0], sr=16_000)
        clean16 = load_audio(to_wav_bytes(clean_x, SR), sr=16_000)[0]
        mixed16 = load_audio(wav_mixed, sr=16_000)[0]
        row = {"id": c["id"], "snr_db": c["snr_db"], "noise": c["noise_parts"][0]["file"].split("/")[1] if c["noise_parts"] else "?",
               "sisdr_mixed": si_sdr(mixed16, clean16), "sisdr_iso": si_sdr(y, clean16)}
        for label, audio in (("mixed", wav_mixed), ("iso", iso[0])):
            ex = tb.fingerprint_match(audio)
            row[f"fp_{label}"] = bool(ex and ex.track_id == tid)
            ids = [k.track_id for k in tb.audio_similarity(audio, None, embedders=["gemma"])["candidates"]]
            row[f"gemma_{label}"] = ids.index(tid) + 1 if tid in ids else None
        rows.append(row)
        log.info("%d/%d %s: SI-SDR %.1f → %.1f dB; fingerprint %s → %s; gemma %s → %s", i, len(cases), c["id"], row["sisdr_mixed"],
                 row["sisdr_iso"], row["fp_mixed"], row["fp_iso"], row["gemma_mixed"], row["gemma_iso"])

    n = len(rows)
    if not n:
        return 1

    def rate(key):
        return 100 * sum(1 for r in rows if r[key]) / n

    def top(key, k):
        return 100 * sum(1 for r in rows if r[key] is not None and r[key] <= k) / n

    md = [f"# SAM Audio on T3 ({args.subset} subset, {eng} engine)", "",
          f"{n} synthetic cases (SNR ≤ {args.max_snr:.0f} dB), 10 s. Isolation: {np.mean(t_iso):.0f} s per case on average "
          f"(base model, Apple Silicon).", "",
          "| Measure | Mixture | Isolated music |", "|---|---|---|",
          f"| Mean SI-SDR vs clean preview | {np.mean([r['sisdr_mixed'] for r in rows]):.1f} dB | {np.mean([r['sisdr_iso'] for r in rows]):.1f} dB |",
          f"| Exact fingerprint found | {rate('fp_mixed'):.0f} % | {rate('fp_iso'):.0f} % |",
          f"| Fingerprint found (mixture **or** isolated) | | {100 * sum(1 for r in rows if r['fp_mixed'] or r['fp_iso']) / n:.0f} % |",
          f"| Gemma top-1 | {top('gemma_mixed', 1):.0f} % | {top('gemma_iso', 1):.0f} % |",
          f"| Gemma top-10 | {top('gemma_mixed', 10):.0f} % | {top('gemma_iso', 10):.0f} % |", "",
          "App policy: isolation only runs when the recording is detected as noisy and has no exact fingerprint match; "
          "the isolated audio gives the fingerprint a second chance, then feeds the similarity search.", "",
          "| Case | SNR | Noise | SI-SDR mix | SI-SDR iso | FP mix | FP iso | Gemma mix | Gemma iso |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['id']} | {r['snr_db']} | {r['noise']} | {r['sisdr_mixed']:.1f} | {r['sisdr_iso']:.1f} | {'✓' if r['fp_mixed'] else '–'} | "
                  f"{'✓' if r['fp_iso'] else '–'} | {r['gemma_mixed'] or '–'} | {r['gemma_iso'] or '–'} |")
    out = PROJECT_ROOT / "eval" / "sam_audio.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
