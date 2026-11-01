"""📏 Apport de SAM Audio (phase 9) sur les enregistrements bruités T3 : empreinte et similarité, avec et sans isolation.

    uv run python scripts/eval_isolation.py --subset dev [--max-snr 5] [--limit N] [--engine mlx]

Pour chaque cas T3 synthétique dont la réponse est dans le pool indexé : rendu du mélange (recette MUSAN),
puis (a) empreinte audfprint/Chromaprint et rang Gemma sur le mélange, (b) idem sur la musique isolée par
SAM Audio (`core.isolate`, résultats en cache). Écrit `eval/sam_audio.md`. Les SI-SDR sont calculés contre
l'extrait propre (10 s, 16 kHz).
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
    ap.add_argument("--max-snr", type=float, default=5.0, help="ne garder que les cas à SNR ≤ cette valeur (dB)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--engine", default=None, help="mlx | torch (défaut : SAM_AUDIO_ENGINE / auto)")
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
        log.error("aucun moteur SAM Audio disponible (SAM_AUDIO_ENGINE, venvs)")
        return 1
    tb = Toolbox(subset=args.subset)
    pool = set(VectorIndex.load("gemma", "audio").ids) if VectorIndex.exists("gemma", "audio") else set(tb.catalog().index)
    cases = [c for c in read_jsonl(testsets_dir() / "t3_noisy.jsonl")
             if "noise_parts" in c and c.get("snr_db") is not None and c["snr_db"] <= args.max_snr and c["reponse_id"] in pool]
    if args.limit:
        cases = cases[: args.limit]
    prev = load_previews().set_index("track_id")
    log.info("%d cas T3 (SNR ≤ %.0f dB) ; moteur %s", len(cases), args.max_snr, eng)

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
            log.warning("%s : isolation impossible", c["id"])
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
        log.info("%d/%d %s : SI-SDR %.1f → %.1f dB ; empreinte %s → %s ; gemma %s → %s", i, len(cases), c["id"], row["sisdr_mixed"],
                 row["sisdr_iso"], row["fp_mixed"], row["fp_iso"], row["gemma_mixed"], row["gemma_iso"])

    n = len(rows)
    if not n:
        return 1

    def rate(key):
        return 100 * sum(1 for r in rows if r[key]) / n

    def top(key, k):
        return 100 * sum(1 for r in rows if r[key] is not None and r[key] <= k) / n

    md = [f"# SAM Audio sur T3 (sous-ensemble {args.subset}, moteur {eng})", "",
          f"{n} cas synthétiques (SNR ≤ {args.max_snr:.0f} dB), 10 s. Isolation : {np.mean(t_iso):.0f} s par cas en moyenne "
          f"(modèle base, Apple Silicon).", "",
          "| Mesure | Mélange | Musique isolée |", "|---|---|---|",
          f"| SI-SDR moyen vs extrait propre | {np.mean([r['sisdr_mixed'] for r in rows]):.1f} dB | {np.mean([r['sisdr_iso'] for r in rows]):.1f} dB |",
          f"| Empreinte exacte retrouvée | {rate('fp_mixed'):.0f} % | {rate('fp_iso'):.0f} % |",
          f"| Empreinte retrouvée (mélange **ou** isolé) | | {100 * sum(1 for r in rows if r['fp_mixed'] or r['fp_iso']) / n:.0f} % |",
          f"| Gemma top-1 | {top('gemma_mixed', 1):.0f} % | {top('gemma_iso', 1):.0f} % |",
          f"| Gemma top-10 | {top('gemma_mixed', 10):.0f} % | {top('gemma_iso', 10):.0f} % |", "",
          "Politique de l'app : l'isolation n'est lancée que si l'enregistrement est détecté bruité et sans empreinte exacte ; "
          "l'audio isolé donne une seconde chance à l'empreinte puis sert à la similarité.", "",
          "| Cas | SNR | Bruit | SI-SDR mél. | SI-SDR iso | Empr. mél. | Empr. iso | Gemma mél. | Gemma iso |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['id']} | {r['snr_db']} | {r['noise']} | {r['sisdr_mixed']:.1f} | {r['sisdr_iso']:.1f} | {'✓' if r['fp_mixed'] else '–'} | "
                  f"{'✓' if r['fp_iso'] else '–'} | {r['gemma_mixed'] or '–'} | {r['gemma_iso'] or '–'} |")
    out = PROJECT_ROOT / "eval" / "sam_audio.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Rapport : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
