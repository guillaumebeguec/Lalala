"""🔇 SAM Audio (Meta) via le portage MLX (`mlx-audio`, poids `mlx-community/sam-audio-base`), Apple Silicon natif.

Même contrat que `scripts/sam_runner.py` (venv `data/raw/sam_audio/venv_mlx`, lancé en sous-processus
par `core/isolate.py`) :

    data/raw/sam_audio/venv_mlx/bin/python scripts/sam_runner_mlx.py --in mixed.wav --out target.wav [--prompt music]
        [--model mlx-community/sam-audio-base] [--residual residual.wav] [--chunk 50]

Écrit une ligne JSON sur stdout : {"sr": ..., "seconds": ..., "load_seconds": ..., "model": ..., "peak_memory_gb": ...}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--residual", default=None)
    ap.add_argument("--prompt", default="music")
    ap.add_argument("--model", default="mlx-community/sam-audio-base")
    ap.add_argument("--chunk", type=int, default=50, help="ode_decode_chunk_size (50 = moins de mémoire)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from mlx_audio.sts.models.sam_audio.model import SAMAudio
    from mlx_audio.sts.models.sam_audio.processor import SAMAudioProcessor, save_audio

    t0 = time.time()
    processor = SAMAudioProcessor.from_pretrained(args.model)
    model = SAMAudio.from_pretrained(args.model)
    t_load = time.time() - t0
    batch = processor(descriptions=[args.prompt.lower()], audios=[args.inp])
    t1 = time.time()
    result = model.separate(batch.audios, batch.descriptions, sizes=batch.sizes, anchor_ids=batch.anchor_ids,
                            anchor_alignment=batch.anchor_alignment, audio_pad_mask=batch.audio_pad_mask,
                            ode_decode_chunk_size=args.chunk)
    t_sep = time.time() - t1
    sr = model.sample_rate
    n = int(batch.wav_sizes[0])
    save_audio(result.target[0][:n], args.out, sr)
    if args.residual:
        save_audio(result.residual[0][:n], args.residual, sr)
    print(json.dumps({"sr": sr, "seconds": round(t_sep, 1), "load_seconds": round(t_load, 1), "model": args.model,
                      "peak_memory_gb": round(float(getattr(result, "peak_memory", 0) or 0), 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
