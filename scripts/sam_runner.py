"""🔇 SAM Audio (Meta) — isolation of a source from a text prompt, run in its own venv.

This script is deliberately **independent of Lalala**: the `sam-audio` package depends on
`perception_models`, which installs a Python package named `core` — clashing with the project's `core/`.
It therefore runs in `data/raw/sam_audio/venv` (see README § SAM Audio), launched by `core/isolate.py`
as a subprocess:

    data/raw/sam_audio/venv/bin/python scripts/sam_runner.py --in mixed.wav --out target.wav [--prompt music]
        [--model facebook/sam-audio-base] [--device auto|mps|cpu] [--residual residual.wav] [--candidates 1]

Input: audio file (wav/mp3…); output: mono wav of the isolated source (model sample rate).
The weights are gated (HF_TOKEN in the environment). Writes a one-line JSON
summary to stdout: {"sr": ..., "seconds": ..., "device": ..., "model": ...}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings


def pick_device(name: str):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--residual", default=None)
    ap.add_argument("--prompt", default="music")
    ap.add_argument("--model", default="facebook/sam-audio-base")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--candidates", type=int, default=1, help="reranked candidates (1 = no reranking, faster)")
    ap.add_argument("--spans", action="store_true", help="predict the time spans (slower)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    import torch
    import torchaudio
    from sam_audio import SAMAudio, SAMAudioProcessor

    t0 = time.time()
    device = pick_device(args.device)
    model = SAMAudio.from_pretrained(args.model)
    processor = SAMAudioProcessor.from_pretrained(args.model)
    model = model.eval().to(device)
    t_load = time.time() - t0
    batch = processor(audios=[args.inp], descriptions=[args.prompt.lower()]).to(device)
    t1 = time.time()
    with torch.inference_mode():
        result = model.separate(batch, predict_spans=args.spans, reranking_candidates=args.candidates)
    t_sep = time.time() - t1
    sr = processor.audio_sampling_rate
    def first(t):
        """`target`/`residual`: list of tensors (one per input) or a (B, 1, n) tensor depending on the version."""
        t = t[0] if isinstance(t, (list, tuple)) else t
        t = t.detach().cpu().float()
        while t.dim() > 2:
            t = t[0]
        return t if t.dim() == 2 else t.unsqueeze(0)

    target = first(result.target)
    torchaudio.save(args.out, target, sr)
    if args.residual:
        torchaudio.save(args.residual, first(result.residual), sr)
    print(json.dumps({"sr": sr, "seconds": round(t_sep, 1), "load_seconds": round(t_load, 1), "device": str(device),
                      "model": args.model, "shape": list(target.shape)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
