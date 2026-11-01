"""🔇 SAM Audio (Meta) — isolation d'une source par consigne texte, exécuté dans son propre venv.

Ce script est volontairement **indépendant de SoundTrace** : le paquet `sam-audio` dépend de
`perception_models`, qui installe un paquet Python nommé `core` — en conflit avec `core/` du projet.
Il tourne donc dans `data/raw/sam_audio/venv` (voir README § SAM Audio), lancé par `core/isolate.py`
en sous-processus :

    data/raw/sam_audio/venv/bin/python scripts/sam_runner.py --in mixed.wav --out target.wav [--prompt music]
        [--model facebook/sam-audio-base] [--device auto|mps|cpu] [--residual residual.wav] [--candidates 1]

Entrée : fichier audio (wav/mp3…) ; sortie : wav mono de la source isolée (fréquence du modèle).
Les poids sont à accès restreint (HF_TOKEN dans l'environnement). Écrit une ligne JSON de
synthèse sur stdout : {"sr": ..., "seconds": ..., "device": ..., "model": ...}.
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
    ap.add_argument("--candidates", type=int, default=1, help="candidats rerankés (1 = pas de reranking, plus rapide)")
    ap.add_argument("--spans", action="store_true", help="prédire les segments temporels (plus lent)")
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
    target = result.target.cpu()
    if target.dim() == 3:
        target = target[0]
    torchaudio.save(args.out, target, sr)
    if args.residual:
        residual = result.residual.cpu()
        if residual.dim() == 3:
            residual = residual[0]
        torchaudio.save(args.residual, residual, sr)
    print(json.dumps({"sr": sr, "seconds": round(t_sep, 1), "load_seconds": round(t_load, 1), "device": str(device),
                      "model": args.model, "shape": list(target.shape)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
