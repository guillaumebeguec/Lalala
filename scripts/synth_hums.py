"""🎶 Fredonnements synthétiques : mélodie dominante de l'extrait → voix fermée « mmm » imparfaite.

Pour N morceaux du sous-ensemble : suivi de hauteur RMVPE sur le mixage (même traceur que la
branche hum2song), choix de la fenêtre la plus « chantée » (saillance moyenne la plus haute),
puis resynthèse d'un fredonnement humanisé : transposition aléatoire (±3 demi-tons, la voix ne
chante pas dans la tonalité d'origine), tempo ±10 %, dérive lente de justesse (±30 cents), vibrato
(5,5 Hz, ±20 cents), attaques/relâchements de 40 ms, un peu de souffle. Sortie 16 kHz mono :
`data/manual/hums_synth/<track_id>_<k>.wav` (lu par T6b, voir `testsets/hums.py`).

    uv run python scripts/synth_hums.py --n 40 [--seconds 10] [--per-track 2] [--subset dev] [--seed 0]

Limites : la mélodie extraite d'un mixage est celle de la source la plus saillante (souvent la
voix, parfois un instrument) ; ce jeu mesure la robustesse de la chaîne mélodique aux déformations
humaines (justesse, tempo, tonalité), pas la mémoire mélodique d'un humain. CHAD (T6a) reste la
référence réelle.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.previews import fetch_audio, load_previews  # noqa: E402
from core.audio_io import to_wav_bytes  # noqa: E402
from core.config import get_settings  # noqa: E402

log = logging.getLogger("synth_hums")
SR = 16_000
HOP_S = 0.01   # trames RMVPE


def best_window(conf: np.ndarray, f0: np.ndarray, frames: int, min_conf: float) -> int:
    """Début (trame) de la fenêtre où la hauteur est la plus souvent sûre et dans la tessiture d'une voix."""
    voiced = ((conf > min_conf) & (f0 > 80) & (f0 < 1000)).astype(np.float32)
    if len(voiced) <= frames:
        return 0
    cs = np.concatenate([[0.0], np.cumsum(voiced)])
    score = cs[frames:] - cs[:-frames]
    return int(np.argmax(score))


def humanize(f0: np.ndarray, voiced: np.ndarray, rng: np.random.Generator, tempo: float, semitones: float,
             drift_cents: float = 30.0, vibrato_hz: float = 5.5, vibrato_cents: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Transpose, change le tempo (rééchantillonnage des trames) et ajoute dérive + vibrato."""
    n = len(f0)
    t_src = np.arange(n) * HOP_S
    t_dst = np.arange(0, t_src[-1] / tempo, HOP_S)
    f0_r = np.interp(t_dst * tempo, t_src, f0)
    v_r = np.interp(t_dst * tempo, t_src, voiced.astype(np.float32)) > 0.5
    # dérive lente : marche aléatoire lissée
    drift = np.cumsum(rng.normal(0, 1, len(t_dst)))
    drift = drift / (np.abs(drift).max() + 1e-6) * drift_cents
    vib = vibrato_cents * np.sin(2 * np.pi * vibrato_hz * t_dst + rng.uniform(0, 2 * np.pi))
    cents = semitones * 100 + drift + vib
    return f0_r * 2 ** (cents / 1200.0), v_r


def render(f0: np.ndarray, voiced: np.ndarray, rng: np.random.Generator, sr: int = SR, attack_s: float = 0.04) -> np.ndarray:
    """Synthèse « mmm » : harmoniques décroissantes (bouche fermée), enveloppe de voisement lissée, souffle léger."""
    n = int(len(f0) * HOP_S * sr)
    t = np.arange(n) / sr
    f0_s = np.interp(t, np.arange(len(f0)) * HOP_S, np.where(voiced, f0, np.nan))
    f0_s = np.nan_to_num(f0_s, nan=0.0)
    # on prolonge la dernière hauteur dans les trous pour éviter les glissements vers 0 pendant le relâchement
    last = 0.0
    for i in range(n):
        if f0_s[i] > 0:
            last = f0_s[i]
        else:
            f0_s[i] = last
    phase = 2 * np.pi * np.cumsum(f0_s) / sr
    amps = [1.0, 0.45, 0.2, 0.08, 0.04, 0.02]
    y = sum(a * np.sin(k * phase) for k, a in enumerate(amps, start=1))
    gate = np.interp(t, np.arange(len(voiced)) * HOP_S, voiced.astype(np.float32))
    k = max(1, int(attack_s * sr))
    env = np.convolve(gate, np.ones(k) / k, mode="same")
    breath = rng.normal(0, 1, n) * 0.01
    y = y * env * 0.3 + breath * (0.5 + 0.5 * env)
    return (y / max(np.abs(y).max(), 1e-6) * 0.7).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--per-track", type=int, default=1, help="variantes (transposition/tempo différents) par morceau")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--min-conf", type=float, default=0.3, help="saillance RMVPE minimale pour considérer une trame chantée")
    ap.add_argument("--min-voiced", type=float, default=0.5, help="part minimale de trames voisées dans la fenêtre, sinon morceau écarté")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from core.embedders.hum2song import Hum2SongEmbedder
    emb = Hum2SongEmbedder(role="song")
    out_dir = get_settings().manual_dir / "hums_synth"
    out_dir.mkdir(parents=True, exist_ok=True)
    cat = cb.load(args.subset)
    prev = load_previews()
    prev = prev[prev.status == "ok"].set_index("track_id")
    cat = cat[cat.track_id.isin(prev.index)]
    if args.ids:
        cat = cat[cat.track_id.isin([s.strip() for s in args.ids.split(",")])]
    rows = cat.to_dict("records")
    rng = np.random.default_rng(args.seed)
    random.Random(args.seed).shuffle(rows)
    frames = int(args.seconds / HOP_S)
    made, tried, t0 = 0, 0, time.time()
    for r in rows:
        if made >= args.n:
            break
        tried += 1
        data = fetch_audio(r["track_id"], prev.at[r["track_id"], "preview_url"], deezer_id=prev.at[r["track_id"], "deezer_id"])
        if not data:
            continue
        try:
            track = emb.pitch_track(data)
        except Exception as e:  # noqa: BLE001
            log.warning("%s : suivi de hauteur impossible (%s)", r["track_id"], e)
            continue
        f0, conf = track[0], track[1]
        s = best_window(conf, f0, frames, args.min_conf)
        f0w, cw = f0[s : s + frames], conf[s : s + frames]
        voiced = (cw > args.min_conf) & (f0w > 80) & (f0w < 1000)
        if voiced.mean() < args.min_voiced:
            log.info("%s écarté : %.0f %% de trames chantées dans la meilleure fenêtre", r["track_id"], 100 * voiced.mean())
            continue
        for k in range(1, args.per_track + 1):
            tempo = float(rng.uniform(0.9, 1.1))
            semis = float(rng.uniform(-3, 3))
            f0h, vh = humanize(f0w, voiced, rng, tempo, semis)
            y = render(f0h, vh, rng)
            (out_dir / f"{r['track_id']}_{k}.wav").write_bytes(to_wav_bytes(y, SR))
        made += 1
        log.info("%d/%d  %s — %s : fenêtre %.1f s, %.0f %% chanté, %d variante(s)", made, args.n, r["title"], r["artist"],
                 s * HOP_S, 100 * voiced.mean(), args.per_track)
    log.info("%d morceaux rendus (%d essayés) en %.0f s → %s", made, tried, time.time() - t0, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
