"""Sélection du device PyTorch : MPS (Apple Silicon) en priorité, CPU en repli.

Tout le code qui place un tenseur ou un modèle sur un device passe par ici,
pour que le repli soit cohérent dans tout le projet.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device() -> "torch.device":  # noqa: F821 — import différé pour garder le module léger
    """Renvoie le device à utiliser : `mps` si disponible, sinon `cpu`.

    La variable d'environnement `SOUNDTRACE_DEVICE` force un device (utile pour
    les tests ou pour comparer MPS et CPU).
    """
    import torch

    forced = os.environ.get("SOUNDTRACE_DEVICE")
    if forced:
        logger.info("Device forcé par SOUNDTRACE_DEVICE=%s", forced)
        return torch.device(forced)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("Device : mps (Apple Silicon)")
        return torch.device("mps")

    logger.warning("MPS indisponible, repli sur le CPU")
    return torch.device("cpu")


def device_name() -> str:
    """Nom court du device, pour les logs et les rapports."""
    return str(get_device())
